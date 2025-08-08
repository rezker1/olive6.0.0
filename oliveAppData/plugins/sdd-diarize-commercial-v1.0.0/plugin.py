import copy, glob, os, time, math, h5py, numpy as np, idento3, re, sys, datetime, importlib, torch

from olive.plugins import *

from idento3 import scoring

import pickle
import idento3 as idt
import idento3.engines as idte
import idento3.scenic_idento as scenic_idento

from nn import NET
from utils import mvn
from eval_buffer import FileLoad_EVAL


##################################################
# CONFIG - DO NOT CHANGE THESE UNLESS TOLD TO DO SO BY A DEVELOPER
default_config = idt.Config(dict(
# Segmentation-by-classification parameters
win_sec         = 4,
step_sec        = 2,
min_speech      = 0.2,
threshold       = 0.0,
max_class_per_frame = True,  # output_only_highest_scoring_detected_speaker if max_class_per_frame = 1. When 1, this means no overlap
max_class_per_file  = 4,  # Maximum speakers per file
unif_seg        = True,
unif_merge_gap  = 0.2,

# SAD
sad_threshold   = -0.5,
sad_filter      = 51,
sad_interpolate = 4,
sad_merge       = 0.3,
sad_padding     = 0.2,

# Diarization
enable_diarization = True,
enable_diarization_unknown_spk_output = True,
dia_fixed_unknown_speaker_score = 0.0,
dia_window         = 200,
dia_window_step    = 100,
dia_min_seg_frames = 30,
dia_threshold      = 1.5,
dia_max_clusters   = 10,
))
##################################################

### STATIC VARIABLES (not configurable on-the-fly)
static_precompute_enroll_stats = False # Used with PLDA for faster scoring - BUG when set to True for multi-file enrollment
static_abs_min_speech = 0.2
static_unknown_label = 'unknownspk'


# HELPER FUNCTION
def get_alignment_from_clusters_and_segarrays(speech, segarrays, indices):
    # Re-segmentation
    realign_idxs = np.zeros(speech.shape[0], dtype=int)
    prevseg, prevind = None, None
    for i, seg in enumerate(segarrays):
        # Assumiing segs are sequential, we can remove half of the measured overlap
        if (i == 0) or (indices[i] == prevind):
            realign_idxs[seg] = indices[i]
        else:
            # Change of index, Must find mid-point of change
            splitpoint = int(np.in1d(seg, prevseg).sum()/2)  # + seg[0]
            realign_idxs[seg[splitpoint:]] = indices[i]
        prevseg, prevind = seg, indices[i]
    realign_idxs[~speech] = 0
    realign_idxs = idt.condense_realign_idxs(realign_idxs)
    return realign_idxs

def get_indices_from_vectors(embeds, llr_thr=None, max_num_clusters=None):
    if len(embeds)==1:
        indices = np.array([1])
    else:
        import fastcluster, scipy
        Z = fastcluster.linkage(embeds, metric='cosine', method='ward', preserve_input=False)
#    Z = fastcluster.linkage(embeds[valid], metric='cosine', method='complete', preserve_input=False)
#    Z = fastcluster.linkage(embeds, metric='euclidean', method='complete', preserve_input=False)  # 45.855 ward
#    Z = fastcluster.linkage(embeds, metric='correlation', method='complete', preserve_input=False) # 37.597 complete
        if llr_thr is not None:
            indices = scipy.cluster.hierarchy.fcluster(Z, llr_thr, criterion='distance')
            if max_num_clusters is not None and len(np.unique(indices)) > max_num_clusters:
                logger.debug("Threshold for diarization [{}] resulted in more than max_num_clusters [{}]. Revising to limit.".format(llr_thr, max_num_clusters))
                indices = scipy.cluster.hierarchy.fcluster(Z, max_num_clusters, criterion='maxclust')
        else:
            indices = scipy.cluster.hierarchy.fcluster(Z, max_num_clusters, criterion='maxclust')
    return indices



#class CustomPlugin(Plugin, RegionScorer, ClassModifier, AudioVectorizer, ClassExporter):
class CustomPlugin(Plugin, RegionScorer, ClassModifier):

    def __init__(self):
        self.config   = default_config
        loader        = importlib.machinery.SourceFileLoader('plugin_config', os.path.join(os.path.dirname(os.path.realpath(__file__)), 'plugin_config.py'))
        spec          = importlib.util.spec_from_loader(loader.name, loader)
        mod           = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.config.update(mod.user_config)
        self.VALID_PARAMS  = ['region', 'speech_frames', 'speech_regions'] + list(default_config.keys()) # For checking user inputs and flagging unknown paramters. Region and channel are passed with 5-column enrollment

        self.task        = "SDD"
        self.label       = "SDD SmOlive Embed with Diarization (Commercial)"
        self.description = "Speaker embeddings framework for diarization with built-in speaker labeling"
        self.vendor = "SRI"
        self.version = '1.0.0'
        self.minimum_runtime_version = "5.7.1"
        self.minimum_olive_version = "5.7.1"
        self.create_date = "2021-9-20"
        self.revision_date = "2021-10-27"
        self.group = "Speaker"
        self.loaded_domains = []
        self.loaded_base = False


    ### Method implementations required of a Plugin ###
    def list_classes(self, domain_id, exclude_reserved=False):
        final_list = []
        if not exclude_reserved and self.config.enable_diarization and self.config.enable_diarization_unknown_spk_output:
            final_list = ['('+static_unknown_label+')']
        try:
            enrollments_dir = self.get_enrollment_storage(domain_id)
            final_list += [x for x in os.listdir(enrollments_dir) if x[0] != '.']
        except AttributeError:
            pass
        return final_list


    def load(self, domain_id):

        # Load the base components (domain-independemt)
        if not self.loaded_base:

            # Base modules: SAD
            sadextractor_config = dict(
                nnet_mvn=self.get_artifact("sad_dnn_mvn.npz"),
                nnet=self.get_artifact("sad_dnn_nnet.npz"),
                linearout=False,
                layer=-1,
                dnn_format='theano',
                dnn_input_config=self.get_artifact("sad_config.py"),
            )
            sadconfig = idt.Config()
            sadconfig.update(sadextractor_config)
            self.sad_engine = idt.Frontend(idt.TFDnnFeatures)(sadconfig)

            self.loaded_base = True

        # Domain dependent components
        if domain_id not in self.loaded_domains:
            domain = self.get_domains()[domain_id]

            # Feat extractor
            feature_config = idt.Config()
            feature_config.update_with_file(domain.get_artifact(os.path.join("embed","embed_config.py")))
            domain.evalfeatiter = FileLoad_EVAL(None, feature_config)

            # Embedding Extractor
            embed_layer = int(open(domain.get_artifact(os.path.join("embed","embed_layer.txt"))).readlines()[0].strip())
            if os.path.exists(domain.get_artifact(os.path.join("embed","embed_nn_int8.pnn"))) or os.path.exists(domain.get_artifact(os.path.join("embed","embed_nn_f32.pnn"))):
                nnet_int8_file = domain.get_artifact(os.path.join("embed","embed_nn_int8.pnn"))
                nnet_f32_file = domain.get_artifact(os.path.join("embed","embed_nn_f32.pnn"))
            else:
                nnet_int8_file = domain.get_artifact(os.path.join("embed","embed_nn_int8.npz"))
                nnet_f32_file = domain.get_artifact(os.path.join("embed","embed_nn_f32.npz"))

            try:
                domain.embed_nnet    = NET.load_from_file(nnet_int8_file)
                cal_model = domain.get_artifact("cal_int8.h5")
                plda_model = domain.get_artifact("lda_plda_int8.h5")
            except:
                domain.embed_nnet    = NET.load_from_file(nnet_f32_file)
                cal_model = domain.get_artifact("cal_f32.h5")
                plda_model = domain.get_artifact("lda_plda_f32.h5")

            embed_mvn     = np.load(domain.get_artifact(os.path.join("embed","embed_mvn.npz")))
            domain.embed_mu      = embed_mvn['mean']
            domain.embed_inv_std = embed_mvn['inv_std']

            # Trim to embeddings layer
            inds = np.where(np.array(domain.embed_nnet.layer_index)==embed_layer-1)[0]
            domain.embed_nnet.model = torch.nn.Sequential(*list(domain.embed_nnet.model.children())[:inds[0]+1])
            domain.embed_nnet.eval()

            # SID backend
            plda_dict, domain.lda, domain.mean = idt.read_data_in_hdf5(plda_model, nodes=['/PLDA', '/IvecTransform/LDA', '/IvecTransform/Mu'])
            domain.plda = idt.SPLDA.load_from_dict(plda_dict)
            if static_precompute_enroll_stats:
                segsperenrol = 1
                domain.plda.prepare_for_scoring(segsperenrol,1)

            # Calibration
            domain.cal_fusion_models = idt.read_data_in_hdf5(cal_model)
            domain.cal_offset = float(open(domain.get_artifact("cal.offset")).readlines()[0].strip())
            domain.sad_offset = float(open(domain.get_artifact("sad.offset")).readlines()[0].strip())

            # Load all enrolments on load and subsample during score_plda we also load SI
            domain.data_hash = -1000
            self.update_speaker_models(domain)

            self.loaded_domains.append(domain_id)

            logger.info("Loading of plugin '%s' domain '%s' complete." % (self.label, domain_id))

    def update_classes(self, domain_id):
        # Function that is called by the server after finalizing all enrollments
        domain = self.get_domains()[domain_id]
        self.update_speaker_models(domain)

    def update_speaker_models(self, domain, do_prep=True):
        # Called on load and finalize/remove class
        enrollment_dir = self.get_enrollment_storage(domain.get_id())

        speaker_ids = self.list_classes(domain.get_id(), exclude_reserved=True)
        domain.speaker_data = []
        domain.speaker_map = []
        for speaker_id in speaker_ids:
            speaker_dir = os.path.join(enrollment_dir, speaker_id)
            if static_unknown_label in speaker_id:
                self.escape_with_error("Can not use the reserved key '{}' in a user-enrolled class ID. Please remove or rename directory [{}].".format(static_unknown_label, speaker_dir))
            for session_id in os.listdir(speaker_dir):
                if session_id[0] == '.':
                    continue
                iv_path = os.path.join(enrollment_dir,speaker_id,session_id,session_id +'.vecs.h5')
                try:
                    vecs = idt.read_data_in_hdf5(iv_path)
                    for audioid in list(vecs.keys()):
                        audioid = str(audioid)
                        if len(vecs[audioid].shape) == 1:
                            embeds = vecs[audioid][np.newaxis,:]
                        else:
                            embeds = vecs[audioid]

                        domain.speaker_data.append(embeds)
                        domain.speaker_map.append(speaker_id)
                except:
                    self.escape_with_error("Corrupt enrollment file path [%s]. Please remove and try again" % os.path.dirname(iv_path))

        if len(domain.speaker_data) > 0:
            domain.speaker_data = np.vstack(domain.speaker_data)
            domain.speaker_data = idt.ivector_postprocessing(domain.speaker_data, domain.lda, domain.mean, lennorm=True)
            domain.speaker_map = np.array(domain.speaker_map)
            speakers, spk_map = np.unique(domain.speaker_map, return_inverse=True)
            domain.enroll_seg2model = np.vstack([np.arange(len(spk_map)), spk_map]).T
            domain.classes = speakers
            if static_precompute_enroll_stats and len(domain.enroll_seg2model)>0:
                segsperenrol = 1
                domain.T_Q1, domain.T_f = domain.plda.prepare_enrollments_for_scoring(domain.speaker_data, segsperenrol, 1, seg2model=domain.enroll_seg2model)

    def update_opts(self, opts, domain):

        # Copy values
        config = idt.Config(dict(self.config))

        if opts is not None:
            # Check that all passed options are valid for this plugin
            param_check = np.in1d(list(opts.keys()), self.VALID_PARAMS)
            if np.any(~param_check):
                raise Exception("Unknown parameter(s) passed [%s]. Please remove from the optional parameter list." % ','.join(np.array(list(opts.keys()))[param_check==False].tolist()))

            config.update(opts)

            # File-passed options are in in text format, so we need to convert these as necessary
            config.threshold       = float(config.threshold)
            config.min_speech      = float(config.min_speech)
            config.sad_threshold   = float(config.sad_threshold)
            config.sad_filter      = int(config.sad_filter)
            config.sad_interpolate = int(config.sad_interpolate)
            config.sad_merge       = float(config.sad_merge)
            config.sad_padding     = float(config.sad_padding)
            config.win_sec         = float(config.win_sec)
            config.step_sec        = float(config.step_sec)

            config.max_class_per_file    = int(config.max_class_per_file)
            config.unif_merge_gap        = float(config.unif_merge_gap)

            config.dia_threshold         = float(config.dia_threshold)
            config.dia_fixed_unknown_speaker_score         = float(config.dia_fixed_unknown_speaker_score)
            config.dia_window            = int(config.dia_window)
            config.dia_window_step       = int(config.dia_window_step)
            config.dia_min_seg_frames    = int(config.dia_min_seg_frames)
            config.dia_max_clusters      = int(config.dia_max_clusters)

            if type(config.max_class_per_frame) == str:
                config.max_class_per_frame  = True if config.max_class_per_frame == 'True' else False
            if type(config.unif_seg) == str:
                config.unif_seg  = True if config.unif_seg == 'True' else False
            if type(config.enable_diarization) == str:
                config.enable_diarization  = True if config.enable_diarization == 'True' else False
            if type(config.enable_diarization_unknown_spk_output) == str or type(config.enable_diarization_unknown_spk_output) == bytes:
                config.enable_diarization_unknown_spk_output    = True if config.enable_diarization_unknown_spk_output == 'True' else False

            if config.min_speech < static_abs_min_speech:
                raise Exception("The minimum allowable value for min_speech is %.2f. Please increase min_speech (currently %.2f)." % (static_abs_min_speech, config.min_speech))
            if config.win_sec <= config.step_sec:
                raise Exception("Variable win_sec must be larger than step_sec!")

            logger.debug("Using user-defined parameter options, new config is: %s" % config)

        return config

    def run_sad(self, audio, config, speech=None, suppress_error=False):
        if speech is None:
            # Overwrite embed system SAD config with the plugin specs. This can be updated on the fly with update_opts
            sad_opts = {'filter_length': config.sad_filter, 'interpolate':config.sad_interpolate, 'threshold':config.sad_threshold}
            speech   = idte.dnn_vad(self.sad_engine.feat_fc(audio), self.sad_engine.embedextractor,
                                    return_llrs=False, **sad_opts)

        # Check duration constraints
        prelim_duration = float(speech.sum())/idt.DEFAULT_FRAMES_PER_SECOND
        if prelim_duration < config.min_speech:
            if not suppress_error:
                aid = audio.id if audio.filename is None else audio.filename
                self.escape_with_error("Insufficient speech found for [%s]. Found %.3f which is less than the required amount of %.3f seconds." % (aid, prelim_duration, config.min_speech))
            else:
                logger.warn("Insufficient speech found for region. Found %.3f which is less than the required amount of %.3f seconds." % (prelim_duration, config.min_speech))
                return [], 0.0

        # Padding to merge close segments within self.config.sad_merge of each other
        merge, demerge = 0.0, 0.0
        if config.sad_merge > 0.0:
            merge = config.sad_merge / 2.0
            demerge = config.sad_merge / 2.0
            if config.sad_padding > 0.0:
                demerge -= config.sad_padding
        elif config.sad_padding > 0.0:
            merge = config.sad_padding

        if merge > 0.0:
            pad_speech = idt.pad(speech[np.newaxis, :], merge)
            if demerge > 0.0:
                speech = idt.pad(pad_speech, -demerge).ravel()
            else:
                speech = pad_speech.ravel()

        duration = float(speech.sum())/idt.DEFAULT_FRAMES_PER_SECOND

        return speech, duration

    def get_audio_vector(self, audio, speech, config, domain, return_aligns=False):

        # Extract the embedding
        infeats = domain.evalfeatiter(audio)
        infeats = idt.Feature.select_indices(infeats, speech)
        feats = mvn(infeats, domain.embed_mu, domain.embed_inv_std)
        feats = torch.tensor(feats).float()

        with torch.no_grad():
            logits, output = domain.embed_nnet(feats.unsqueeze_(0))
            embed = logits.numpy()

        if return_aligns:
            align_inds = [np.where(speech)[0]]
            return embed, align_inds
        else:
            return embed

    def get_audio_vectors(self, audio, speech, config, domain):

        # Extract the embedding
        infeats = domain.evalfeatiter(audio)
        infeats = idt.Feature.select_indices(infeats, speech)
        feats = mvn(infeats, domain.embed_mu, domain.embed_inv_std)
        feats = torch.tensor(feats).float()

        win, step      = int(idt.DEFAULT_FRAMES_PER_SECOND * config.win_sec), int(idt.DEFAULT_FRAMES_PER_SECOND * config.step_sec)
        embeds = []
        aligns = []

        for start in range(0, feats.shape[0]-step, step):
            feat_chunk = feats[start:min(start+win, feats.shape[0])]
            aligns.append([start, min(start+win, feats.shape[0])])
            with torch.no_grad():
                logits, output = domain.embed_nnet(feat_chunk.unsqueeze_(0))
                embeds.append(logits.numpy())

        # Now map the aligns to the speech indices
        speech_inds    = np.where(speech)[0]
        align_inds     = [speech_inds[st:en] for st,en in aligns]

        return embeds, align_inds

    ### RegionScorer ###
    def run_region_scoring(self, domain_id, audio, workspace, classes=None, opts=None):
        """
        Main scoring method
        Segmentation-by-Classification speaker detection approach
        """
        domain = self.get_domains()[domain_id]
        audio.make_mono()

        # Update options if they are passed in
        config = self.update_opts(opts, domain)

        # Check that speakers are enrolled
        available_classes = np.array(self.list_classes(domain_id, exclude_reserved=True))
        if classes is None:
            classes = available_classes
        else:
            non_existant = np.array(classes)[~np.in1d(classes, available_classes)]
            if len(non_existant) > 0:
                self.escape_with_error("%d speaker(s) requested that are not enrolled: [%s]" % (len(non_existant), ','.join(non_existant.tolist())))

        if len(classes) == 0 and (not config.enable_diarization or not config.enable_diarization_unknown_spk_output):
            # if still no classes for this domain, then fail
            self.escape_with_error("No speakers enrolled for domain '{}'. Please enroll at least one speaker.".format(domain_id))


        # Check for passed speech_frames and/or regions
        audio_frame_count = int(audio.duration*100)
        timing_map, passed_speech, regions, compiled_speech = None, None, None, np.zeros(audio_frame_count, dtype=bool)
        if opts is not None:
            if 'speech_regions' in opts and 'speech_frames' in opts:
                logger.warn("Both 'speech_regions' and 'speech_frames' passed. Using only speech_regions.")
            if 'region' in opts and 'speech_regions' in opts:
                logger.warn("Both 'region' and 'speech_regions' passed. Using only speech_regions.")
            if 'region' in opts:
                regions = opts['region']
            if 'speech_regions' in opts:
                regions = opts['speech_regions'].get_result()['speech']

        if regions is not None:
            if len(regions) == 0:
                return {self.task: []}  # Nothing to process
            timing_map = []
            speech_frame_count = 0
            for reg in regions:
                if reg[1] - reg[0] > config.min_speech:
                    timing_map.append((reg[0], reg[1]))
                    speech_frame_count += int((reg[1]-reg[0])*idt.DEFAULT_FRAMES_PER_SECOND)
                    logger.debug("Appending region {} to timing_map for processing. Speech_frame_count now {}".format(timing_map[-1], speech_frame_count))
                elif (timing_map[-1][1] - reg[0]) <= config.sad_merge:
                    speech_frame_count += int((reg[1]-timing_map[-1][1])*idt.DEFAULT_FRAMES_PER_SECOND)
                    timing_map[-1] = (timing_map[-1][0],reg[1])
                    logger.debug("Merged region {} to last timing_map entry for processing which is now {}. Speech_frame_count now {}".format(reg, timing_map[-1], speech_frame_count))
                else:
                    logger.warn("Region [{}, {}] shorter than minimum duration of {} seconds; moving to next region.".format(reg[0],reg[1], config.min_speech))
            if len(timing_map) == 0:
                logger.warn("No regions found longer that the minimum duration constraint")
                return {self.task: []}  # Nothing to process
            if opts is not None and 'speech_regions' in opts:
                passed_speech = np.ones(speech_frame_count+1, dtype=bool)
            audio_orig = audio.data.copy()  # Copy if chunking
        else:
            timing_map = [(0.0, audio.duration)]
            audio_orig = audio.data  # Just a pointer if processing the whole thing
        if opts is not None and passed_speech is None and 'speech_frames' in opts:
            speech_llrs = np.array(opts['speech_frames'].get_result()).ravel()
            passed_speech = speech_llrs >= config.sad_threshold

        embeds, aligns, spk_clst_ids, all_feats = [], [], [], []
        pscounter = 0
        # Get all the embeddings for the audio
        for reg in timing_map:
            region_offset = reg[0]
            frame_offset = int(region_offset*idt.DEFAULT_FRAMES_PER_SECOND)
            audio = idt.Wave(audio_orig.copy(), audio.sample_rate, audio.id)
            audio.trim_samples([reg])

            # Get speech if passed
            this_speech = None
            if passed_speech is not None:
                start, end = reg[0], reg[1]
                psspan = int(end*idt.DEFAULT_FRAMES_PER_SECOND - start*idt.DEFAULT_FRAMES_PER_SECOND)
                this_speech = passed_speech[pscounter:pscounter+psspan]
                pscounter += psspan
#                this_speech = passed_speech[int(start*idt.DEFAULT_FRAMES_PER_SECOND):int(end*idt.DEFAULT_FRAMES_PER_SECOND)]

            speech, duration = self.run_sad(audio, config, speech=this_speech, suppress_error=(regions is not None))
            if duration < config.min_speech:
                logger.warn("Region [{}, {}] shorter than minimum duration of {} seconds; moving to next region.".format(reg[0],reg[1], config.min_speech))
                continue

            # Add to compiled speech array
            compiled_speech[frame_offset:frame_offset+speech.shape[0]] = speech

            # Valid audio found, proceed
            if config.enable_diarization:
                segs = [(x,x+int(config.dia_window)) for x in np.arange(0,speech.shape[0]-int(config.dia_window),int(config.dia_window_step))]
                if len(segs) == 0:
                    segs = [(0, speech.shape[0]-2)]
                diff = speech.shape[0] - segs[-1][-1]
                if diff > 0:
                    if diff < config.dia_window_step:
                        # Extend the last seg
                        segs[-1] = (segs[-1][0], speech.shape[0]-2)
                    else:
                        # Add a new short seg
                        segs.append(((segs[-1][0] + config.dia_window_step), speech.shape[0]-2))

                segarrays = idt.get_segarrays(segs, speech, min_seg_frames=int(config.dia_min_seg_frames))
                infeats = domain.evalfeatiter(audio)
                feats = mvn(infeats, domain.embed_mu, domain.embed_inv_std)
                feats = torch.tensor(feats).float()
                for seg in segarrays:
                    with torch.no_grad():
                        logits, output = domain.embed_nnet(feats[seg].unsqueeze_(0))
                        embeds.append(logits.numpy())
                all_feats += [feats]
                align_inds = segarrays
            else:
                embed_arr, align_inds = self.get_audio_vectors(audio, speech, config, domain)
                if type(embed_arr) != list:
                    embed_arr = [embed_arr]
                embeds += embed_arr
            aligns += [x+frame_offset for x in align_inds]

        if len(embeds) == 0:
            self.escape_with_error("Insufficient continuous speech in any region to proceed.")
        embeds = np.vstack(embeds)
        feats = torch.tensor(np.vstack(all_feats)).float()

        speech = compiled_speech
        audio = idt.Wave(audio_orig.copy(), audio.sample_rate, audio.id)
        infeats = domain.evalfeatiter(audio)
        feats = mvn(infeats, domain.embed_mu, domain.embed_inv_std)
        feats = torch.tensor(feats).float()

        # Perform diarization if enabled
        if config.enable_diarization:
            indices = get_indices_from_vectors(embeds, llr_thr=config.dia_threshold, max_num_clusters=config.dia_max_clusters)
            realign_idxs = get_alignment_from_clusters_and_segarrays(speech, aligns, indices)
            num_clusters = realign_idxs.max()
            logger.info("NUM CLUSTERS {}".format(num_clusters))

        # Regather embeds and aligns with diariztion clusters
        if num_clusters == 1:
            aligns = [np.unique(np.hstack([x for x in aligns]))]
            with torch.no_grad():
                logits, output = domain.embed_nnet(idt.Feature.select_indices(feats, speech).unsqueeze_(0))
                embeds = logits.numpy()

        else:
            embeds, aligns = [], []
            for iclst in np.unique(realign_idxs):
                if iclst == 0:
                    continue  # non-speech
                seg = np.where(realign_idxs==iclst)[0]
                with torch.no_grad():
                    logits, output = domain.embed_nnet(idt.Feature.select_indices(feats, seg).unsqueeze_(0))
                    embeds.append(logits.numpy())

                aligns += [seg]
            embeds = np.vstack(embeds)

        if config.enable_diarization:
            test_ids = ["%s%02d" % (static_unknown_label, x) for x in np.arange(embeds.shape[0]).astype(int)]
        else:
            test_ids = ["%06d" % x for x in np.arange(embeds.shape[0]).astype(int)]

        if len(classes) > 0:
            # create the alignment key, ready for calibration
            align_key = idt.Key(domain.classes, test_ids, np.ones((len(domain.classes), len(test_ids)), dtype=np.int8))

            # Process then score the embeddings and calibrate scores
            embeds = idt.ivector_postprocessing(embeds, domain.lda, domain.mean, lennorm=True)
            if static_precompute_enroll_stats:
                scores = domain.plda.score_with_constantN_and_prepped_enrollment(embeds, domain.T_Q1, domain.T_f, 1, 1).T
            else:
                scores = idt.plda_verifier(domain.speaker_data, embeds, domain.plda, Tseg2model=domain.enroll_seg2model, tseg2model=None).T
            scores = idt.Scores(domain.classes, test_ids, scores.T)
            scores = idt.fusion.apply_calibration_and_fusion({'system0': scores}, align_key, domain.cal_fusion_models, sid=True)
            # Domain-specific cal offset
            scores.score_mat -= domain.cal_offset

            # Max scoring
            if config.max_class_per_frame:
                new_scores = (config.threshold - 1.0) * np.ones_like(scores.score_mat)
                max_idx = np.argmax(scores.score_mat,axis=0)
                new_scores[max_idx, np.arange(new_scores.shape[1])] = scores.score_mat[max_idx, np.arange(new_scores.shape[1])]
                scores.score_mat = new_scores

            # Filter to the requested subset of classes
            scores = scores.filter(classes, scores.test_ids)

            # Reduce the score matrix to the config.max_class_per_file classes based on max detection time.
            if config.max_class_per_file > 0 and config.max_class_per_file < len(classes):
                keep = scores.score_mat.mean(1).argsort()[-config.max_class_per_file:]
                scores = scores.filter(np.array(scores.train_ids)[keep], scores.test_ids)

            if config.enable_diarization and config.enable_diarization_unknown_spk_output:
                # Append unknown speaker clusters ready for idt.utils.get_region_detections_from_scores_with_frame_indices
                # This will make a square socre matrix
                dscores = idt.Scores(test_ids, test_ids, np.eye(len(test_ids))*(config.threshold+1.0))
                testid_to_add_as_unk = scores.score_mat.max(0) < config.threshold
                n_add = testid_to_add_as_unk.sum()
                add_ids = np.array(scores.test_ids)[:n_add].tolist()
                add_mat = np.ones([testid_to_add_as_unk.sum(), scores.score_mat.shape[1]]) * (config.threshold-1)
                add_mat[np.arange(testid_to_add_as_unk.sum()).astype(int), testid_to_add_as_unk] = config.dia_fixed_unknown_speaker_score
                scores = idt.Scores(scores.train_ids + add_ids, scores.test_ids, np.vstack([scores.score_mat, add_mat]))

        elif config.enable_diarization_unknown_spk_output:
            # Diarization case with no speakers
            scores = idt.Scores(test_ids, test_ids, np.ones([len(test_ids), len(test_ids)])*(config.threshold - 1.0))
            np.fill_diagonal(scores.score_mat, config.threshold + 1.0)

        # Process the scores and alignments to arrive at final outcome
        gap_to_merge = config.unif_merge_gap if config.unif_seg else 0.0
        final_scores = idt.utils.get_region_detections_from_scores_with_frame_indices(scores, aligns, threshold=config.threshold, region_min=config.min_speech, region_gap_to_merge=gap_to_merge, max_class_per_frame=config.max_class_per_frame)

        # Bug fix for float32 conversion, and re-mapping of unknown speaker labels
        sorted_final_scores = []
        spk_map = {}
        spk_cnt = 0
        for score in final_scores:
            if static_unknown_label in score[2] and score[2] not in spk_map.keys():
                new_score = (score[0], score[1], static_unknown_label+f"{spk_cnt:02}", config.dia_fixed_unknown_speaker_score)
                spk_map[score[2]] = static_unknown_label+f"{spk_cnt:02}"
                spk_cnt += 1
                sorted_final_scores.append(new_score)
            elif static_unknown_label in score[2] and score[2] in spk_map.keys():
                new_score = (score[0], score[1], spk_map[score[2]], config.dia_fixed_unknown_speaker_score)
                sorted_final_scores.append(new_score)
            else:
                sorted_final_scores.append(score)

        return {self.task: sorted_final_scores}

    ### ClassEnroller/ClassModifier ###
    def add_class_audio(self, domain_id, audio, class_id, enrollspace, opts=None):

        if static_unknown_label in class_id:
            self.escape_with_error("Can not use the reserved key '{}' in a user-enrolled class ID.".format(static_unknown_label))
        if len(class_id) == 0:
            self.escape_with_error("Please pass a valid class_id for enrollment (can not be empty string '')")

        domain = self.get_domains()[domain_id]
        config = self.update_opts(opts, domain)
        enrollment_dir = self.get_enrollment_storage(domain.get_id())

        if opts is not None and 'region' in opts:
            audio.trim_samples(opts['region'])

        duration = audio.duration

        pd_key = self.get_id() +'-'+ domain_id
        audio_dir = os.path.join(enrollspace, "staging", pd_key, audio.id)
        utils.mkdirs(audio_dir)

        # OPTIONAL: Add chunkng of enrollment data for calibration adaptation
        embed_filename = os.path.join(enrollment_dir, class_id, audio.id, audio.id + '.vecs.h5')
        out_embed_filename = os.path.join(audio_dir, audio.id + '.vecs.h5')
        if not os.path.exists(embed_filename):
            speech, duration = self.run_sad(audio, config)

            # Extract just one embedding per file enrolled
            embed = self.get_audio_vector(audio, speech, config, domain)
            embed_dict = dict([(audio.id + '_' + str(0),embed)])
            idt.save_dict_in_hdf5(out_embed_filename, embed_dict)
        else:
            shutil.copy(embed_filename, out_embed_filename)

        # Return location for the purpose of audiovectors (not needed for audio enrollment)
        return audio_dir, duration


    def remove_class_audio(self, domain_id, audio, class_id, enrollspace):

        removal_dir = os.path.join(enrollspace, "removals")

        utils.mkdirs(removal_dir)

        with open(os.path.join(removal_dir, audio.id), "w") as f:
            f.write("please")


    def finalize_class(self, domain_id, class_id, enrollspace):

        final_enrollment_dir = self.get_enrollment_storage(domain_id, class_id)

        removal_dir = os.path.join(enrollspace, "removals")
        if os.path.isdir(removal_dir):
            for file in os.listdir(removal_dir):
                target = os.path.join(final_enrollment_dir, file)
                if os.path.isdir(target):
                    shutil.rmtree(target)
            shutil.rmtree(removal_dir)

        pd_key = self.get_id() +'-'+ domain_id
        for file in glob.glob(os.path.join(enrollspace, "staging", pd_key, "*")):
            if len(os.listdir(file))>0:
                dest = os.path.join(final_enrollment_dir, os.path.basename(file))
                if os.path.exists(dest):
                    shutil.rmtree(dest)
                shutil.move(file, dest)
            else:
                logger.warn("Audio id [%s] for class_id [%s] failed to enroll" % (file,class_id))

    def remove_class(self, domain_id, class_id, workspace):
        if static_unknown_label in class_id:
            self.escape_with_error("Can not remove the reserved class key '{}'.".format(static_unknown_label))
        if len(class_id) == 0:
            self.escape_with_error("Please pass a valid class_id for unenrollment (can not be empty string '')")
        if class_id in self.list_classes(domain_id):
            class_dir = self.get_enrollment_storage(domain_id, class_id)
            shutil.rmtree(class_dir)
        else:
            self.escape_with_error("Can not locate requested class for unenrollment '{}'.".format(class_id))

    def get_region_scoring_opts(self):
        """
        These options are used in the OLIVE GUI and may be configured on the commandline by passing a file to --options
        """
        trait_options = [
            TraitOption('threshold', "Threshold for making a language detection", "Higher values result in more missed detections but less false alarms (default 2.2)", TraitType.CHOICE_TRAIT, list(range(-100, 100)), self.config.threshold),
            TraitOption('sad_threshold', "Threshold to determine speech frames", "Higher value results in less speech from processing (default 1.0)", TraitType.CHOICE_TRAIT, "", self.config.sad_threshold),
            TraitOption('min_speech', "Amount of speech needed to process audio", "Higher value results in less scores being output, but higher confidence (default 2.0)", TraitType.CHOICE_TRAIT, "", self.config.min_speech),
        ]
        return trait_options


# This line is very important! Every plugin should have one
plugin = CustomPlugin()
