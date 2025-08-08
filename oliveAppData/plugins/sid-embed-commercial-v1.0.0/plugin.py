import glob, os, numpy as np, idento3, re, time
import torch

from utils import mvn
from olive.plugins import *
from olive.plugins.trait import TraitType
from olive.plugins.trait import TraitOption

import importlib
import idento3 as idt
import idento3.engines as idte
from idento3.segmentation import NoSegmentationException, frame_to_seconds, Segmentation

from nn import NET

##################################################
# CONFIG - DO NOT CHANGE THESE UNLESS TOLD TO DO SO BY A DEVELOPER
default_config = idt.Config(dict(
# DETECTION OPTIONS
# SAD
sad_threshold   = 1.0,
sad_filter      = 31,
sad_interpolate = 4,
))
##################################################

DCMN_LIMIT_PER_SPK = 5 # prevents bias in DCMN from speakers with many enrollments
static_audio_length_before_chunking = 300  # seconds

class CustomPlugin(Plugin, GlobalScorer, ClassModifier, AudioVectorizer, ClassExporter):

    def __init__(self):
        self.task        = "SID"
        self.label       = "SID SmOlive Embeddings (Commercial)"
        self.description = "Speaker embeddings framework with automatic DCMN updating and a pytorch-based embedding extractor"
        self.vendor = "SRI"
        self.version = '1.0.0'
        self.minimum_runtime_version = "5.3.0"
        self.minimum_olive_version = "5.3.0"
        self.create_date = "2020-8-6"
        self.revision_date = "2021-09-15"
        self.group = "Speaker"
        self.loaded_domains = []
        self.loaded_base = False
        self.VALID_PARAMS    = ['region'] + list(default_config.keys()) # These are the valid keys to check that users don’t pass rubbish keys in opts.
        # Load the user_config in memory
        self.config   = default_config
        loader        = importlib.machinery.SourceFileLoader('plugin_config', os.path.join(os.path.dirname(os.path.realpath(__file__)), 'plugin_config.py'))
        spec          = importlib.util.spec_from_loader(loader.name, loader)
        mod           = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.config.update(mod.user_config)

    ### Method implementations required of a Plugin ###
    def list_classes(self, domain_id):
        try:
            enrollments_dir = self.get_enrollment_storage(domain_id)
            return [x for x in os.listdir(enrollments_dir) if x[0] != '.']
        except AttributeError:
            return []


    def load(self, domain_id):
        # Load the base components (domain-independemt)
        if not self.loaded_base:
            self.loaded_base = True

        # Domain dependent components
        if domain_id not in self.loaded_domains: #not self.base_loaded:
            domain = self.get_domains()[domain_id]
            sysdir = domain.get_artifact('embed')

            self.load_sid_engine(sysdir, domain)

            # Prep for holding model data
            domain.speaker_models = {}
            domain.speaker_meta = {}
            domain.train_tau = {}

            self.update_speaker_models(domain, do_prep=False)

            self.loaded_domains.append(domain_id)

            logger.info("Loading of plugin '%s' domain '%s' complete." % (self.label, domain_id))


    def load_sid_engine(self, sysdir, domain):
        # Determine the metadata extractors
        metaclasses = [re.sub('sgb.iv.h5.','',cls) for cls in os.listdir(sysdir) if 'sgb.iv.h5' in cls]
        metaclasses = ','.join(metaclasses) if len(metaclasses) > 0 else None
        domain_dir = domain.get_artifact('')

        # Embedding Extractor 
        domain.embed_mvn     = np.load(os.path.join(sysdir,"embed_mvn.npz"))
        embed_layer   = np.loadtxt(os.path.join(sysdir,"embed_layer.txt"), dtype=int).ravel()[0]

        if os.path.exists(domain.get_artifact(os.path.join(sysdir,"embed_nn_int8.pnn"))) or os.path.exists(domain.get_artifact(os.path.join(sysdir,"embed_nn_f32.pnn"))):
            nnet_int8_file = domain.get_artifact(os.path.join(sysdir,"embed_nn_int8.pnn"))
            nnet_f32_file = domain.get_artifact(os.path.join(sysdir,"embed_nn_f32.pnn"))
        else:
            nnet_int8_file = domain.get_artifact(os.path.join(sysdir,"embed_nn_int8.npz"))
            nnet_f32_file = domain.get_artifact(os.path.join(sysdir,"embed_nn_f32.npz"))

        try:
            domain.embed_nnet    = NET.load_from_file(nnet_int8_file)
            cal_model = domain.get_artifact("cal_int8.h5")
            plda_model = domain.get_artifact("lda_plda_int8.h5")
        except:
            domain.embed_nnet    = NET.load_from_file(nnet_f32_file)
            cal_model = domain.get_artifact("cal_f32.h5")
            plda_model = domain.get_artifact("lda_plda_f32.h5")

        # SAD
        sadextractor_config = dict(
            nnet_mvn=os.path.join(sysdir, "sad_dnn_mvn.npz"),
            nnet=os.path.join(sysdir, "sad_dnn_nnet.npz"),
            linearout=False,
            layer=-1,
            dnn_format='theano',
            dnn_input_config=os.path.join(sysdir, "sad_config.py"),
        )
        sadconfig = idt.Config()
        sadconfig.update(sadextractor_config)
        self.sad_engine = idt.Frontend(idt.TFDnnFeatures)(sadconfig)

        embedconfig = idt.Config()
        embedconfig.update_with_file(os.path.join(sysdir, "embed_config.py"))
        domain.feat_engine = idt.Frontend(embedconfig.featclass)(embedconfig)

        # Embedding Extractor
        embed_mvn     = np.load(os.path.join(sysdir, "embed_mvn.npz"))
        domain.embed_mu      = embed_mvn['mean']
        domain.embed_inv_std = embed_mvn['inv_std']
        embed_layer   = np.loadtxt(os.path.join(sysdir, "embed_layer.txt"), dtype=int).ravel()[0]

        # Trim to embeddings layer
        inds = np.where(np.array(domain.embed_nnet.layer_index)==embed_layer-1)[0]
        domain.embed_nnet.model = torch.nn.Sequential(*list(domain.embed_nnet.model.children())[:inds[0]+1])
        domain.embed_nnet.eval() 

        # Note that the SID engine contains the SAD tools for SAD
#        eng_config = idt.Config(dict(
#            feat_config=os.path.join(sysdir, "feat.config"),
#            sys_config=os.path.join(sysdir, "system.config"),
#            vad_dnn=os.path.join(sysdir, "sad_dnn_nnet.npz"),
#            vad_dnn_mvn=os.path.join(sysdir, "sad_dnn_mvn.npz"),
#            vad_type = "tensorflow",
#            backend_model=plda_model,
#            backend_model_metaextractor=os.path.join(domain_dir, "lda_plda.metaextractor.h5"),
#            metadata=os.path.join(sysdir, "sgb.iv.h5"),
#            metaclasses = metaclasses,
#            metaretention = True,
#        ))
#        kwargs = {}
#        kwargs['metaclasses'] = eng_config.metaclasses
#        domain.eng = idte.DnnPytorchSidEngine(eng_config, outdir=None, **kwargs)

        plda_dict, domain.lda, domain.mean = idt.read_data_in_hdf5(plda_model, nodes=['/PLDA', '/IvecTransform/LDA', '/IvecTransform/Mu'])
        if ['U'] in list(plda_dict.keys()):
            domain.plda = idt.JPLDA.load_from_dict(plda_dict)
            domain.is_jplda = True
        else:
            domain.plda = idt.SPLDA.load_from_dict(plda_dict)
            domain.is_jplda = False

        # The calibration model
        domain.calibration_model = idento3.read_data_in_hdf5(cal_model)

        # Overwrite embed system SAD config with the plugin specs. This can be updated on the fly with update_opts
        domain.vad_opts = {'filter_length': self.config.sad_filter, 'interpolate':self.config.sad_interpolate, 'threshold':self.config.sad_threshold}


    def update_classes(self, domain_id):
        # Function that is called by the server after finalizing all enrollments
        domain = self.get_domains()[domain_id]
        self.update_speaker_models(domain)


    def update_speaker_models(self, domain, do_prep=True):
        # Called on load and finalize/remove class
        enrollment_dir = self.get_enrollment_storage(domain.get_id())

        speaker_ids = self.list_classes(domain.get_id()) #, exclude_reserved=True)
        domain.speaker_data = []
        domain.speaker_map = []
        for speaker_id in speaker_ids:
            speaker_dir = os.path.join(enrollment_dir, speaker_id)
#            if static_unknown_label in speaker_id:
#                self.escape_with_error("Can not use the reserved key '{}' in a user-enrolled class ID. Please remove or rename directory [{}].".format(static_unknown_label, speaker_dir))
            for session_id in os.listdir(speaker_dir):
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
                    self.escape_with_error("Corrupt language audio file path [%s]. Please remove and try again" % os.path.dirname(iv_path))

        if len(domain.speaker_data) > 0:
            domain.speaker_data = np.vstack(domain.speaker_data)
            domain.speaker_data = idt.ivector_postprocessing(domain.speaker_data, domain.lda, domain.mean, lennorm=True)
            domain.speaker_map = np.array(domain.speaker_map)
            speakers, spk_map = np.unique(domain.speaker_map, return_inverse=True)
            domain.enroll_seg2model = np.vstack([np.arange(len(spk_map)), spk_map]).T
            domain.classes = speakers

    def xupdate_speaker_models(self, domain, do_prep=True):
        # Called on load and finalize/remove class
        enrollment_dir = self.get_enrollment_storage(domain.get_id())

        # Clear slate
        domain.speaker_models = {}
        domain.speaker_meta = {}

        speaker_ids = self.list_classes(domain.get_id())

        # Now go ahead and update the require speakers
        speaker_map = {}
        for speaker_id in speaker_ids:
            speaker_dir = os.path.join(enrollment_dir, speaker_id)
            for session_id in os.listdir(speaker_dir):
                iv_path = os.path.join(enrollment_dir,speaker_id,session_id,session_id +'.iv')
                try:
                    ivs, meta, audioid = idt.read_data_in_hdf5(iv_path, nodes=['/iv','/meta','/md5sum'])
                    
                    audioid = str(audioid.astype(str))
                    if speaker_id not in list(speaker_map.keys()):
                        speaker_map[speaker_id] = []
                    speaker_map[speaker_id].append(session_id)

                except:
                    raise Exception("Corrupt speaker audio file path [%s]. Please remove and try again" % os.path.dirname(iv_path))

        # Enroll all the speaker models and gather calibration meta
        if len(list(speaker_map.keys()))>0:
            for speaker_id,session_ids in speaker_map.items():
                speaker_ivs = [domain.eng.ivs[x] for x in session_ids]
                domain.speaker_models[speaker_id] = idte.IVectorModel(speaker_id, session_ids, np.r_[[iv[0] for iv in speaker_ivs if iv is not None]])
                domain.speaker_meta[speaker_id] = np.atleast_2d(np.sum([domain.eng.retained_meta_dict[x]['speech_duration'] for x in session_ids]))


    def fetch_audio_vector(self, domain, iv_storage, logid, audioid=None):
        # Additionally source the tst ivs
        if iv_storage is not None and os.path.exists(iv_storage):
            eng_iv_keys = list(domain.eng.ivs.keys())

            iv_path     = os.path.join(iv_storage, logid, logid + '.iv')
            if os.path.exists(iv_path):
                try:
                    ivs, meta, audioid = idt.read_data_in_hdf5(iv_path, nodes=['/iv','/meta','/md5sum'])
                except:
                    raise Exception("Corrupt audio file path [%s]. Please remove and try again" % os.path.dirname(iv_path))
                audioid = str(audioid)
                if audioid not in eng_iv_keys:
                    domain.eng.ivs.update(dict(list(zip([audioid],ivs[:,np.newaxis,:]))))
                    domain.eng.retained_meta_dict[audioid] = {'speech_duration': meta[0][np.newaxis,:]}


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
            config.sad_threshold   = float(config.sad_threshold)
            config.sad_filter      = int(config.sad_filter)
            config.sad_interpolate = int(config.sad_interpolate)

            logger.debug("Using user-defined parameter options, new config is: %s" % config)

        return config

    def run_sad(self, audio, config):

        # Detect speech frames
        waveobject = idt.Wave(audio.data.copy(), audio.sample_rate, audio.id)

        # Overwrite embed system SAD config with the plugin specs
        sad_opts = {'filter_length': config.sad_filter, 'interpolate':config.sad_interpolate, 'threshold':config.sad_threshold}
        speech   = idte.dnn_vad(self.sad_engine.feat_fc(audio), self.sad_engine.embedextractor,
                                return_llrs=False, **sad_opts)

        if speech.sum() == 0:
            raise Exception("In sufficient speech to process file")

        return speech, speech.sum()/100.


    def get_audio_vector(self, domain, audio, speech, config):

        # Extract the embedding
        embeds = []
        if audio.duration > static_audio_length_before_chunking:
            chunk_cnt  = np.ceil(audio.duration/static_audio_length_before_chunking).astype(int)
            chunk_size = audio.duration / chunk_cnt
            chunks = [(x*chunk_size, min(audio.duration, x*chunk_size+chunk_size)) for x in np.arange(chunk_cnt).astype(int)]
        else:
            chunks = [(0, audio.duration)]

        for st,en in chunks:
            tmpaudio = idt.Wave(audio.data, audio.sample_rate)
            tmpaudio.trim_samples([(st, en)])
            tmpspeech = speech[int(st*100):int(en*100)]
            feats = domain.feat_engine(tmpaudio, indices=tmpspeech)
            feats = mvn(feats, domain.embed_mu, domain.embed_inv_std)
            feats = torch.tensor(feats).float()

            with torch.no_grad():
                logits, output = domain.embed_nnet(feats.unsqueeze_(0))
                embed = logits.numpy()
            embeds.append(embed)
        embed = np.mean(embeds,0)

        return embed


    def dump_vector_to_disk(self, iv, duration, iv_storage, audio):
        #logid = os.path.basename(os.path.splitext(audio.filename)[0])
        logid = os.path.basename(os.path.splitext(audio.id)[0])
        storage_dir = os.path.join(iv_storage, logid)
        if not os.path.exists(os.path.join(storage_dir,logid)+'.iv'):
            idt.save_vectors_in_dir([{'iv':iv, 'meta':duration, 'md5sum':audio.id}], [logid], storage_dir, ext=".iv", with_meta=True)
            name_fh = open(os.path.join(storage_dir, "audio_filename"), 'w')
            name_fh.write("%s\n" % audio.filename)
            name_fh.close()


    ### GlobalScorer ###
    def run_global_scoring(self, domain_id, audio, workspace, classes=None, opts=None):
        """
        Main scoring method
        Run diarization to segment audio into individual speaker clusters, then if mode is
        SID or SPEAKER_DETECTION, for each cluster we compared to enrolled speaker models
        and replace the speaker cluster label with known identities if detected.
        """
        # Deal with region opts
        if opts is not None and 'region' in opts:
            audio.trim_samples(opts['region'])        

        domain = self.get_domains()[domain_id]
        audio.make_mono()

        # Update options if hey are passed in
        config = self.update_opts(opts, domain)
        self.update_speaker_models(domain)

        # Check that speakers are enrolled
        available_classes = np.array(self.list_classes(domain_id))
        if classes is None:
            classes = available_classes
        if len(classes)==0:
            raise Exception("No speaker models enrolled for plugin")

        # Overwrite embed system SAD config with the plugin specs
        speech, duration = self.run_sad(audio, config)
        spk_clst_alignment = np.zeros(speech.shape,dtype=int)
        spk_clst_alignment[speech] = 1

        # Add the output to an alignment object
        finalseg = idento3.Alignment()
        for i in [x for x in np.unique(spk_clst_alignment) if x>0]:
            finalseg.add_from_indices('%s-spk%d' % (audio.id,i), spk_clst_alignment==i, 'speech')

        #Process each cluster, and take the highest score as the label IFF it's above the threshold
        sid_scores = None # For the collection over clusters, prior to locating max segment score per model
        for spkid in list(finalseg.keys()):
            # For each cluster, trim to the speech of the speaker as given by diarization
            waveobject = idt.Wave(audio.data.copy(), audio.sample_rate, spkid)
            waveobject.trim_samples(finalseg.get_start_end(spkid, unit="seconds"))
            waveobject = Wave(waveobject.data, waveobject.sample_rate)
            #waveobject.filename = audio.filename # required for get_audio_vector iv storing vectors

            # Get the embedding
            speech, duration = self.run_sad(audio, config)
            iv = self.get_audio_vector(domain, waveobject, speech, config)

            # Score the cluster against the enrolled speaker models
            enroll_ivs = []
            test_ivs = idento3.ivector_postprocessing(iv, domain.lda, domain.mean)

            Tseg2mod = np.arange(len(classes))
            Tseg2mod = np.c_[Tseg2mod, Tseg2mod].astype(int)
            tseg2mod = np.zeros([1,2], dtype=int) # Just the test file alone
            if domain.is_jplda:
                raise Exception("Untested")
                score_mat = domain.eng.plda.score(enroll_ivs, test_ivs, Tseg2mod, tseg2mod, Pss=0.5, Psd=0.5, Tchannels=None, tchannels=None)
            else:
#                score_mat = idt.plda_verifier(enroll_ivs, test_ivs, domain.plda, Tseg2model=Tseg2mod, tseg2model=tseg2mod)
                score_mat = idt.plda_verifier(domain.speaker_data, test_ivs, domain.plda, Tseg2model=domain.enroll_seg2model, tseg2model=None)
            scores = idento3.Scores(domain.classes, [spkid], score_mat)

            # create the alignment key, ready for calibration
            # the complexity with np.in1d allows the key train_ids ordering to be the same as domain.train_ids and speed up self.get_cal_similarities later
            select_classes = domain.classes[np.in1d(domain.classes, classes)]
            align_key = idento3.Key([clsid for clsid in select_classes],
                                [aid for aid in [spkid]],
                                np.ones((len(select_classes), 1), dtype=np.int8))
            #align_key = idento3.Key([str(clsid).encode('ascii','ignore') for clsid in select_classes],
            #                    [aid.encode('ascii','ignore') for aid in [spkid]],
            #                    np.ones((len(select_classes), 1), dtype=np.int8))

            ##############################
            # GLOBAL duration-aware calibration for each duration pair automatically processed using side information
            ##############################
            # Duration dependent calibration
            score_map = {'system0': scores}

            sid_scores = scores.score_mat
            side_info = None
            if len(domain.calibration_model['dsi_map']) > 1:
                cont_si_dict = {k: domain.speaker_meta[k]/10.0 for k in select_classes}
                cont_si_dict[spkid] =  duration/10.0
                disc_si_dict, val_map = idt.discretize_dict_values(cont_si_dict, [50.0,150.0,350.0,750.0])
                #[2.0,4.5,102.0,104.5,202.0,204.5,502.0,504.5,1002.0,1004.5]
                side_info, dsi_map = idt.fusion.get_disc_sideinfo_matrix_for_key(disc_si_dict, align_key, len(val_map))
            sid_scores = idt.fusion.apply_calibration_and_fusion(score_map, align_key, domain.calibration_model, dsi_data=side_info)

            if sid_scores is None:
                # No clusters had sufficient speech for testing
                raise Exception("Insufficient speech for testing of file [%s]" % audio.filename)

        # For each enrolled speaker, report a single result as the highest scoring speaker cluster from diarization
        result = []
        result = {speaker: sid_scores.score_mat[idx][0] for (idx, speaker) in enumerate(sid_scores.train_ids)}

        return result


    ### ClassEnroller/ClassModifier ###
    def add_class_audio_vector(self, domain_id, audio_vector_path, class_id, enrollspace, params):
        """
        Takes an audio vector path and adds it to the enrollments for a domain
        """
        audio_dir = os.path.join(enrollspace, "staging", params['audio_id'])
        utils.mkdirs(audio_dir)
        for file in glob.glob(os.path.join(audio_vector_path, "*")):
            dest = os.path.join(audio_dir, os.path.basename(file))
            if not os.path.exists(dest): # Assumes the audio.id was used as the last folder
                shutil.copy(file, dest)


    def add_class_audio(self, domain_id, audio, class_id, enrollspace, opts=None):

        domain = self.get_domains()[domain_id]
        config = self.update_opts(opts, domain)

        if opts is not None and 'region' in opts:
            audio.trim_samples(opts['region'])
           
#        speech, duration = self.run_sad(audio, config)
#        iv = self.get_audio_vector(domain, audio, speech, config)

#        audio_dir = os.path.join(enrollspace, "staging", audio.id)
#        utils.mkdirs(audio_dir)

        # Save the enrollment model/data
#        if not os.path.exists(os.path.join(audio_dir,audio.id)+'.iv'):
#            idt.save_vectors_in_dir([{'iv':iv, 'meta':duration, 'md5sum':audio.id}], [audio.id], audio_dir, ext=".iv", with_meta=True)

        # Save if stroage exists

        audio_dir = os.path.join(enrollspace, "staging", audio.id)
        utils.mkdirs(audio_dir)

        enrollment_dir = self.get_enrollment_storage(domain_id, class_id)
        embed_filename = os.path.join(enrollment_dir, class_id, audio.id, audio.id + '.vecs.h5')
        out_embed_filename = os.path.join(audio_dir, audio.id + '.vecs.h5')
        if not os.path.exists(embed_filename):
            # Generate vecs for 10, 20, 30, 60, 120, 240 durations AND the duration of each in embed_data
            speech, duration = self.run_sad(audio, config)
            embeds = self.get_audio_vector(domain, audio, speech, config)

            # Save vecs and meta
            embed_dict = dict([(audio.id + '_' + str(i), v) for i, v in enumerate(embeds)])
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
        
        for file in glob.glob(os.path.join(enrollspace, "staging", "*")):
            dest = os.path.join(final_enrollment_dir, os.path.basename(file))
            if os.path.exists(dest):
                shutil.rmtree(dest)
            shutil.move(file, dest)

    def remove_class(self, domain_id, class_id, workspace):

        class_dir = self.get_enrollment_storage(domain_id, class_id)
        shutil.rmtree(class_dir)

    def export_enrollment(self, domain_id, class_id):

        if self.is_enrolled(domain_id, class_id):
            return self.get_enrollment_storage(domain_id, class_id), {}

        # Not a valid enrollment
        self.escape_with_error("Export failed, no enrollment found for this class id: '{}'".format(class_id))

    def import_enrollment(self, domain_id, class_id, enrollment_path, enrollment_metadata):

        final_enrollment_dir = self.get_enrollment_storage(domain_id, class_id)

        # Clean out speaker (we don't attempt merging)
        if os.path.isdir(final_enrollment_dir):
            shutil.rmtree(final_enrollment_dir)
        shutil.move(enrollment_path, final_enrollment_dir)

    ##############################################################################################################
    ############################################# TRAINING FUNCTIONS #############################################
    ##############################################################################################################
    def get_learning_timeout_weight(self):
        return 5.0

    def get_global_scoring_opts(self):
        """
        These options are used in the OLIVE GUI and may be configured on the commandline by passing a file to --options
        """
        global_scoring_trait_options = [
            TraitOption('sad_threshold', "Threshold to determine speech frames", "Higher value results in less speech fro processing", TraitType.CHOICE_TRAIT, "", self.config.sad_threshold),
        ]
        return global_scoring_trait_options


# This line is very important! Every plugin should have one
plugin = CustomPlugin()
