import glob, os, numpy as np, json, importlib, shutil
import site
from olive.plugins import Plugin, RegionScorer, ClassModifier, logger, utils, TraitOption, TraitType
import olive.core.config as olive_config

import torch
import idento3 as idt
import idento3.engines as idte
from scipy.linalg import LinAlgError

# dynapy
import dynaPy.DynaSpeakScenicContainer as ds



##################################################
# CONFIG - DO NOT CHANGE THESE UNLESS TOLD TO DO SO BY A DEVELOPER
default_config = idt.Config(dict(
# Configurable as [region scoring] parameters
# DETECTION OPTIONS
threshold           = 0.0,
window              = 60.0,
min_speech          = 10.0,
window_overlap      = 20.0,  # Overwritten in init to be 1/3 of window
# SAD
sad_threshold       = -1.0,
sad_filter          = 11,
sad_interpolate     = 4,
sad_padding         = 0.3,

max_lda_dim         = 200,
region_merge        = 4.0,
min_region          = 4.0,
))
MAX_SEQ_LENGTH      = 500
##################################################


class CustomPlugin(Plugin, RegionScorer, ClassModifier):

    def __init__(self):
        self.create_date = "2025-03-20"
        self.revision_date = "2025-03-20"
        self.group = "Content"
        self.task            = "TPD"
        self.label = "TPD DynaPy (Commercial)"
        self.description = "Run ASR using DynaPy for TPD of XMLRoBERTa features"
        self.minimum_runtime_version = "5.7.0"
        self.minimum_olive_version = "5.7.0"
        self.version = "1.0.0"
        self.vendor = "SRI International"
        self.loaded_domains  = []
        self.loaded_base     = False
        self.config          = default_config
        loader               = importlib.machinery.SourceFileLoader('plugin_config', os.path.join(os.path.dirname(os.path.realpath(__file__)), 'plugin_config.py'))
        spec                 = importlib.util.spec_from_loader(loader.name, loader)
        mod                  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.config.update(mod.user_config)
        self.config.window_overlap = self.config.window / 3.0
        self.VALID_PARAMS    = ['region'] + list(self.config.keys()) # For checking user inputs and flagging unknown paramters. Region is passed with 5-column enrollment
        site.addsitedir(os.path.dirname(os.path.realpath(__file__)))


    ### Method implementations required of a Plugin ###
    def list_classes(self, domain_id, include_base=True):
        try:
            enrollments_dir = self.get_enrollment_storage(domain_id)
            user_topics = [x for x in os.listdir(enrollments_dir) if x[0] != '.']
            if include_base:
                domain = self.get_domains()[domain_id]
                base_topics  = np.loadtxt(domain.get_artifact("model.classes"), dtype=str).tolist()
            else:
                base_topics = []
            return np.unique(user_topics + base_topics).tolist()
        except AttributeError:
            return []

    def load(self, domain_id, device=None): 
        # Load the base components (domain-independemt)
        if not self.loaded_base:

            # SAD
            sadextractor_config = dict(
                nnet_mvn    = self.get_artifact("sad_mvn.npz"),
                nnet        = self.get_artifact("sad_nn.npz"),
                linearout   = False,
                layer       = -1,
                dnn_format  = 'theano',
                dnn_input_config  = self.get_artifact("sad.config"),
            )
            sadconfig = idt.Config()
            sadconfig.update(sadextractor_config)
            self.sad_engine = idt.Frontend(idt.TFDnnFeatures)(sadconfig)

            self.loaded_base = True

        # Domain dependent components
        if domain_id not in self.loaded_domains: #not self.base_loaded:
            import bert_extractor_v5
            domain = self.get_domains()[domain_id]
            domain.device = device

            # Models
            domain.topic_embeds = {}
            domain.topic_meta   = {}
            model_data          = domain.get_artifact("model_data.h5")
                
            # The calibration model
            domain.base_data    = idt.read_data_in_hdf5(model_data)
            domain.base_topics  = np.loadtxt(domain.get_artifact("model.classes"), dtype=str).tolist()
            domain.cal_offset = float(open(domain.get_artifact("cal.offset")).readlines()[0].strip())
            plda_params        = idt.read_data_in_hdf5(domain.get_artifact("lda_plda.h5"))
            domain.lda  = plda_params['IvecTransform']['LDA']
            domain.mean = plda_params['IvecTransform']['Mu']
            domain.plda = idt.SPLDA.load_from_dict(plda_params['PLDA'])

            # Check on models in memory
            self.update_topic_models(domain)

            # make scratch space for log
            domain.tempdir = os.path.join(Plugin.scratch_space, 'dynaspeak_logs', domain_id)
            if os.path.exists(domain.tempdir) == False:
                os.makedirs(domain.tempdir)

            model_dir = domain.get_artifact('roberta')
            domain.bert_extractor = bert_extractor_v5.XLMRobertaExtractor(model_dir, MAX_SEQ_LENGTH)

            self.initDynaPy(domain)
            domain.dtype = [('kwd', 'S200'), ('seg', 'S200'), ('start', np.float32), ('end', np.float32), ('scr', float)]

            self.loaded_domains.append(domain_id)

            logger.info("Loading of plugin '%s' domain '%s' complete." % (self.label, domain_id))


    def get_cuda_device(self, domain_id):
        domain = self.get_domains()[domain_id]
        device_conf = domain.config.get('domain', 'device') if domain.device is None else domain.device
        cuda_device = "-1"
        if not ('gpu' == device_conf[:3] or 'cpu' == device_conf[:3]):
            self.escape_with_error("'device' parameter in meta.conf of domain [{}] should be 'cpu' or 'gpuN' where N is the index of the GPU to use. Instead, it is set to '{}'".format(domain_id, device_conf))
        if 'gpu' == device_conf[:3]:
            try:
                # Mkae sure gpu index can be extracted as int
                gpu_index = int(device_conf[3:])
            except ValueError:
                self.escape_with_error("'device' parameter in meta.conf of domain [{}] should be 'cpu' or 'gpuN' where N is the index of the GPU to use. Instead, it is set to '{}'".format(domain_id, device_conf))
            # Check for CVD
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                if os.environ['CUDA_VISIBLE_DEVICES'] == "":
                    self.escape_with_error("Requested gpu use in meta.conf of domain, but environment variable 'CUDA_VISIBLE_DEVICES' is empty. Either unset this variable or set it apprioriately to GPUs to be used")
                else:
                    cvd = np.array(os.environ['CUDA_VISIBLE_DEVICES'].split(','), dtype=int)
                    cvd_map = dict(zip(cvd, np.arange(len(cvd)).astype(int)))
                    if gpu_index not in cvd_map:
                        self.escape_with_error("Requested gpu {} in meta.conf of domain {} but this GPU was not listed in environment variable CUDA_VISIBLE_DEVICES.".format(gpu_index, os.environ['CUDA_VISIBLE_DEVICES']))
                    else:
                        gpu_index = cvd_map[gpu_index]
            cuda_device = "{}".format(gpu_index)
            logger.info("Allocated GPU {} to plugin/domain {}/{}".format(cuda_device, self.label, domain_id))
        return cuda_device


    def update_classes(self, domain_id):
        # Function that is called by the server after finalizing all enrollments
        domain = self.get_domains()[domain_id]
        self.update_topic_models(domain)


    def update_topic_models(self, domain, do_prep=True):
        # Called on load and finalize/remove class
        enrollment_dir = self.get_enrollment_storage(domain.get_id())
        topic_ids = self.list_classes(domain.get_id(), include_base=False)

        # Reset variables before load
        domain.topic_embeds = {}

        # Now go ahead and update the require topics
        topic_map    = {}
        topic_embeds = {}
        for topic_id in topic_ids:
            topic_dir = os.path.join(enrollment_dir, topic_id)
            for session_id in os.listdir(topic_dir):
                if session_id[0] == '.':
                    continue
                iv_path = os.path.join(enrollment_dir,topic_id,session_id,session_id +'.h5')
                try:
                    data, ids = idt.read_data_in_hdf5(iv_path, nodes=['/data', '/ids'])
                    for i,d in enumerate(ids):
                        topic_embeds[d] = data[i,:][np.newaxis,:]
                    if topic_id not in list(topic_map.keys()):
                        topic_map[topic_id] = []
                    topic_map[topic_id] += ids.tolist()

                except:
                    raise self.escape_with_error("Corrupt topic enrollment file path [%s]. Please remove and try again" % os.path.dirname(iv_path))

        # Enroll all the topic models and gather calibration meta
        if True: #len(list(topic_map.keys()))>0:
            for topic_id,session_ids in topic_map.items():
                domain.topic_embeds[topic_id] = [topic_embeds[x] for x in session_ids]

            # Update LDA/Mu
            classes     = domain.base_data['ids'].tolist()
            if len(list(topic_map.keys()))==0:
                enr_classes = []
            else:
                enr_classes = np.hstack([len(domain.topic_embeds[t])*[t] for t in list(domain.topic_embeds.keys())]).ravel().tolist()
            classes    += enr_classes
            classes, class_indices = np.unique(classes, return_inverse=True)
            if len(list(topic_map.keys()))==0:
                all_data = domain.base_data['data']
            else:
                all_data    = np.vstack([domain.base_data['data'], np.vstack([np.vstack(domain.topic_embeds[t]) for t in list(domain.topic_embeds.keys())])])
            domain.mu   = np.mean(all_data, axis=0).dot(domain.lda)

            data        = idt.ivector_postprocessing(all_data, domain.lda, domain.mu)

            # Get the data for calibration
            domain.enroll_ids  = list(domain.topic_embeds.keys())
            complementary = np.array(domain.base_topics)[~np.in1d(np.array(domain.base_topics), domain.enroll_ids)].tolist()
            if len(complementary) == 0:
                classes   = enr_classes
                data      = np.vstack([np.vstack(domain.topic_embeds[t]) for t in list(domain.topic_embeds.keys())])
            else:
                # Remove base data if it is the same class as the user-enrolled data
                base_index = [i for i,x in enumerate(domain.base_data['ids'].tolist()) if x in complementary]
                classes    = domain.base_data['ids'][base_index].tolist()
                classes   += enr_classes
                if len(list(topic_map.keys()))==0:
                    data = domain.base_data['data'][base_index]
                else:
                    data       = np.vstack([domain.base_data['data'][base_index], np.vstack([np.vstack(domain.topic_embeds[t]) for t in list(domain.topic_embeds.keys())])])
            classes, class_indices = np.unique(classes, return_inverse=True)
            data     = idt.ivector_postprocessing(data, domain.lda, domain.mu)

            # Update Calibration
            test_logids = ["%03d" % x for x in np.arange(data.shape[0])]
            try:
                test_ids = np.arange(class_indices.shape[0])
                test_seg2model = np.vstack([test_ids, test_ids]).T
                domain.enroll_seg2model = np.vstack([test_ids, class_indices]).T
                cal_scores = idt.plda_verifier(data, data, domain.plda, Tseg2model=domain.enroll_seg2model, tseg2model=test_seg2model)
                cal_scores = idt.Scores(classes, test_logids, cal_scores)
                domain.enroll_data = data
                domain.enroll_classes = classes
            except LinAlgError:
                self.escape_with_error("Insufficient data in at least one enrolled class. Please add more to the class with the fewest examples [%s has only %d example(s)] and retry, or remove the class." % (classes[np.argmin(np.bincount(class_indices))], np.min(np.bincount(class_indices))))
            cal_key          = idt.LidKey.fromiter(list(zip(test_logids, classes[class_indices])))
            domain.cal_model = idt.fusion.train_calibration_and_fusion({'system0':cal_scores}, cal_key, priors=None, sid=False)


    def update_opts(self, opts, domain):

        # Copy values
        config = idt.Config(dict(self.config))

        if opts is not None:
            # Check that all passed options are valid for this plugin
            param_check = np.in1d(list(opts.keys()), self.VALID_PARAMS)
            if np.any(~param_check):
                raise self.escape_with_error("Unknown parameter(s) passed [%s]. Please remove from the optional parameter list." % ','.join(np.array(list(opts.keys()))[param_check==False].tolist()))

            config.update(opts)

            # File-passed options are in in text format, so we need to convert these as necessary
            config.threshold       = float(config.threshold)
            config.window          = float(config.window)
            config.window_overlap  = float(config.window_overlap)
            config.min_speech      = float(config.min_speech)
            if config.window_overlap >= config.window:
                raise self.escape_with_error("Window overlap [%d] must be less than the window size [%d]!" %(config.window_overlap, config.window))

            config.sad_threshold   = float(config.sad_threshold)
            config.sad_filter      = int(config.sad_filter)
            config.sad_interpolate = int(config.sad_interpolate)
            config.sad_padding     = float(config.sad_padding)

            config.max_lda_dim     = int(config.max_lda_dim)
            config.region_merge    = float(config.region_merge)
            config.min_region      = float(config.min_region)

            logger.debug("Using user-defined parameter options, new config is: %s" % config)

        return config

    def initDynaPy(self, domain):
        """
        Initialize DynaPy from domain and read vocabulary.
        """

        # Create and initialize DS object
        domain.DS = ds.DynaSpeakScenicContainery()
        try:
            # Try to call setEnableLogging(), older dynapy may not have this
            # method.  This turns the DynaSpeak log file on if the plugin was
            # run with --debug, otherwise it turns the DynaSpeak log file off.
            #
            # If this method does not exist, then DynaSpeak log file creation is
            # controlled by the value of the "EnableLogging" key/value pair in
            # the domain's dynaspeak.config file.
            logger.info("initDynaPy: DynaPy setEnableLogging({})!".format(olive_config.debug))
            domain.DS.setEnableLogging(olive_config.debug)
        except AttributeError:
            pass
        domain.DS.load(domain.get_path(), domain.tempdir)

        # Read vocabulary into python (unordered) set:
        domain.vocab = None
        words_filename = domain.get_artifact(os.path.join('DNN_Models','words.txt'))
        if os.path.isfile(words_filename):
            with open(words_filename, 'r') as f:
                domain.vocab = set([line.split()[0] for line in f.readlines()])
                logger.debug("initDynaPy: Read {} vocabulary words!".format(len(domain.vocab)))
        else:
            logger.debug("initDynaPy: words.txt file '{}' doesn't exist!".format(words_filename))

        # Get recognizer sample rate
        domain.dynaspeak_sample_rate = int(domain.DS.get_parameter("audio.SamplingRate"))

    def JSON_to_KWS_output(self, json_string, segname, dtype):
        """
        Process recognizer output in JSON format.  Add segment start time to all keyword
        start and end times to get correct times.

        Inputs:
        - json_string: JSON recognizer output
        - segname: segment information in format ['id' 'segN' start end]
        - dtype: desired KWS datatypes

        Outputs:
        - hyp: Recognizer output hypothesis plus segment_id, "id.segN result_string"
        - kws: KWS results with corrected times, as numpy list ['words', id, adj_start, adj_end, score]
        - this_json: Original JSON plus segment_id, "id.segN JSON_string"
        """
        ## JSON dynaspeak KWS format is
        #{u'channel': 1,
        # u'endFrame': 296,
        # u'endMillis': 2960,
        # u'key': u'KW1',
        # u'pass': 1,
        # u'prob': 0.998049,
        # u'startFrame': 180,
        # u'startMillivs': 1800,
        # u'utteranceId': u'HTK-LATTICE#HTK-LATTICE',
        # u'words': u'some responsibility'}

        # convert JSON string to dictionary
        json_input = json.loads(json_string)

        # return value, format is "id.segN JSON_string"
        this_json = segname[0] + '.' + segname[1] + ' ' + json_string

        # return value, format is "id.segN result_string"
        hyp = segname[0] + '.' + segname[1] + ' ' + self.tryutf8encode(json_input['result'])

        # shift is how much we need to add to result times
        shift = float(segname[2])

        if 'parseTreeElements' in json_input:
            kws = np.array([(self.tryutf8encode(f['value']), segname[0], shift + float(f['startMillis'])/1000.0, shift + float(f['endMillis'])/1000.0, float(f['confidence']))
                    for f in json_input['parseTreeElements']], dtype = dtype)
            kws = np.unique(kws)
        else:
            kws = np.array([])
        return hyp, kws, this_json


    def run_sad(self, audio, config, suppress_error=False):

        # Detect speech frames
        waveobject = idt.Wave(audio.data.copy(), audio.sample_rate, audio.id)

        self.vad_opts = {'filter_length': config.sad_filter, 'interpolate':config.sad_interpolate, 'threshold':config.sad_threshold}
        speech = idte.vad.dnn_vad(self.sad_engine.feat_fc(waveobject), self.sad_engine.embedextractor, return_llrs=False, **self.vad_opts)

        duration = speech.sum()
        if duration < int(config.min_speech*100):
            if suppress_error:
                return []
            else:
                self.escape_with_error("Insufficient speech found for [%s]. Found %.3f which is less than the required amount of %.3f seconds." % (audio.filename, duration/100., config.min_speech))

        if config.sad_padding > 0.0:
            speech = idt.pad(speech[np.newaxis,:], config.sad_padding).ravel()

        # Check that there is enough speech in at least one extended region of min_speech
        # The goal here is to find min_speech of a continuous segment whereby gaps of less than region_merge are considered to join speech segments.
        temp = idt.Alignment()
        temp.add_from_indices('temp', speech, 'speech')
        passes, mm, cre, maxspan = False, 0.0, 0, 0.0
        for i, sten in enumerate(temp.get_start_end('temp')):

            if sten[0] - cre < int(config.region_merge*100):
                cre = sten[1]
                maxspan += cre - sten[0]
            else:
                cre = sten[1]
                maxspan = cre - sten[0]

            mm = max(mm, maxspan)
            if maxspan >= int(config.min_speech*100):
                passes = True
                break

        if not passes and not suppress_error:
            raise self.escape_with_error("Insufficient continuous speech found for [%s]. Found %.3f of speech, but no continuous speech of %.3f seconds. Maximum continuous segment was %.3f seconds." % (audio.filename, duration/100., config.min_speech, maxspan/100.))

        return speech


    def get_chunked_bert_vectors(self, domain_id, audio, config, suppress_error=False):
        
        domain = self.get_domains()[domain_id]
        speech   = self.run_sad(audio, config, suppress_error=suppress_error)

        # Account for the case of low speech and suppress_error
        if len(speech)==0:
            return [], []

        speech_inds = np.where(speech)[0]

        embeds_aligns = []
        embeds_chunked = []
        hyps_aligns = []
        hyps_chunked = []
        domain.DS.resetState()
        align = idt.Alignment()
        align.add_from_indices('temp', speech, 'speech')
        # Chunk the speech segments from the original Wave, if a speech segment is long, chunk that up.
        # The aim here is to end up with chunks of ASR text in hyps_chunked and the corresponding timestamps in hyps_aligns
        for st, en in align.get_start_end('temp', unit='frames'):
            if en-st > (config.window_overlap*100):
                nchunks = np.ceil((en-st)/(config.window_overlap*100)).astype(int)
                chunk_size = np.ceil((en-st) // nchunks).astype(int) + 1
                current = st
                chunks = []
                while current < en:
                    real_en = min(en, current + chunk_size)
                    chunks.append((current, real_en))
                    current += chunk_size
            else:
                chunks = [(st,en)]

            for chunk in chunks:
                st, en = chunk
                wav = idt.Wave(audio.data.copy(), audio.sample_rate)
                wav.trim_samples([(st/100.0,en/100.0)])
                wav.unnormalize()
                
                # RUN ASR using Dynapy on the chunk of audio
                decoded = domain.DS.recognize(domain.get_path(), None, wav.data, len(wav), domain.tempdir)

                # Get recognition results
                resultStr = ""
                if decoded:
                    resultStr = domain.DS.getResult()

                if len(resultStr) > 0:
                    logger.debug("Recognized             = %s" % resultStr)

                    hyps_chunked.append('temp '+resultStr)
                    hyps_aligns.append([st, en])

        if len(hyps_chunked) == 0:
            self.escape_with_error("Found no continuous speech segments with a minimum of %.3f seconds of speech." % config.min_speech)

        # Now window the text for embeddings
        current = []
        starts = []
        ends = []

        # Now get the BERT embeddings for each chunk to end with embeds_chunked, and embeds_aligns
        for hypi, sten in enumerate(hyps_aligns):
            st, en = sten
            ends.append(en)
            starts.append(st)
            current.append(hyps_chunked[hypi].split(' ', 1)[1])

            # Shift the window start as needed
            to_remove = []
            for endi, ending in enumerate(ends):
                if en - ending > config.window*100:
                    to_remove.append(endi)

            current = [x for i,x in enumerate(current) if i not in to_remove]
            starts = [x for i,x in enumerate(starts) if i not in to_remove]
            ends = [x for i,x in enumerate(ends) if i not in to_remove]
            instr = 'templogid ' + ' '.join(current)

            # Extract the BERT and store info for timing
            logid, embed = domain.bert_extractor.extract([instr])

            embeds_chunked.append(embed)
            embeds_aligns.append(speech_inds[np.min(starts):en])

        # Finish the final parts of the sliding window
        while len(current) > 1:
            current, starts, ends = current[1:], starts[1:], ends[1:]
            instr = 'templogid ' + ' '.join(current)

            logid, embed = domain.bert_extractor.extract([instr])
 
            embeds_chunked.append(embed)
            embeds_aligns.append(speech_inds[np.min(starts):en])

        if len(embeds_chunked) == 0:
            self.escape_with_error("Unable to decode a transcription for audio file.")

        embeds = np.vstack(embeds_chunked)

        return embeds, embeds_aligns


    ### RegionScorer ###
    def run_region_scoring(self, domain_id, audio, workspace, classes=None, opts=None):
        """
        Main scoring method
        """
        domain = self.get_domains()[domain_id]
        audio.make_mono()

        # Device to run on
        if not hasattr(domain, 'cuda_device'):
            domain.cuda_device = self.get_cuda_device(domain_id)
        if domain.cuda_device != "-1":
            device = torch.device('cuda:{}'.format(domain.cuda_device))
        else:
            device = torch.device('cpu')
        domain.bert_extractor.model = domain.bert_extractor.model.to(device)

        # Update options if they are passed in
        config = self.update_opts(opts, domain)

        # Check that topics are enrolled
        if classes is None:
            classes = self.list_classes(domain_id)
        else:
            non_existant = np.array(classes)[~np.in1d(classes, domain.enroll_ids)]
            if len(non_existant) > 0:
                self.escape_with_error("%d topic(s) requested that are not enrolled: [%s]" % (len(non_existant), ','.join(non_existant.tolist())))

#        if len(classes)==0:
#            self.escape_with_error("No topic models enrolled for plugin")

        result = []
        audios, offsets = [], []
        timing_map = False
        # Trim to the region of interest if it was passed in
        if opts is not None and 'region' in opts:
            timing_map = True
            for reg in opts['region']:
                if (reg[1] - reg[0]) < config.min_speech:
                    logger.warn("Region [{}, {}] is too short to contain sufficient speech (requires {} seconds per region); moving to next region.".format(reg[0], reg[1], config.min_speech))
                    continue
                audiox = idt.Wave(audio.data.copy(), audio.sample_rate, audio.id)
                audiox.trim_samples([reg])
                audios.append(audiox)
                offsets.append(int(reg[0]*1000))
        else:
            audios.append(audio)
            offsets.append(0)

        speech_found = False
        for ii, audio in enumerate(audios):
            regoffset = offsets[ii]

            # Get the embeddings
            embeds, aligns  = self.get_chunked_bert_vectors(domain_id, audio, config, suppress_error=timing_map)

            if len(embeds) == 0 and timing_map:
                logger.warn("Region found insufficient speech (requires {} seconds per region); moving to next region.".format(config.min_speech))
                continue
            else:
                speech_found = True

            # Score the cluster against the enrolled topic models
            test_embeds = idt.ivector_postprocessing(embeds, domain.lda, domain.mu)
            test_ids    = np.arange(test_embeds.shape[0], dtype=int)
            test_seg2model = np.vstack([test_ids, test_ids]).T
            score_mat = idt.plda_verifier(domain.enroll_data, test_embeds, domain.plda, Tseg2model=domain.enroll_seg2model, tseg2model=test_seg2model)
            scores = idt.Scores(domain.enroll_classes, test_ids.tolist(), score_mat)

            # Calibration
            score_map = {'system0': scores}
            align_key = idt.scoring.Key(scores.train_ids, scores.test_ids, scores.mask)
            align_key = align_key.filter(domain.cal_model['classes'], scores.test_ids)
            topic_scores = idt.fusion.apply_calibration_and_fusion(score_map,align_key,domain.cal_model,sid=False)
            topic_scores.score_mat = idt.loglh2detection_llr(topic_scores.score_mat.T).T
            
            # create the alignment key, ready for calibration
            # the complexity with np.in1d allows the key train_ids ordering to be the same as domain.gb['classes'] and speed up self.get_cal_similarities later
            select_classes = domain.enroll_classes[np.in1d(domain.enroll_classes, classes)]
            topic_scores = topic_scores.filter(select_classes, topic_scores.test_ids)
            topic_scores.score_mat = topic_scores.score_mat - domain.cal_offset # Offset to 0.0 threshold
            
            # Output detections as all scores above 0.0
            regpad = config.region_merge/2.0
            score_array = np.ones(int(audio.duration*100+1), dtype='f')*-1
            for i,topic_id in enumerate(topic_scores.train_ids):
                detections = np.where(topic_scores.score_mat[i,:]>=config.threshold)[0]
                score_array[:] = config.threshold - 1 # Holds the score per frame dominated by largest score for the topic

                # Process smallest to largest so that largest score is dominant in alignment
                for idet in detections[np.argsort(topic_scores.score_mat[i,detections])]:
                    score_array[aligns[idet]] = topic_scores.score_mat[i,idet]
                if np.any(score_array>config.threshold):
                    align = idt.Alignment()
                    align.add_from_indices(audio.id,score_array>config.threshold,topic_id)
                    # Merge segments of the topic that are really close.
                    # Pad for region merge
                    rst = -1 # Tracking of tue first start
                    for st, en, topic_id in align.get_start_end_label(audio.id):
                        if rst == -1:
                            rst = st
                        align.add(audio.id, 1, st/100. - regpad, en/100. + regpad, topic_id)

                    # Unpad for region merge
                    first = True
                    for st, en, topic_id in align.get_start_end_label(audio.id):
                        if first:
                            # Account for actual first instance
                            if st/100. > regpad:
                                rst = st/100. + regpad
                            else:
                                rst = rst/100.
                            align.add(audio.id + 'mod', 1, rst, en/100. - regpad, topic_id)
                            first = False 
                        else:
                            align.add(audio.id + 'mod', 1, st/100. + regpad, en/100. - regpad, topic_id)

                    for st, en, topic_id in align.get_start_end_label(audio.id + 'mod'):
                        if en - st >= config.min_region*100: # Apply minimum constraints
                            result += [(regoffset + int(st*10), regoffset + int(en*10), topic_id, score_array[st:en].max())]

        if not speech_found:
            aid = audio.id if audio.filename is None else audio.filename
            self.escape_with_error("Insufficient speech found for [%s]. Requires at least one region with %.3f seconds of speech." % (aid, config.min_speech))

        if len(result)>0:
            result = sorted(result, key = lambda x: x[0])

        # Convert timestamps from ms to s
        result = [(np.float32(res[0] / 1000.0), np.float32(res[1] / 1000.0), res[2], res[3]) for res in result]


        return {'topics':result}


    ### ClassEnroller/ClassModifier ###
    def add_class_audio_vector(self, domain_id, audio_vector_path, class_id, enrollspace, params):
        """
        Takes an audio vector path and adds it to the enrollments for a domain
        """
        if len(class_id) == 0:
            self.escape_with_error("Please pass a valid class_id for enrollment (can not be empty string '')")
        audio_dir = os.path.join(enrollspace, "staging", params['audio_id'])
        utils.mkdirs(audio_dir)
        for file in glob.glob(os.path.join(audio_vector_path, "*")):
            dest = os.path.join(audio_dir, os.path.basename(file))
            if not os.path.exists(dest): # Assumes the audio.id was used as the last folder
                shutil.copy(file, dest)


    def add_class_audio(self, domain_id, audio, class_id, enrollspace, opts=None):

        domain = self.get_domains()[domain_id]
        config = self.update_opts(opts, domain)
        if len(class_id) == 0:
            self.escape_with_error("Please pass a valid class_id for enrollment (can not be empty string '')")
        if opts is not None and 'region' in opts:
            if 'region' in opts:
                audio.trim_samples(opts['region'])
            
        duration = audio.duration
        audio_dir = os.path.join(enrollspace, "staging", audio.id)
        if not os.path.exists(os.path.join(audio_dir,audio.id)+'.h5'):
            embeds, _  = self.get_chunked_bert_vectors(domain_id, audio, config)

            utils.mkdirs(audio_dir)

            # Save the enrollment model/data
            ids = ["%s_%05d" % (audio.id, x) for x in np.arange(embeds.shape[0])]
            idt.save_consolidated_vectors(os.path.join(audio_dir,audio.id)+'.h5', ids, embeds)

        # Return location for the purpose of audiovectors (not needed for audio enrollment)
        return audio_dir, duration


    def remove_class_audio(self, domain_id, audio, class_id, enrollspace):

        if len(class_id) == 0:
            self.escape_with_error("Please pass a valid class_id for unenrollment (can not be empty string '')")

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

        if len(class_id) == 0:
            self.escape_with_error("Please pass a valid class_id for unenrollment (can not be empty string '')")
        if class_id in self.list_classes(domain_id, include_base=False):
            class_dir = self.get_enrollment_storage(domain_id, class_id)
            shutil.rmtree(class_dir)
        else:
            self.escape_with_error("Can not locate requested class for unenrollment '{}'.".format(class_id))


    def get_region_scoring_opts(self):
        """
        These options are used in the OLIVE GUI and may be configured on the commandline by passing a file to --options
        """
        region_scoring_trait_options = [
            TraitOption('threshold', "Detection threshold", "Higher value results in less detections being output (default %0.1f)" % self.config['threshold'], TraitType.CHOICE_TRAIT, "", ""),
            TraitOption('window', "Size of sliding window over speech", "Each window is assumed to contain a single topic. Longer is more robust (default %0.1f)" % self.config['window'], TraitType.CHOICE_TRAIT, "", ""),
            TraitOption('min_speech', "Minimum speech for detection", "Required amount of speech in file to enroll and perform topic detection (default %0.1f)" % self.config['min_speech'], TraitType.CHOICE_TRAIT, "", ""),
            TraitOption('sad_threshold', "SAD threshold", "Speech activity detection threshold. Lower results in more speech at the cost of more noise (default %0.1f)" % self.config['sad_threshold'], TraitType.CHOICE_TRAIT, "", ""),
            ]

        return region_scoring_trait_options


    def tryutf8encode(self, string):
        """
        From http://www.pitt.edu/~naraehan/python2/unicode.html:
        encode() and decode() are the pair of methods used to convert between
        the Unicode and the string types.
        encode() is used to turn a Unicode string into a regular string.
        decode() is used to turn a regular string into a Unicode string.

        Else returns a regular string.
        Modified by Julien van Hout Aug 29-2018
        """
        try:
            return string.encode('utf-8')
        except UnicodeError:
            return string

# This line is very important! Every plugin should have one
plugin = CustomPlugin()
