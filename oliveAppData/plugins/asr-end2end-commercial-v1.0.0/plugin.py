# Imports

# general
import os, time, json, numpy as np, re
import importlib, unicodedata
from itertools import groupby

if os.name != 'nt':
    import torch.multiprocessing as mp
else:
    import multiprocessing.dummy as mp

# olive
import olive.core.pem as pem
import olive.core.config as olive_config
from olive.plugins import *

# idento
import idento3
import idento3.engines as idte
import torch

# # HuggingFace
from transformers import AutoProcessor, AutoModelForCTC, Wav2Vec2ProcessorWithLM, modeling_outputs
from pyctcdecode import build_ctcdecoder
import ctranslate2
from transformers.utils import logging
logging.set_verbosity_error() 

# Static parameters
# (Should never change)

# When using SAD, it’s probably worth knowing how your system deals
# with something like one or two frames of speech.  If it’s an issue,
# you should have a min_speech parameter so that it can raise a nice
# error.
static_min_speech_secs = 0.31

loaded = False
domain = None

##################################################
# CONFIG - DO NOT CHANGE THESE UNLESS TOLD TO DO SO BY A DEVELOPER
#
# These are parameters that can be changed by the developer or via the API or options file.
#
# Parameters in plugin_config.py can be changed by the developer or user in the
# plugin_config.py or the API or options file (CLI mode).
#
# Load and init functions should not use any of these parameters.
#
default_config = idento3.Config(dict(
    # Configurable as [region scoring] parameters
    sad_threshold=0.0,
    sad_interpolate=1,
    sad_filter_frames=51,
    sad_speech_padding_secs=0.5,
    use_sad=True,
    max_segment_length_secs=8.0,
    overlap_segment_length_secs=1.0,
    reset_recognizer=True,
    unicode_output=True,
    unicode_normalization=None,
    LM_file_names=['language_model.bin','language_model.arpa'],
    LM_vocab_file='language_model.vocab',
    # LM decoding weights
    # alpha: Weight for language model during shallow fusion
    # beta: Weight for length score adjustment of during scoring
    decode_alpha=None,
    decode_beta=None,
    # unk_score_offset default is -10.0, lower reduces unk output but also transcribed words
    unk_score_offset=-10.0,
    remove_unk_char=True,
))

##################################################
# use_sad
#   Use SAD to compute segments for recognition.  If False, will just
#   segment into MAX_SEGMENT_LENGTH_SECS length segments with
#   OVERLAP_SEGMENT_LENGTH_SECS overlap.

# max_segment_length_secs
#   Maximum segment length to recognize.  If segments from SAD are
#   longer than this, then they are chopped with
#   OVERLAP_SEGMENT_LENGTH_SECS overlap.

# overlap_segment_length_secs
#   If segments from SAD are longer than MAX_SEGMENT_LENGTH_SECS, they
#   are chopped to maximum length MAX_SEGMENT_LENGTH_SECS seconds, with
#   overlap OVERLAP_SEGMENT_LENGTH_SECS seconds.  Note that overlaps
#   can cause plugin to output overlapping and possibly different
#   words during the overlap.

# reset_recognizer
#   Reset recognizer back to initial state before each new utterance
#   (not before each segment).  Set to True to ensure recognition output
#   doesn't depend on past history.  Set to false if performing "live"
#   recognition with a single recognizer, so that recognition output
#   DOES depend on past history.

# sad_threshold
#   Threshold to use for Speech/nonspeech decision

# sad_interpolate
#   SAD interpolation value.  Higher values use less computation.  Typical
#   value is 4.

# sad_filter_frames
#   Number of frames to smooth when computing SAD output.

# sad_speech_padding_secs
#   Pad detected speech regions with this many seconds.

# unicode_output
#   If False, and domain's intermediate output is in Buckwalter
#   format, then do not convert ASR output to Unicode.  Otherwise does
#   nothing.

# unicode_normalization
#   If not None, then apply unicode normalization to Arabic languages.
#   If value is None, do not normalize.
#   If set, must be None (no normalization), "NFC", "NFD", "NFKC", or "NFKD".

# *------------------------------------------------------------------------*
# Plugin
# *------------------------------------------------------------------------*

class CustomPlugin(Plugin, RegionScorer):

    # *------------------------------------------------------------------------*
    # Initialization and standard plugin routines:
    # *------------------------------------------------------------------------*
    #
    # Regarding self/domain storage, the idea is that if the thing to store is used domain
    # independently, it’s stored in self.  If it’s a domain-specific model/parameter/setting, it’s
    # stored in domain.  Anything stored in self MUST be done during init/load and then not touched
    # during running of the plugin - that just avoids bugs by following that protocol.  Things like
    # enrolled models, should be in the domain.* and updated with update_classes.
    #
    # *------------------------------------------------------------------------*

    def __init__(self):
        self.task  = "ASR"
        self.label = "ASR End-to-End (Commercial)"
        self.description = "Run ASR using a wav2vec2 model. Segment audio with SAD DNN V7 multi-v1."
        self.vendor = "SRI International"
        self.version = '1.0.0'
        self.minimum_runtime_version = '5.7.1'
        self.minimum_olive_version = '5.7.1'
        self.create_date = "2024-1-12"
        self.revision_date = "2024-10-28"
        self.group = "Content"

        self.loaded_domains  = []
        self.loaded_base     = False

        # The process is to define a self.config from the
        # default_config, update it with the plugin_config.py file and
        # then never touch self.config after that. For all
        # processing/scoring of audio, the opts parameter is used to
        # update a run-localized copy of self.config and hand that
        # around to functions that need it. Doing this prevents a bug
        # we found that overwrites parameters on an OLIVE worker and
        # they become sticky.

        # For checking user inputs and flagging unknown parameters.
        self.VALID_PARAMS    = ['region', 'speech_regions'] + list(default_config.keys()) # These are the valid keys to check that users don’t pass rubbish keys in opts.
        # Load the user_config in memory
        self.config   = default_config
        loader        = importlib.machinery.SourceFileLoader('plugin_config', os.path.join(os.path.dirname(os.path.realpath(__file__)), 'plugin_config.py'))
        spec          = importlib.util.spec_from_loader(loader.name, loader)
        mod           = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.config.update(mod.user_config)


    # ## Method implementations required of a Plugin ###

    def list_classes(self, domain_id):
        return []

    def submit_ctranslate2_work(self, domain_id, input_values):
        domain = self.get_domains()[domain_id]
        if domain.cuda_device != "-1":
            model_dir = domain.get_artifact('wav2vec2')
            device = torch.device('cuda:{}'.format(domain.cuda_device))
            if not hasattr(domain, "ct2_model"):
                domain.model = None
                if os.path.isfile(os.path.join(model_dir, "wav2vec2_partial", "pytorch_model.bin")):
                    model = AutoModelForCTC.from_pretrained(os.path.join(model_dir, "wav2vec2_partial"))
                    del model.wav2vec2.encoder.layers
                    del model.wav2vec2.encoder.layer_norm
                    domain.model = model.to(device)
                with open(os.path.join(model_dir, "model.bin"), "rb") as m:
                    model_bin = m.read()
                with open(os.path.join(model_dir, "vocabulary.json"), "rb") as v:
                    vocab_json = v.read()
                domain.ct2_model = ctranslate2.models.Wav2Vec2("model_bin", device=device.type, device_index=[int(domain.cuda_device)], compute_type='int8', intra_threads=4, inter_threads=4, files={"model.bin": model_bin, "vocabulary.json": vocab_json})
            return self.inference_process(input_values, domain.ct2_model, domain.model)
        else:
            output_queue = domain.manager.Queue()
            domain.queue.put((input_values, output_queue))
            return output_queue.get()
            
    def ctranslate2_work(self, cuda_device, queue, model_dir):
        if cuda_device != "-1":
            self.escape_with_error("ctranslate2_work() should not be called with a GPU configured")
        device = torch.device('cpu')
        model = None
        if os.path.isfile(os.path.join(model_dir, "wav2vec2_partial/pytorch_model.bin")):
            model = AutoModelForCTC.from_pretrained(os.path.join(model_dir, "wav2vec2_partial")).to(device)
            del model.wav2vec2.encoder.layers
            del model.wav2vec2.encoder.layer_norm
            import gc; gc.collect()
        with open(os.path.join(model_dir, "model.bin"), "rb") as m:
            model_bin = m.read()
        with open(os.path.join(model_dir, "vocabulary.json"), "rb") as v:
            vocab_json = v.read()
        ct2_model = ctranslate2.models.Wav2Vec2("model_bin", device=device.type, device_index=[0], compute_type='int8', intra_threads=4, inter_threads=4, files={"model.bin": model_bin, "vocabulary.json": vocab_json})
        while True:
            input_values, output_queue = queue.get()
            thread = Thread(target=self.fire_inference_process, args=(input_values, ct2_model, model, output_queue))
            thread.start()


    def fire_inference_process(self, input_values, ct2_model, model, output_queue):
        logits = self.inference_process(input_values, ct2_model, model)
        output_queue.put(logits)


    def load(self, domain_id, device=None):
        domain = self.get_domains()[domain_id]
        domain.device = device

        # Load the base components (domain-independent)
        if not self.loaded_base:
            # Load SAD
            self.load_base()
            self.loaded_base = True

        # Domain dependent components
        if domain_id not in self.loaded_domains:
            # Define domain
            domain = self.get_domains()[domain_id]

            # Inform user
            logger.info("")
            logger.info("Plugin Task          = {}".format(self.task))
            # label and description are from init(), not from meta.conf
            logger.info("Plugin Label         = {}".format(self.label))
            logger.info("Plugin Description   = {}".format(self.description))
            logger.info("Plugin Vendor        = {}".format(self.vendor))
            logger.info("Domain Label         = {}".format(domain.get_label()))
            logger.info("Domain Description   = {}".format(domain.get_description()))
            logger.info("Domain resample rate = {}".format(domain.get_resample_rate()))
            if domain.config.has_option('domain', 'language'):
                logger.info("Domain Language      = {}".format(domain.config.get('domain', 'language')))
            else:
                logger.info("Domain Language      = Unknown")
            logger.info("")

            # # Initialize wav2vec2 recognizer
            model_dir = domain.get_artifact('wav2vec2')
            domain.processor = AutoProcessor.from_pretrained(os.path.join(model_dir, "wav2vec2_processor"))
            domain.processor.hasLM = False
            domain.unk_char = domain.processor.tokenizer.unk_token.lower()
            unigram_list = None

            self.config.LM_file = next((fname for fname in self.config.LM_file_names if os.path.exists(os.path.join(model_dir, fname))), None)
            # it selects either 'language_model.bin' or 'language_mode.arpa' if found. When both found, 'language_model.bin' is selected
            if self.config.LM_process and self.config.LM_file is not None and os.path.exists(os.path.join(model_dir, self.config.LM_file)):
                vocab_dict = domain.processor.tokenizer.get_vocab()
                sorted_vocab_dict = {k: v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}

                if self.config.LM_file[-3:] == 'bin' and os.path.exists(os.path.join(model_dir, self.config.LM_vocab_file)):
                # 'language_model.bin' should be loaded with unigram_list for the accuracy loss prevention
                    with open(os.path.join(model_dir, self.config.LM_vocab_file), encoding='utf-8') as f:
                        unigram_list = [t for t in f.read().strip().split("\n")]
                        logger.info("Unigram is loaded from {0}".format(os.path.join(model_dir, self.config.LM_vocab_file)))

                if self.config.decode_alpha == None or self.config.decode_beta == None:
                    decoder = build_ctcdecoder(labels=list(sorted_vocab_dict.keys()), kenlm_model_path=os.path.join(model_dir, self.config.LM_file),
                                               unigrams=unigram_list, alpha=0.25, beta=1.0, unk_score_offset=self.config.unk_score_offset)
                else:
                    decoder = build_ctcdecoder(labels=list(sorted_vocab_dict.keys()), kenlm_model_path=os.path.join(model_dir, self.config.LM_file),
                                               unigrams=unigram_list, alpha = self.config.decode_alpha, beta = self.config.decode_beta, unk_score_offset = self.config.unk_score_offset)
                    logger.info("External LM {0} is configured with alpha {1} and beta {2}".format(self.config.LM_file, str(self.config.decode_alpha), str(self.config.decode_beta)))
                domain.processor = Wav2Vec2ProcessorWithLM(feature_extractor=domain.processor.feature_extractor, tokenizer=domain.processor.tokenizer, decoder=decoder)
                domain.processor.hasLM = True

                # Determine the unk character that is output
                unk_vec = np.zeros([1,domain.processor.tokenizer.vocab_size])
                unk_vec[:,domain.processor.tokenizer.unk_token_id] = 1.0
                domain.unk_char = decoder.decode(unk_vec)

            domain.sample_rate = 16000

            domain.cuda_device = self.get_cuda_device(domain_id)
            # We need to funnel the ctranslate2 work through a special process when using the CPU to prevent multiple copies
            # of the model from loading. This is an artifact of ctranslate2 internals and not being compatible with OLIVE's forked
            # worker process model.
            if domain.cuda_device == "-1":
                domain.manager = mp.Manager()
                domain.queue = domain.manager.Queue()
                domain.ctranslate2_process = mp.Process(target=self.ctranslate2_work, args=(domain.cuda_device, domain.queue, model_dir), name="ctranslate2-worker")
                domain.ctranslate2_process.start()
            
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

    def load_base(self):

        # SAD
        fmt = idento3.get_dnn_format(self.get_artifact(os.path.join("sad", "sad_dnn_nnet.npz")))
        sadextractor_config = dict(
            nnet_mvn=self.get_artifact(os.path.join("sad", "sad_dnn_mvn.npz")),
            nnet=self.get_artifact(os.path.join("sad", "sad_dnn_nnet.npz")),
            linearout=False,
            layer=-1, 
            dnn_format=fmt,
            dnn_input_config=self.get_artifact(os.path.join("sad", "sad.config")),
        )
        sadconfig = idento3.Config()
        sadconfig.update(sadextractor_config)
        self.sad_engine = idento3.Frontend(idento3.TFDnnFeatures)(sadconfig)
        # opts will be overridden in run_sad_idt()
        self.sad_engine.opts = {'filter_length': self.config.sad_filter_frames,
                                'threshold': self.config.sad_threshold,
                                'interpolate': self.config.sad_interpolate}


    def update_opts(self, opts):
        """
        Handle user defined options, check validity
        and convert from strings as needed for CLI operations.
        """

        # Copy values
        config = idento3.Config(dict(self.config))

        if opts is not None:
            # Check that all passed options are valid for this plugin
            param_check = np.in1d(list(opts.keys()), self.VALID_PARAMS)
            if np.any(~param_check):
                self.escape_with_error("Unknown parameter(s) passed {}. Please remove from the optional parameter list."
                                       .format(np.array(list(opts.keys()))[~param_check].tolist()))

            config.update(opts)

            # File-passed options are in in text format, so we need to convert these as necessary
            config.sad_threshold               = float(config.sad_threshold)
            config.sad_interpolate             = int(config.sad_interpolate)
            config.sad_filter_frames           = int(config.sad_filter_frames)
            config.sad_speech_padding_secs     = float(config.sad_speech_padding_secs)
            if type(config.use_sad) != bool:
                if config.use_sad == 'True': config.use_sad = True
                elif config.use_sad == 'False': config.use_sad = False
                else: self.escape_with_error("Parameter 'use_sad' set to '{}' but must be 'True' or 'False'".format(config.use_sad))

            config.max_segment_length_secs     = float(config.max_segment_length_secs)
            config.overlap_segment_length_secs = float(config.overlap_segment_length_secs)
            if type(config.reset_recognizer) != bool:
                if config.reset_recognizer == 'True': config.reset_recognizer = True
                elif config.reset_recognizer == 'False': config.reset_recognizer = False
                else: self.escape_with_error("Parameter 'reset_recognizer' set to '{}' but must be 'True' or 'False'".format(config.reset_recognizer))

            if type(config.unicode_output) != bool:
                if config.unicode_output == 'True': config.unicode_output = True
                elif config.unicode_output == 'False': config.unicode_output = False
                else: self.escape_with_error("Parameter 'unicode_output' set to '{}' but must be 'True' or 'False'".format(config.unicode_output))

            # Check value, None means do not normalize
            if type(config.unicode_normalization) == str:
                if config.unicode_normalization == 'None': config.unicode_normalization = None
                elif (config.unicode_normalization != 'NFC' and config.unicode_normalization != 'NFD' and
                      config.unicode_normalization != 'NFKC' and config.unicode_normalization != 'NFKD'):
                    self.escape_with_error("Parameter 'unicode_normalization' set to '{}' but must be one of None, 'None', 'NFC', 'NFD', 'NFKC' or 'NFKD'".format(config.unicode_normalization))
            elif type(config.unicode_normalization) != NoneType:
                self.escape_with_error("Parameter unicode_normalization set to '{}' but must be one of None, 'None', 'NFC', 'NFD', 'NFKC' or 'NFKD'".format(config.unicode_normalization))


            logger.debug("Using user-defined parameter options, new config is: %s" % config)

        return config
    

    # *------------------------------------------------------------------------*
    # inference process
    # *------------------------------------------------------------------------*
    def inference_process(self, input_values, ct2_model, w2v2_model=None):
        if w2v2_model: # old model type
            with torch.no_grad():
                extract_features = w2v2_model.wav2vec2.feature_extractor(
                    input_values.to(w2v2_model.device)
                ).transpose(1, 2)
                hidden_states, extract_features = w2v2_model.wav2vec2.feature_projection(
                    extract_features
                )
                position_embeddings = w2v2_model.wav2vec2.encoder.pos_conv_embed(
                    hidden_states
                )
                hidden_states = position_embeddings + hidden_states
    
            if ct2_model.device == "cuda":
                hidden_states = hidden_states.cpu()
            else:
                hidden_states = hidden_states.numpy()
    
            hidden_states = np.ascontiguousarray(hidden_states)
            hidden_states = ctranslate2.StorageView.from_array(hidden_states)
            to_cpu = (
                ct2_model.device == "cuda" and len(ct2_model.device_index) > 1
            )
            ct2_output = ct2_model.encode(
                hidden_states,
                to_cpu=to_cpu,
            )  # 24 x Wav2Vec2EncoderLayerStableLayerNorm processed
            if ct2_model.device == "cuda":
                hidden_states = torch.as_tensor(
                    ct2_output,
                    device=torch.device('{}:{}'.format(ct2_model.device, ct2_model.device_index[0])),
                )
            else:
                hidden_states = torch.as_tensor(
                    np.array(ct2_output),
                    dtype=torch.float32,
                    device=torch.device('cpu'),
                )
    
            encoder_outputs = modeling_outputs.BaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=None,
                attentions=None,
            )
            hidden_states = encoder_outputs[0]
            outputs = modeling_outputs.Wav2Vec2BaseModelOutput(
                last_hidden_state=hidden_states,
                extract_features=extract_features,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )
            hidden_states = outputs[0]
    
            with torch.no_grad():
                return w2v2_model.lm_head(hidden_states.to(torch.float32))[0]

        else: # new model type
            hidden_states = np.ascontiguousarray(input_values.unsqueeze(0))
            hidden_states = ctranslate2.StorageView.from_array(hidden_states)
            to_cpu = (ct2_model.device == "cuda" and len(ct2_model.device_index) > 1)
            ct2_output = ct2_model.encode(hidden_states,to_cpu=to_cpu)

            if ct2_model.device=="cuda":
                return torch.as_tensor(ct2_output, device=ct2_model.device)[0]
            else:
                return torch.as_tensor(np.array(ct2_output), dtype=torch.float32, device=ct2_model.device)[0]


    # *------------------------------------------------------------------------*
    # RegionScorer Methods:
    # *------------------------------------------------------------------------*

    def run_region_scoring(self, domain_id, audio, workspace, classes=None, opts=None):
        domain = self.get_domains()[domain_id]

        start_time = time.time()

        # Update options if they are passed in to get a local config
        # and pass it to any function that needs the variables.
        config = self.update_opts(opts)

        # Check sample rates.
        if (audio.sample_rate != domain.sample_rate):
            audio = idento3.Wave(audio.data.copy(), audio.sample_rate, id=audio.id).resample(domain.sample_rate)

        wav_array, wav_list, seg_list = [], "", ""
        # Trim to the region of interest if it was passed in
        regions = None
        if opts is not None:
            if 'region' in opts and 'speech_regions' in opts:
                logger.warn("Both 'region' and 'speech_regions' passed. Using only speech_regions.")
            if 'region' in opts:
                regions = opts['region']
            if 'speech_regions' in opts:
                speech_scores = opts['speech_regions'].get_result()['speech']
                if len(speech_scores[0]) == 4:
                    regions = [(x[0], x[1]) for x in speech_scores]
                else:
                    regions = speech_scores
                logger.debug(f'Received speech_regions to segment speech: {regions}')

        if regions is not None:
            for reg in regions:
                audiox = idento3.Wave(audio.data.copy(), audio.sample_rate, audio.id)
                audiox.trim_samples([reg])
                try:
                    wav_arrayx, wav_listx, seg_listx = self.run_asr_sad_tomemory(audiox, domain, workspace, config, offset=float(reg[0]), fullaudio=audio)
                except Exception as e:
                    logger.error("ASR failed to run SAD")
                    self.escape_with_error("ASR could not run internal SAD: %s." %(str(e)))
                wav_array += wav_arrayx
                wav_list += wav_listx
                seg_list += seg_listx
        else:
            try:
                wav_array, wav_list, seg_list = self.run_asr_sad_tomemory(audio, domain, workspace, config)
            except Exception as e:
                logger.error("ASR failed to run SAD")
                self.escape_with_error("ASR could not run internal SAD: %s." %(str(e)))

        # Recognize one segment at a time
        # [list of [id segN start end]]
        inputs = np.array([f.split() for f in str.splitlines(seg_list)])

        asr_out = []
        time_rec = 0
        time1 = time.time()

        for audio_segment, segname in zip(wav_array, inputs):
            # segname = ['89ddcab31989b7d657e300f130818171d0dc2c2b3205db09970abc640e9a24b7' 'seg0' '0.73' '2.94']
            logger.debug("Recognizing utterance: {}".format(segname))
            if audio_segment.duration < static_min_speech_secs:
                logger.warn("SAD region [{}, {}] shorter than minimum duration; moving to next region.".format(segname[2],segname[3]))
            else:
                # Recognize in-memory audio segment
                start = time.time()
                input_values = domain.processor([audio_segment.data.astype(np.float32)], sampling_rate=audio.sample_rate, return_tensors="pt", padding=True).input_values

                # inference process for logits
                logits = self.submit_ctranslate2_work(domain_id, input_values)
                # take argmax and decode
                predicted_ids = torch.argmax(logits, dim=-1)
 
                # perform decoding
                if domain.processor.hasLM:
                    output = domain.processor.decode(logits.cpu().numpy(), output_word_offsets=True, beam_width=25)
                else:
                    output = domain.processor.decode(predicted_ids, output_word_offsets=True)

                transcription = output['text']
                duration_sec = input_values.shape[1] / audio.sample_rate
                predicted_ids = predicted_ids.tolist()
                this_asr = []

                for item in output.word_offsets:
                    begin_time = ( item['start_offset'] / len(predicted_ids) ) * duration_sec
                    end_time = ( item['end_offset'] / len(predicted_ids) ) * duration_sec
                    word = item['word'].lower()
                    segment = (segname[0] +'.'+ segname[1], str(float(segname[2])+begin_time), str(float(segname[2])+end_time), word, 1.000000)
                    this_asr.append(segment)

                if asr_out and this_asr and float(this_asr[0][1]) < float(asr_out[-1][2]):
                    for _ in range(len(asr_out)):
                        if asr_out and float(this_asr[0][1]) < float(asr_out[-1][1]):
                            asr_out = asr_out[:-1]
                        else:
                            break
                    for _ in range(len(this_asr)):
                        if asr_out and this_asr and float(asr_out[-1][2]) > float(this_asr[0][1]):
                            this_asr = this_asr[1:]
                        else:
                            break

                asr_out = asr_out + this_asr

                time_rec += time.time() - start

        time2 = time.time()

        if len(asr_out) > 0:

            if self.config.remove_unk_char and domain.unk_char is not None:
                asr_result = {}
                for index, [segment, st, et, wd, conf] in enumerate(asr_out):
                    if domain.unk_char in wd:
                        newwd = re.sub(r'{}*[\[\]0-9]*'.format(domain.unk_char), '', wd).replace(r'[unk]','') # removing both [unk] and ⁇unk⁇0000
                        if len(newwd) > 0:
                            logger.warn("UNKNOWN CHARACTERS {} changed to {}".format(wd, newwd))
                            asr_result[index] = [(np.float32(st), np.float32(et), newwd, conf)]
                        else:
                            logger.warn("UNKNOWN CHARACTERS {} dropped".format(wd))
                    else:
                        asr_result[index] = [(np.float32(st), np.float32(et), wd, conf)]
            else:
                asr_result = {index:[(np.float32(st), np.float32(et), wd, conf)] for index, [segment, st, et, wd, conf] in enumerate(asr_out)}

            logger.debug("asr_result = {}".format(asr_result))

        else:
            asr_result = {}
            logger.debug("No transcription found in this file!")

        end_time = time.time()
        logger.info("TOTAL TIME RUN_REGION_SCORING: "+str(end_time-start_time))
        logger.info("STEPS TIME RUN_REGION_SCORING: "+str(time1-start_time)+" "+str(time2-time1)+" "+str(end_time-time2))
        logger.info("REC_ONLY TIME RUN_REGION_SCORING: "+str(time_rec))

        return asr_result


    # *------------------------------------------------------------------------*
    # SAD Methods:
    # *------------------------------------------------------------------------*

    def run_sad_idt(self, domain, audio, config, offset=0.0):
        """
        Run SAD to get speech regions.

        Input:
        - domain plugin domain
        - audio audio object
        - offset (optional) added the segment times

        Output:
        Returns speech_regions as list of float (start_time, end_time) tuples,
        for example [(0.56, 3.0), (3.04, 6.38)]

        If regions can't be computed because duration is less than
        sad_minimum_duration, function returns None.

        If no speech regions are detected (because the LLR is too low),
        function returns None.
        """

        # retrieve SAD parameters from config
        sad_minimum_duration = static_min_speech_secs
        sad_speech_padding_secs = config.sad_speech_padding_secs
        sad_speech_llr_threshold = config.sad_threshold

        # override default values from config
        sad_engine_opts = {'filter_length': config.sad_filter_frames,
                           'threshold': config.sad_threshold,
                           'interpolate': config.sad_interpolate}

        # Catch minimum waveform duration
        duration = float(audio.data.shape[0]) / float(audio.sample_rate)
        if duration < sad_minimum_duration:
            logger.debug("WARNING: run_sad: waveform too short ({} < {}) - not computing SAD!".format(duration, sad_minimum_duration))
            return None

        # Now run SAD
        waveobject = idento3.Wave(audio.data.copy(), audio.sample_rate, audio.id)
        speech = idte.dnn_vad(self.sad_engine.feat_fc(waveobject), self.sad_engine.embedextractor,
                              return_llrs=False, **sad_engine_opts)

        if sad_speech_padding_secs > 0.0:
            speech = idento3.pad(speech[np.newaxis,:], sad_speech_padding_secs).ravel()

        if speech.sum() == 0:
            logger.debug(
                "Audio contained no speech for threshold {} and padding {}".format(sad_speech_llr_threshold,
                                                                                   sad_speech_padding_secs))
            return None

        align = idento3.Alignment()
        if offset > 0.0:
            speech = np.hstack([np.zeros(int(offset*idento3.DEFAULT_FRAMES_PER_SECOND), dtype=bool), speech])
        align.add_from_indices('temp', speech, 'speech')
        segment_array = align.get_start_end('temp', unit='seconds')
    
        # Returns speech_regions as list of float (start_time, end_time) tuples, for example [(0.56, 3.0), (3.04, 6.38)]
        return segment_array

    def convertSegments_tomemory(self, audio, domain, workspace, segments_array, config):
        """Processes audio and list of (start,end) tuples into objects more
        useful for recognition.

        Any segment that is longer than the maximum duration is split
        into multiple segments with overlap.

        Function is complicated by need to possibly resample audio, be memory efficient.

        Outputs (one per input segment):
        - wav_array:
          List of idento3.audio.wavio.Wave objects
        - wav_list:
          Space-separated string mapping "segment name" as "audioID.segN" to
          temporary filename in workspace (as "/path/to/WORK/file_id/audioID.segN.wav"),
          one line per segment.
        - seg_list:
          Space-separated string with 4 elements per line and one line per segment,
          "audioID segN start end"

        Returns [], "", "" if no segments
        """
        wav_array = []
        wav_list = ""
        seg_list = ""

        iseg_extras = len(segments_array) - 1
        segments_array_enum = enumerate(segments_array)

        # Possibly resample audio file, but only once:
        resampled = False
        local_audio = audio
        if (audio.sample_rate != domain.sample_rate):
            if (domain.sample_rate > audio.sample_rate):
                self.escape_with_error("Cannot recognize data, audio sample rate {} < recognizer sample rate {}!".format(audio.sample_rate, domain.sample_rate))

            new_audio_data = audio.data.copy()
            if (domain.sample_rate < audio.sample_rate and audio.sample_rate % domain.sample_rate == 0):
                # returns Wave object, type int16 if input was int16
                logger.debug("Downsampling output sampling rate to {} Hz.".format(domain.sample_rate))
                local_audio = idento3.Wave(new_audio_data, audio.sample_rate, id=audio.id).downsample(domain.sample_rate)
            else:
                # returns Wave object, type float64
                # need data to be int16 format...
                logger.debug("Resampling output sampling rate to {} Hz.".format(domain.sample_rate))
                local_audio = idento3.Wave(new_audio_data, audio.sample_rate, id=audio.id).resample(domain.sample_rate).unnormalize()
            resampled = True

        # Trim segments to maximum length (with overlap)
        for iseg,seg in segments_array_enum:
            start = seg[0]
            end = seg[1]
            while end - start > config.max_segment_length_secs:
                iseg_extras += 1
                SEG = (start, start + config.max_segment_length_secs)
                x = idento3.Wave(local_audio.data.copy(), local_audio.sample_rate)
                x.trim_samples([SEG])
                seg_list += "{0} seg{1} {2} {3}\n".format(audio.id, iseg_extras, SEG[0], SEG[1])
                wav_list += "{0}.seg{2} {1}/{0}.seg{2}.wav\n".format(audio.id, workspace, iseg_extras)
                wav_array.append(x)
                del x
                start += config.max_segment_length_secs - config.overlap_segment_length_secs

            seg = (start,seg[1])
            if seg[1] != seg[0]:
                x = idento3.Wave(local_audio.data.copy(), local_audio.sample_rate)
                logger.debug("WAVE: {} {}".format(seg, x.duration))
                x.trim_samples([seg])
                seg_list += "{0} seg{1} {2} {3}\n".format(audio.id, iseg, seg[0], seg[1])
                wav_list += "{0}.seg{2} {1}/{0}.seg{2}.wav\n".format(audio.id, workspace, iseg)
                wav_array.append(x)
                del x
            else:
                logger.warn("Removing SAD frame in isolation at timestamp {}".format(seg[0]))

        # Delete local (downsampled) copy of audio:
        if resampled is True:
            logger.debug("Deleting resampled audio data!")
            del new_audio_data, local_audio

        return wav_array, wav_list, seg_list

    def run_asr_sad_tomemory(self, audio, domain, workspace, config, offset=0.0, fullaudio=None):
        """
        Optionally run SAD to compute speech segments, then convert to
        in-memory format for recognition and other processing.

        Inputs:
        - audio object
        - domain
        - workspace (to save temporary and other artifacts)
        - offset (optional)
        - fullaudio (optional - required if offset>0.0 and config.use_sad)

        Outputs (one per input segment):
        - wav_array:
          List of idento3.audio.wavio.Wave objects
        - wav_list:
          Space-separated string mapping "segment name" as "audioID.segN" to
          temporary filename in workspace (as "/path/to/WORK/file_id/audioID.segN.wav"),
          one line per segment.
        - seg_list:
          Space-separated string with 4 elements per line and one line per segment,
          "audioID segN start end"
          Offset is added to start and end

        NOTE: We don't actually write the temporary waveform files, we
        only generate their filenames!  Instead, we do all audio processing
        in memory.

        Returns [], "", "" if no segments
        """

        if config.use_sad:
            # Run SAD.
            # Convert to empty list if audio too short or no segments found.
            if offset > 0.0 and fullaudio is None:
                self.escape_with_error("The full audio file must be passed if offset>0.0")

            try:
                # Returns [] if no segments
                segments_array = self.run_sad_idt(domain, audio, config, offset=offset)
                logger.debug("segments_array = {}".format(segments_array))
                if (segments_array is None):
                    segments_array = []
            except Exception as e:
                logger.error("run_asr_sad_tomemory: ASR failed to run SAD")
                self.escape_with_error("KWS could not run internal SAD: %s." % (str(e)))

            # Returns [], "", "" if no segments
            if fullaudio is None:
                wav_array, wav_list, seg_list = self.convertSegments_tomemory(audio, domain, workspace, segments_array, config)
            else:
                wav_array, wav_list, seg_list = self.convertSegments_tomemory(fullaudio, domain, workspace, segments_array, config)
        else:
            # Not using SAD, so create artificial output that contains entire waveform
            # Can also use:
            # seg_ready = [(0, int(100 * audio.get_duration()), 'speech')]
            # segments_array = [(s[0]/100., min(audio.get_duration(),(s[1]+1)/100.)) for s in seg_ready]
            segments_array = [(offset, offset + audio.duration)]

            # Returns [], "", "" if no segments
            wav_array, wav_list, seg_list = self.convertSegments_tomemory(audio, domain, workspace, segments_array, config)

        return wav_array, wav_list, seg_list

    # *------------------------------------------------------------------------*
    # Plugin options
    # *------------------------------------------------------------------------*

    def get_region_scoring_opts(self):
        """
        Implement plugin options. These options are used in the OLIVE GUI
        and may be configured on the commandline by passing a file to
        --options.

        The GUI intends to query a plugin to see what a user can
        change, the default values and the options if available. The
        defaults should come from self.config and the two TraitOptions
        are CHOICE_TRAIT and BOOLEAN_TRAIT. The example below is for
        region scoring, but the word ‘region’ can be replaced with
        global or frame as needed for your plugin. The parameters
        defined here should ONLY be the ones exposed in
        plugin_config.py for the user. Note the list passed as the
        options for sad_interpolate in the example which restricts the
        choices for the user.
        """

        # ASR/Plugin:
        region_scoring_trait_options = [
            TraitOption('sad_threshold', "Threshold to determine speech frames", "Higher value results in less speech for processing (default 0.0)", TraitType.CHOICE_TRAIT, "", self.config.sad_threshold),
            TraitOption('unicode_output', "Unicode Output", "Convert intermediate output to Unicode if language supports it", TraitType.BOOLEAN_TRAIT, "", self.config.unicode_output),
            TraitOption('unicode_normalization', "Unicode Normalization", "Unicode Normalization to apply", TraitType.CHOICE_TRAIT, list(["None", "NFC", "NFD", "NFKC", "NFKD"]), self.config.unicode_normalization),
        ]

        return region_scoring_trait_options

# This line is very important! Every plugin should have one
plugin = CustomPlugin()

