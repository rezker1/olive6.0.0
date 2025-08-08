import os, numpy as np, idento3, importlib
from olive.plugins import Plugin, FrameScorer, SupervisedAdapter, RegionScorer, TraitOption, \
        TraitType, logger, Domain
from idento3 import Alignment, Feature, Frontend, Segmentation, NoLabelException, \
        NoSegmentationException, Config, External, TFDnnFeatures, get_dnn_format, Wave
import upfirdn
import time
import idento3.engines as idte
from nn import NET, DNN
from idento3.utils import logit

import threading

## SAD testing setttings
TARGET = -1
#FILTER_LENGTH = 51
MAX_BUFFER = 1000000
MAX_CACHE_SEC = 300  # Memory reduction through cached processing
USE_BUFFERING = True
#NOISE_LEVEL = 100  # Noise level for dithering
MIN_DUR = 0.31  # Mininmum duration in seconds
APPLY_THRESHOLD_PADDING = 0.3

## SAD supervised adapter setttings
INCREMENT_SUP = 500000
BATCHSIZE_SUP = 10000
DECI_SUP = 1
CLASSMAP = {'NS': 0, 'S': 1}
LEARN_RATE = 0.1
START_HALVING = 2.0
END_HALVING = 0.5
NUM_CLASSES = 2
REG_PARAM = 0.0
MOMENTUM = 0.0
MIN_EPOCH = 3
MAX_EPOCH = 6
MIN_ADAPT_DUR = 3 # Minimum adaptation duration in seconds
TESTED_ADAPT_DUR = 60 # Suggested minimum adaptation in seconds

FRAME_SCORE_OPTION_FILTER_LENGTH = "filter_length"
FRAME_SCORE_OPTION_INTERPOLATE = "interpolate"

import torch

##################################################
# CONFIG - DO NOT CHANGE THESE UNLESS TOLD TO DO SO BY A DEVELOPER
default_config = Config(dict(
# Configurable parameters
threshold = 0.0,
filter_length = 51,
llr_offset = 0.0
))
##################################################

class CustomPlugin(Plugin, FrameScorer, SupervisedAdapter, RegionScorer):
    def __init__(self):
        self.task = "SAD"
        self.label = "SAD DNN (Commercial)"
        self.description = "A fast, single-feature DNN SAD model"
        self.vendor = "SRI"
        self.version = '1.0.0'
        self.minimum_runtime_version = '5.7.0'
        self.minimum_olive_version = '5.7.0'
        self.group = "Speech"
        self.create_date = "2025-03-20"
        self.revision_date = "2025-03-20"
        self.loaded_domains  = []
        self.config          = default_config
        loader               = importlib.machinery.SourceFileLoader('plugin_config', os.path.join(os.path.dirname(os.path.realpath(__file__)), 'plugin_config.py'))
        spec                 = importlib.util.spec_from_loader(loader.name, loader)
        mod                  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.config.update(mod.user_config)
        self.store = threading.local()

    ### Method implementations required of a Plugin ###
    def list_classes(self, domain_id):
        domain = self.get_domains()[domain_id]
        if domain.config is not None:
            import olive.plugins as op
            try:
                return [domain.config.get('domain', 'class_id')]
            except:
                return ["speech"]
            
    def load(self, domain_id, device=None):
        domain = self.get_domains()[domain_id]
        domain.device = device

        if domain_id not in self.loaded_domains:
            # Define domains
            domain = self.get_domains()[domain_id]

            # Get domain config
            domain.usr_config = None
            if os.path.exists(domain.get_artifact("domain.config")):
                domain.usr_config = Config(dict(self.config))
                domain.usr_config.update_with_file(domain.get_artifact("domain.config"))

            dnn_file = domain.get_artifact("nn.pnn")
            mvn_data = np.load(domain.get_artifact("mvn.npz"))
            domain.nnet = NET.load_from_file(dnn_file)
            domain.mean, domain.inv_std = mvn_data['mean'], mvn_data['inv_std']

            # Feature config
            sadconfig = Config()
            sadconfig.update_with_file(domain.get_artifact("sad.config"))
            domain.sadconfig = sadconfig
            domain.fc = Frontend(sadconfig.featclass)(sadconfig)
            domain.padding, domain.interpolate, domain.window, domain.overlap = \
                    domain.fc.config.vad_padding, \
                    domain.fc.config.vad_interpolate, \
                    domain.fc.config.window, \
                    domain.fc.config.overlap

            # Define domain.llrframespersecond whether it’s in the sad_config or not
            domain.llrframespersecond = idento3.DEFAULT_FRAMES_PER_SECOND
            if 'vad_llrframespersecond' in domain.fc.config.keys():
                domain.llrframespersecond = domain.fc.config.vad_llrframespersecond

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

    def buffered_call(self, fnc, data, buffer_size, overlap, feat_shift, *args, **kwargs):
        """
        Allows fnc to be called multiple times by chunking data, then stitch the result together
        """

        # Determine how many splits of data to process
        datalen = data.shape[0]
        if datalen > buffer_size:
            nbatch = np.round(datalen / float(buffer_size) + 1).astype(int)
        else:
            nbatch = 1

        # Calculate the shift in 'data' per batch
        shift = datalen / float(nbatch)
        # Correct shift to be divisable by the feat_shift
        if shift % feat_shift > 0: shift = int(shift - shift % feat_shift + feat_shift)

        # A list to collect results. Could be more efficient by predefining an array if we had the information on output length
        result = []
        for ibatch in np.arange(nbatch).astype(int):
            # Pass each batch of data to the fnc
            bstart = max(0, int(ibatch * shift))
            bend = min(int((ibatch + 1.0) * shift) + overlap, datalen - 1)
            res = fnc(data[bstart:bend].copy(), *args, **kwargs)

            # Now determine the values for trimming the batch result
            if ibatch == 0:
                samples_per_frame = int((bend - bstart) / res.shape[0])
                frame_overlap = overlap // samples_per_frame
                # These are the number of frames consumed by the process due to window. 
                # In features, you may only get 998 frames for 10 seconds of audio.
                missing_in_action = ((bend - bstart) - (bend - bstart) // res.shape[0] * res.shape[
                    0]) // samples_per_frame
                trim_offset = 0
            else:
                trim_offset = frame_overlap // 2
            strim = trim_offset
            etrim = int(res.shape[0]) - frame_overlap // 2 + missing_in_action if bend < datalen - 1 else int(res.shape[0])
            result.append(res[strim:etrim])
            if etrim == res.shape[0]: break  # In case batch size is smaller than overlap
        return np.hstack(result)

    def update_opts(self, opts, domain):

        # Copy values
        if domain.usr_config is None:
            config = Config(dict(self.config))
        else:
            config = Config(dict(domain.usr_config))

        if opts is not None:
            config.update(opts)
            # File-passed options are in text format, so we need to conver these as necessary
            config.threshold = float(config.threshold)
            config.llr_offset = float(config.llr_offset)
            config.filter_length = int(config.filter_length)

        return config

    ### RegionScorer ###
    def run_region_scoring(self, domain_id, audio, workspace, classes=None, opts=None):

        domain = self.get_domains()[domain_id]
        config = self.update_opts(opts, domain)

        segments = []
        timing_map = None
        if opts is not None and 'region' in opts:
            timing_map = opts['region']
            del opts['region']  # So frame scoring doesn't see it as we'll handle it here

        if timing_map is None:
            frame_sad_scores = self.run_frame_scoring(domain_id, audio, workspace, classes, opts)
            sad_regions = self.apply_threshold(frame_sad_scores[self.list_classes(domain_id)[0]], config.threshold)
            for start, end in sad_regions:
                segments.append((start, end, self.list_classes(domain_id)[0], config.threshold))
        else:
            offset = 0
            audio_orig = audio.data.copy()
            for reg in timing_map:
                audio = idento3.Wave(audio_orig.copy(), audio.sample_rate, audio.id)
                audio.trim_samples([reg])
                offset = float(reg[0])
                if audio.duration < MIN_DUR:
                    logger.warn("Region [{}, {}] shorter than minimum duration; moving to next region.".format(reg[0],reg[1]))
                else:
                    frame_sad_scores = self.run_frame_scoring(domain_id, audio, workspace, classes, opts)

                    sad_regions = self.apply_threshold(frame_sad_scores[self.list_classes(domain_id)[0]], config.threshold)
                    for start, end in sad_regions:
                        segments.append((offset + start, offset + end, self.list_classes(domain_id)[0], config.threshold))
        return {self.list_classes(domain_id)[0]: segments}

    ### FrameScorer ###
    def run_frame_scoring(self, domain_id, audio, workspace, classes=None, opts=None):

        # Update the self.config variables to use any options the user passed in for this job
        # This will possibly update self.config.threshold
        domain = self.get_domains()[domain_id]
        config = self.update_opts(opts, domain)

        fc = Frontend(domain.sadconfig.featclass)(domain.sadconfig)

        #import torch
#        os.environ['CUDA_VISIBLE_DEVICES'] = domain.cuda_device
        # Device to run on 
        if not hasattr(domain, 'cuda_device'):
            domain.cuda_device = self.get_cuda_device(domain_id)
#        if use_cuda and 'CUDA_VISIBLE_DEVICES' in os.environ and os.environ['CUDA_VISIBLE_DEVICES'] != "":
        if domain.cuda_device != "-1":
            device = torch.device('cuda:{}'.format(domain.cuda_device))
        else:
            device = torch.device('cpu')
        config.device = device
        domain.nnet = domain.nnet.to(device)
        domain.nnet.eval()

        unnormed = audio.is_unnormalized
        if unnormed:
            audio.normalize()

        audios = []
        llrs_list = []
        # Trim to the region of interest if it was passed in
        if opts is not None and 'region' in opts:
            for reg in opts['region']:
                audiox = idento3.Wave(audio.data.copy(), audio.sample_rate, audio.id)
                audiox.trim_samples([reg])
                if audiox.duration < MIN_DUR:
                    self.escape_with_error("Region [{}, {}] shorter than minimum duration [{}]".format(reg[0], reg[1], MIN_DUR))
                audios.append(audiox)
        else:
            audios.append(audio)

        for audio in audios:
            if audio.duration < MIN_DUR:
                self.escape_with_error("ERROR: File shorter than minimum duration")

            # Compute LLRs via the normal process
            else:

                # Provide better validation?
                #                if opts is not None and FRAME_SCORE_OPTION_FILTER_LENGTH in opts:
                #filter = int(opts[FRAME_SCORE_OPTION_FILTER_LENGTH])
                #                else:
                filter = config.filter_length
                debug = True
                if opts is not None and FRAME_SCORE_OPTION_INTERPOLATE in opts:
                    interpolate = int(opts[FRAME_SCORE_OPTION_INTERPOLATE])
                else:
                    interpolate = domain.interpolate

                logger.debug("Using filter length of: {}".format(filter))
                logger.debug("Using interpolate value of: {}".format(interpolate))

                # DNN evaluation

                # A buffered call to produce the features
                batch_size = 1 * MAX_CACHE_SEC * audio.sample_rate  # MAX_CACHE_SEC seconds of audio
                overlap = audio.sample_rate*2  # overlap of two seconds

                # Define a batch-able function
                def get_nnet_llrs(batch_data, device, interpolate=interpolate):
                    # Compute features
                    batch_audio = Wave(batch_data, audio.sample_rate)
                    # Filter is 1 here since it's applied later
                    vad_opts = {'filter_length': 1, 'interpolate':interpolate, 'threshold':0.0} # Filter override to be applied after buffered call
                    #_, llrs = idte.dnn_vad(domain.sad_engine.feat_fc(batch_audio), domain.sad_engine.embedextractor, return_llrs=True, speech_indices=[TARGET], **vad_opts)

                    feat = fc(batch_audio)
                    feat -= domain.mean
                    feat *= domain.inv_std
                    feat = torch.tensor(feat)
                    feat = feat.to(device)
#                    feat = domain.sad_engine.embedextractor.mvn(feat, domain.sad_engine.embedextractor.mean, domain.sad_engine.embedextractor.inv_std)
                    #nnet = NET.load_from_file(domain.sad_engine.nn)

                    if False:
                        # Used to convert the DNN to PNN
#                        config = {'nntype': DNN, 'hidden_layers_sizes': (500, 500, 500), 'hidden_activations': 'Sigmoid'}
                        config = {'nntype': DNN, 'hidden_layers_sizes': (500, 100), 'hidden_activations': 'Sigmoid'}
#                        dnn = DNN(config, (620,), 2)
                        dnn = DNN(config, (150,), 2)
                        #m = np.load(domain.sad_engine.nn, encoding='latin1', allow_pickle='True')
                        m = np.load(filename, encoding='bytes', allow_pickle='True')
                        s = dnn.model.state_dict()
                        if old:
                            keys = list(s.keys())
                            params = {}
                            cnt = 0
                            for y in list(m['arr_0']):
                                if len(y.shape)==2:
                                    params[keys[cnt]] = torch.tensor(y).T
                                else:
                                    params[keys[cnt]] = torch.tensor(y)
                                cnt+=1
                        else:
                            param_list = []
                            for y in np.arange(len(m['params'])):
                                for x in m['params'][y]:
                                    param_list.append(torch.Tensor(x.T))
                            params = dict(zip(s.keys(), param_list))
                        dnn.model.load_state_dict(params)
                        dnn.save('temp.pnn')

                    interpolate = vad_opts['interpolate']
                    filter_length = vad_opts['filter_length']
                    with torch.no_grad():
                        _, logits = domain.nnet(feat[0::vad_opts['interpolate']])
                    complete = feat.shape[0]
                    llrs = logit(logits[:,TARGET].detach().cpu())
                    # Interpolate if needed
                    if interpolate != 1:
                        w = np.r_[np.linspace(0, 1, interpolate+1), np.linspace(1, 0, interpolate+1)[1:]][1:-1]
                        llrs = upfirdn.upfirdn(llrs, w, interpolate)[interpolate - 1: interpolate - 1 + complete]

                    # filt llr
                    winm = np.ones(filter_length)
                    llrs = np.convolve(llrs, winm/np.sum(winm), 'same')

                    return llrs

                # Change from 'all at once' to 'buffered' calling of SAD
                if not USE_BUFFERING:
                    llrs = get_nnet_llrs(audio.data, config.device)
                else:
                    audio = fc.audio_func(audio, channels=None, bounds=None, **fc.config)
                    def passthrough(wave, **kwargs):
                        return wave
                    oldfnc = fc.audio_func
                    fc.audio_func = passthrough
                    llrs = self.buffered_call(get_nnet_llrs, audio.data, batch_size, overlap,
                            domain.window - domain.overlap, config.device)
                    fc.audio_func = oldfnc

                # For multi-fast-v2 domain: Convert 50 frames/sec to 100 frames/sec
                if domain.llrframespersecond != idento3.DEFAULT_FRAMES_PER_SECOND:
                    if (idento3.DEFAULT_FRAMES_PER_SECOND % domain.llrframespersecond) != 0:
                        self.escape_with_error("The frame rate [%d] must be divisible by vad_llrframespersecond. Chose a different vad_llrframespersecond." % idento3.DEFAULT_FRAMES_PER_SECOND)
                    interpol_factor = int(idento3.DEFAULT_FRAMES_PER_SECOND / domain.llrframespersecond)
                    w = np.r_[np.linspace(0, 1, interpol_factor + 1), np.linspace(1, 0, interpol_factor + 1)[1:]][1:-1]
                    llrs = upfirdn.upfirdn(llrs, w, interpol_factor)[interpol_factor - 1: interpol_factor - 1 + 2*llrs.shape[0]]

                # Smoothing
                if filter > 0:
                    if llrs.shape[0] < float(filter):
                        filter = llrs.shape[0]
                    llrs = np.convolve(llrs, np.ones(filter) / float(filter), 'same')

            # LLR calibration for threshold 0.0
            llrs = llrs - config.llr_offset

            # Map to correct values and pad ends so that the full audio duration is annotated
            target_shape = int(audio.duration * idento3.DEFAULT_FRAMES_PER_SECOND)
            missing = int(target_shape - llrs.shape[0])
            add = missing // 2
            extra = missing - add
            llrs_list.append(np.hstack([[llrs[0]]*add, llrs, [llrs[-1]]*extra]))

        return {self.list_classes(domain_id)[0]: np.hstack(llrs_list).tolist()}

    def apply_threshold(self, llrs, threshold):
        result = []
        align = Alignment()
        speech = llrs > np.float64(threshold)
        speech[0] = 0  # so that it never starts with a speech frame
        align.add_from_indices("isolated", speech, 'speech')

        # handle post processing
        align.pad(APPLY_THRESHOLD_PADDING)

        # Try/catch exception for non speech segments
        try:
            for segment in align.get_start_end('isolated', unit='seconds'):
                result.append((segment[0], segment[1]+0.01))
        except:
            # no segments
            logger.warn("Sample contained no speech segments")

        return result

    def get_region_scoring_opts(self):
        return [TraitOption(FRAME_SCORE_OPTION_FILTER_LENGTH, "Filter length", "filter_length", TraitType.CHOICE_TRAIT,
                            list(range(1, 102, 2)), self.config.filter_length),
                TraitOption('threshold', "Threshold to determine speech regions", "Higher value is more strict when detecting speech", TraitType.CHOICE_TRAIT, "", self.config.threshold),
                TraitOption(FRAME_SCORE_OPTION_INTERPOLATE, "Interpolate", "interpolate", TraitType.CHOICE_TRAIT,
                            [1,2,4,8,16], 4)]

    def get_frame_scoring_opts(self):
        return [TraitOption(FRAME_SCORE_OPTION_FILTER_LENGTH, "Filter length", "filter length", TraitType.CHOICE_TRAIT,
                            list(range(1, 102, 2)), self.config.filter_length),
                TraitOption(FRAME_SCORE_OPTION_INTERPOLATE, "Interpolate", "interpolate", TraitType.CHOICE_TRAIT,
                            [1,2,4,8,16], 4)]

    def get_frame_rate(self):
        return 100

    def get_frame_offset(self):
        return 0.0

# This line is very important! Every plugin should have one
plugin = CustomPlugin()
