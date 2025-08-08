import glob
import os
import re
import sys
import time
import numpy as np
import importlib
import upfirdn
import idento3 as idt
import idento3.engines as idte
from typing import Dict, List, Tuple, Union, Optional
from olive.plugins import logger, Plugin, GlobalScorer, ClassModifier, utils, TraitOption, TraitType, shutil

from transformers import AutoModelForAudioXVector, AutoFeatureExtractor

from nn import NET, DNN
from idento3.utils import logit
import torch

import scipy.spatial.distance
from sklearn import mixture
import fastcluster
import scipy.cluster.hierarchy

try:
    x = idt.DEFAULT_FRAMES_PER_SECOND
except AttributeError:
    idt.DEFAULT_FRAMES_PER_SECOND = 100

default_config = idt.Config(dict(
##################################################
# Configurable [global scoring] parameters
# WARNING - these should be changed in plugin_config.py, not here

# DETECTION OPTIONS
min_speech = 0.2,

# SAD
sad_threshold   = 0.0,
sad_filter      = 31,
sad_interpolate = 4,
sad_merge       = 0.2,
sad_padding     = 0.1,

min_detection_speech = 2.0,
ok_detection_speech  = 10.0,
tgt_detection_speech = 30.0,

output_per_seconds   = 2.0,

# Define the emotions enabled for each domain in ./domains/$domain/domain_config.txt
emo_padding = 0.5,
speaker_threshold = 0.0,
manual_calibration = {'normal':1.0},
##################################################
))


static_chunk_len_in_secs = 30
static_sample_rate = 16000
static_batch_size = 1
static_normalize_chunk = True

MAX_BUFFER_DURATION = 45.0
# Dialect mapping to base emotion if available
static_enable_emotion_mapping = True
static_precompute_enroll_stats = True  # Used with PLDA for faster scoring
static_max_calibration_segments_per_emotion = 1000  # Subset limit when enrolling/calibrating. Full set is used for backend model training.
static_backend = 'GB'  # 'PLDA' or 'GB'
# For instances of when a user enrolls data for which a pre-enrolled emotion exists. USER: Ignore pre-enrolled data; AUGMENT: Combine the data sources.
static_emotion_data_use = 'AUGMENT' #'AUGMENT'  # 'USER' or 'AUGMENT' or 'USER_ONLY'
#static_emotion_data_use = 'USER_ONLY' #'AUGMENT'  # 'USER' or 'AUGMENT' or 'USER_ONLY'
static_fusion_domains = ['embed-v1', 'wav2vec-v1']

# HELPER CLASS
# A simple iterator class to return successive chunks of samples

class AudioChunkIterator(torch.utils.data.IterableDataset):
    def __init__(self, samples: Union[np.ndarray, torch.Tensor], chunk_len_in_samples: int, normalize: bool = True):
        self._samples = samples
        self._chunk_len = chunk_len_in_samples
        self._start = 0
        self.output = True
        self.normalize = normalize

    def __iter__(self):
        return self

    def __next__(self):
        if not self.output:
            raise StopIteration
        last = int(self._start + self._chunk_len)
        if last < len(self._samples):
            chunk = self._samples[self._start: last]
            if self.normalize:
                chunk = (chunk - chunk.mean()) / np.sqrt(chunk.var() + 1e-5)
            self._start = last
        else:
            chunk = self._samples[self._start:]
            samp_len = len(self._samples) - self._start
            if self.normalize:
                chunk = (chunk - chunk[:samp_len].mean()) / np.sqrt(chunk[:samp_len].var() + 1e-5)
            self.output = False
        return chunk

class CustomPlugin(Plugin, GlobalScorer, ClassModifier):


    def __init__(self):
        self.task  = "EMO"
        self.label = "Emotion Recognition - Embeddings with Gaussian Backend (Commercial)" 
        self.description = "Emotion identification using embeddings and Gaussian Backend"
        self.vendor = "SRI"
        self.version = '1.0.0'
        self.minimum_runtime_version = '5.4.0'
        self.minimum_olive_version = '5.4.0'
        self.group = "Speech"
        self.create_date = "2023-03-29"
        self.revision_date = "2022-03-29"
        self.loaded_domains = []
        self.config          = default_config
        loader               = importlib.machinery.SourceFileLoader('plugin_config', os.path.join(os.path.dirname(os.path.realpath(__file__)), 'plugin_config.py'))
        spec                 = importlib.util.spec_from_loader(loader.name, loader)
        mod                  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.config.update(mod.user_config)
        self.VALID_PARAMS    = ['region','speaker_id'] + list(self.config.keys()) # For checking user inputs and flagging unknown paramters. Region is passed with 5-column enrollment
        sys.path.insert(0,os.path.join(os.path.dirname(os.path.realpath(__file__))))
        loader               = importlib.machinery.SourceFileLoader('models_hf', os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.join('models_hf', '__init__.py')))
        spec                 = importlib.util.spec_from_loader(loader.name, loader)
        mod                  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)


    ### Method implementations required of a Plugin ###
    def list_classes(self, domain_id, enrolled=True, included=True, is_mapped=True, return_maps=False, exclude_spks=False, exclude_emos=False):

        if not enrolled and not included:
            self.escape_with_error("At least one of enrolled or included must be True")

        # User enrollments
        user_emos = []
        if enrolled:
            enrollment_dir = self.get_enrollment_storage(domain_id)
            user_emos = [x for x in os.listdir(enrollment_dir) if x[:6] != 'models']
            if exclude_spks:
                user_emos = [x for x in user_emos if x[:4] != 'user']
            if exclude_emos:
                user_emos = [x for x in user_emos if x[:4] == 'user']

        # Pre-enrolled emotions
        domain_emos = []
        domain = self.get_domains()[domain_id]
        if included:
            domain_config_file = domain.get_artifact('domain_config.txt')
            enabled_pre_enrolled_emotions = [x.strip() for x in open(domain_config_file) if '#' not in x]
            cal_domain_emos = idt.read_data_in_hdf5(domain.get_artifact("base_data.h5"), nodes=['/emotions'])[0].tolist()
            # Check that the user has only set valid emotions:
            if not np.all(np.in1d(enabled_pre_enrolled_emotions, cal_domain_emos)):
                self.escape_with_error("Config contains emotions in file %s that are not available. Requested: [%s]. Available: [%s]" % (domain_config_file, ','.join(enabled_pre_enrolled_emotions), ','.join(cal_domain_emos)))
            domain_emos = [x for x in cal_domain_emos if x in enabled_pre_enrolled_emotions]

        dom_classes = np.unique(user_emos + domain_emos).tolist()

        # Handle mapping if available
        dialect_map, dialect_map_inv = None, None
        if static_enable_emotion_mapping and is_mapped and os.path.exists(domain.get_artifact('emotion.map')):
            dialect_map = dict(zip(dom_classes, dom_classes))
            # Ensure user-enrolled emotions are not mapped as we assume they are by definition important as is
            if not hasattr(domain, 'map_from_file'):
                map_from_file = dict(np.atleast_2d(np.loadtxt(domain.get_artifact('emotion.map'), dtype=str)))
            else:
                map_from_file = dict(domain.map_from_file)
            for emo in user_emos:
                if emo in map_from_file.keys():
                    del map_from_file[emo]
            dialect_map.update(map_from_file)
            dialect_map_inv = np.array(list(zip(dialect_map.values(), dialect_map.keys())))
            # Map the emotions to the output labels
            dom_classes = np.unique([dialect_map[x] for x in dom_classes]).tolist()

        if return_maps:
            return dom_classes, dialect_map, dialect_map_inv
        else:
            return dom_classes


    def load(self, domain_id, device=None):
        domain = self.get_domains()[domain_id]
        domain.device = device

        # Base modules: SAD
        sadconfig = idt.Config()
        sadconfig.update_with_file(self.get_artifact("sad.config"))
        domain.sad_fc = idt.Frontend(sadconfig.featclass)(sadconfig)
        domain.sad_padding = domain.sad_fc.config.vad_padding
        domain.sad_interpolate = domain.sad_fc.config.vad_interpolate
        domain.sad_window = domain.sad_fc.config.window
        domain.sad_overlap = domain.sad_fc.config.overlap

        # Define domain.sad_llrframespersecond whether it’s in the sad_config or not
        if 'vad_llrframespersecond' in domain.sad_fc.config.keys():
            domain.sad_llrframespersecond = domain.sad_fc.config.vad_llrframespersecond
        else:
            domain.sad_llrframespersecond = idt.DEFAULT_FRAMES_PER_SECOND

        dnn_file = self.get_artifact("sad_dnn_nnet.pnn")
        domain.sad_nnet = NET.load_from_file(dnn_file)

        mvn_data = np.load(self.get_artifact("sad_dnn_mvn.npz"))
        domain.sad_mean, domain.sad_inv_std = mvn_data['mean'], mvn_data['inv_std']
        domain.embed_type = 'tdnn'

        # Domain-ddependent components: EmbeddingDNN, PLDA, CAL
        if 'fusion-v1' == domain_id:
            # For the fusion domain, we load both domains in memory and process them differently
            domain = self.get_domains()[domain_id]
            domain.feat_fc = {}
            domain.embed_type = {}
            domain.embed_mean = {}
            domain.embed_inv_std = {}
            domain.nnet = {}
            domain.cal_offset = {}
            domain.data_hash = {}

            for idomain_id in static_fusion_domains:
                idomain = self.get_domains()[idomain_id]

                embed_config_file = idomain.get_artifact("embed_config.py")
                embedconfig = idt.Config()
                embedconfig.update_with_file(embed_config_file)
                domain.feat_fc[idomain_id] = idt.Frontend(embedconfig.featclass)(embedconfig)

                domain.embed_type[idomain_id] = 'wav2vec'
                if 'wav2vec' not in idomain_id:

                    domain.embed_type[idomain_id] = 'tdnn'
                    embed_layer = int(open(idomain.get_artifact("embed_layer.txt")).readlines()[0].strip())
                    if os.path.exists(idomain.get_artifact("embed_nn.pnn")):
                        nnet_file = idomain.get_artifact("embed_nn.pnn")
                    else:
                        self.escape_with_error("Please run each domain independently (not fusion-v1) to define models")

                    mvn_data = np.load(idomain.get_artifact("embed_mvn.npz"))
                    domain.embed_mean[idomain_id], domain.embed_inv_std[idomain_id] = mvn_data['mean'], mvn_data['inv_std']

                    domain.nnet[idomain_id] = NET.load_from_file(nnet_file)
                    inds = np.where(np.array(domain.nnet[idomain_id].layer_index)==embed_layer-1)[0]
                    domain.nnet[idomain_id].model = torch.nn.Sequential(*list(domain.nnet[idomain_id].model.children())[:inds[0]+1])

                # Domain-specific cal offset
                domain.cal_offset[idomain_id] = 0.0
                if os.path.exists(idomain.get_artifact("cal.offset")):
                    domain.cal_offset[idomain_id] = float(open(idomain.get_artifact("cal.offset")).readlines()[0].strip())

                # Load all enrolments on load and subsample during score_plda we also load SI
                domain.data_hash[idomain_id] = -1000.

                # Dialect map
            if os.path.exists(domain.get_artifact('emotion.map')):
                domain.map_from_file = dict(np.atleast_2d(np.loadtxt(domain.get_artifact('emotion.map'), dtype=str)))

            self.update_emotion_models(domain) #, return_fusion_data_forced=True)

            # Domain-specific cal offset
            domain.cal_offset = 0.0
            if os.path.exists(domain.get_artifact("cal.offset")):
                domain.cal_offset = float(open(domain.get_artifact("cal.offset")).readlines()[0].strip())



        else:

            if domain_id not in self.loaded_domains:
                domain = self.get_domains()[domain_id]

                model_path = domain.get_artifact("model")    
                domain.model = AutoModelForAudioXVector.from_pretrained(model_path, local_files_only=True)
                domain.feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)

                if True:

                    # PNCC or wav2vec
                    embed_config_file = domain.get_artifact("embed_config.py")
                    embedconfig = idt.Config()
                    embedconfig.update_with_file(embed_config_file)
                    domain.feat_fc = idt.Frontend(embedconfig.featclass)(embedconfig)

                    domain.embed_type = 'wav2vec'
                    if 'wav2vec' not in domain_id:
                        domain.embed_type = 'tdnn'

                        # Embedding Extractor
                        embed_layer = int(open(domain.get_artifact("embed_layer.txt")).readlines()[0].strip())
                        if os.path.exists(domain.get_artifact("embed_nn.pnn")):
                            nnet_file = domain.get_artifact("embed_nn.pnn")
                        else:
                            nnet_file = domain.get_artifact("embed_nn.npz")

                            from nn import StatsPoolingTDNN, StatsPoolingTDNNBA
                            dnn_file = nnet_file
                            m = np.load(dnn_file, encoding='bytes', allow_pickle='True')
                            config = idt.embeddings.eval_config(dict(m['config']))
                            del config['mode']
                            config['hidden_activations'] = 'ReLU'
                            config['nntype'] = StatsPoolingTDNNBA
                            dnn = StatsPoolingTDNNBA(config, config['in_shape'], config['out_shape'])

                            s = dnn.model.state_dict()
                            keys = list(s.keys())
                            params = {}
                            cnt = 0
                            for y in np.arange(len(m['params'])):
                                for i, x in enumerate(m['params'][y]):
                                    if 'num_batches_tracked' in keys[cnt]:
                                        print(keys[cnt], m['param_names'][y][i], cnt, i)
                                        params[keys[cnt]] = s[keys[cnt]]
                                        cnt += 1
                                    if i < 2: # W and B from affine
                                        print(keys[cnt], m['param_names'][y][i], cnt, i)
                                        params[keys[cnt]] = torch.Tensor(x.T)
                                    elif i == 2 or i == 3:
                                        print(keys[cnt], m['param_names'][y][i], cnt, i)
                                        params[keys[cnt+2]] = torch.Tensor(x.T)
                                        #params[keys[cnt]] = torch.Tensor(x.T)
                                    elif i == 4 or i == 5:
                                        print(keys[cnt], m['param_names'][y][i], cnt, i)
                                        params[keys[cnt-2]] = torch.Tensor(x.T)
                                        #params[keys[cnt]] = torch.Tensor(x.T)
                                    else:
                                        raise Exception("No option configured for 6-element layer!")

                                    cnt += 1
                            dnn.model.load_state_dict(params)
                            dnn_file = domain.get_artifact("embed_nn.pnn")
                            dnn.save(dnn_file)
                            nnet_file = dnn_file

                        mvn_data = np.load(domain.get_artifact("embed_mvn.npz"))
                        domain.embed_mean, domain.embed_inv_std = mvn_data['mean'], mvn_data['inv_std']

                        domain.nnet = NET.load_from_file(nnet_file)
                        inds = np.where(np.array(domain.nnet.layer_index)==embed_layer-1)[0]
                        domain.nnet.model = torch.nn.Sequential(*list(domain.nnet.model.children())[:inds[0]+1])


                # Domain-specific cal offset
                domain.cal_offset = 0.0
                if os.path.exists(domain.get_artifact("cal.offset")):
                    domain.cal_offset = float(open(domain.get_artifact("cal.offset")).readlines()[0].strip())

                # Load all enrolments on load and subsample during score_plda we also load SI
                domain.data_hash = -1000.

                self.update_emotion_models(domain)

                # Dialect map
                domain.map_from_file = None
                if os.path.exists(domain.get_artifact('emotion.map')):
                    domain.map_from_file = dict(np.atleast_2d(np.loadtxt(domain.get_artifact('emotion.map'), dtype=str)))

        # Voice exclusion system
        embed_config_file = self.get_artifact(os.path.join("domains","embed-v1","embed_config.py"))
        embedconfig = idt.Config()
        embedconfig.update_with_file(embed_config_file)
        domain.spk_feat_fc = idt.Frontend(embedconfig.featclass)(embedconfig)

        # Embedding Extractor
        embed_layer = int(open(self.get_artifact(os.path.join("domains","embed-v1","embed_layer.txt"))).readlines()[0].strip())
        nnet_file = self.get_artifact(os.path.join("domains","embed-v1", "embed_nn.pnn"))
        mvn_data = np.load(self.get_artifact(os.path.join("domains","embed-v1","embed_mvn.npz")))
        domain.spk_embed_mean, domain.spk_embed_inv_std = mvn_data['mean'], mvn_data['inv_std']
        domain.spk_nnet = NET.load_from_file(nnet_file)
        inds = np.where(np.array(domain.spk_nnet.layer_index)==embed_layer-1)[0]
        domain.spk_nnet.model = torch.nn.Sequential(*list(domain.spk_nnet.model.children())[:inds[0]+1])

        # Scoring for speaker exclusion
        plda_dict = self.get_artifact(os.path.join("domains","embed-v1","sid","lda_plda.h5"))
        param_dic = idt.read_data_in_hdf5(plda_dict)
        domain.spk_lda = param_dic['IvecTransform']['LDA']
        domain.spk_mu  = param_dic['IvecTransform']['Mu']
        domain.spk_plda = idt.SPLDA.load_from_dict(param_dic['PLDA'])
        cal_dict = self.get_artifact(os.path.join("domains","embed-v1","sid","cal.h5"))
#        domain.spk_cal_fusion_models = idt.read_data_in_hdf5(cal_dict)

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
        self.update_emotion_models(domain)


    def update_emotion_models(self, domain, return_fusion_data_if_changed=False, return_fusion_data_forced=False):
        
        domain.emo_data = []
        domain.emo_map  = []
        domain.spk_data = {}

        domain_id = domain.get_id()
        enrollment_dir = self.get_enrollment_storage(domain.get_id())
        domain.user_emos = self.list_classes(domain.get_id(), included=False, enrolled=True, is_mapped=False, exclude_spks=True, exclude_emos=False)
        domain.user_spks = self.list_classes(domain.get_id(), included=False, enrolled=True, is_mapped=False, exclude_spks=False, exclude_emos=True)
        # Load the speaker models available
        for speaker_id in domain.user_spks:
            speaker_dir = os.path.join(enrollment_dir, speaker_id)
            spk_data = []
            for session_id in os.listdir(speaker_dir):
                iv_path = os.path.join(enrollment_dir, speaker_id, session_id, session_id +'.vecs.h5')
                try:
                    vecs = idt.read_data_in_hdf5(iv_path)
                    for audioid in list(vecs.keys()):
                        audioid = str(audioid)
                        spk_data.append(vecs[audioid][np.newaxis, :])
                except:
                    self.escape_with_error("Corrupt speaker audio file path [%s]. Please remove and try again" % os.path.dirname(iv_path))
            domain.spk_data[speaker_id] = np.vstack(spk_data)


        # Load the user-defined emotions
        for emotion_id in domain.user_emos:
            emotion_dir = os.path.join(enrollment_dir, emotion_id)
            for session_id in os.listdir(emotion_dir):
                iv_path = os.path.join(enrollment_dir, emotion_id, session_id, session_id +'.vecs.h5')
                try:
                    vecs = idt.read_data_in_hdf5(iv_path)
                    for audioid in list(vecs.keys()):
                        audioid = str(audioid)
                        if domain_id == 'fusion-v1':
                            domain.emo_data.append(vecs[audioid])
                        else:
                            domain.emo_data.append(vecs[audioid][np.newaxis, :])
                        domain.emo_map.append(emotion_id)
                except:
                    self.escape_with_error("Corrupt emotion audio file path [%s]. Please remove and try again" % os.path.dirname(iv_path))

        # Retrain LDA, GB, and Calibration with enrolled data if it has changed since last update
        if domain_id == 'fusion-v1':
            new_hash = np.abs(np.sum(np.array(domain.emo_data[0::2])/1000.)) + np.abs(np.sum(np.array(domain.emo_data[1::2])/1000.))
        else:
            new_hash = np.abs(np.sum(np.array(domain.emo_data)/1000.))
        if static_backend == 'PLDA':
            domain.enroll_seg2model = {}
            domain.plda = {}
            domain.T_Q1 = {}
            domain.T_f = {}
            domain.data = {}
        else:
            domain.gb_means = {}
            domain.gb_wcov = {}
        domain.lda = {}
        domain.mean = {}
        domain.classes = {}
        domain.key = {}
        domain.scoredict = {}

        if domain_id == 'fusion-v1':
            ################ START fusion DOMAIN loading
            domain.emo_subdata = {}
            domain.emo_submap = {}
            is_retraining = False

            # Check if it has changed
            for idomain_id in static_fusion_domains:
                idomain = self.get_domains()[idomain_id]

                # Divide the data
                ind = static_fusion_domains.index(idomain_id)
                domain.emo_subdata[idomain_id] = domain.emo_data[ind::len(static_fusion_domains)]
                domain.emo_submap[idomain_id] = domain.emo_map[ind::len(static_fusion_domains)]

                if static_backend == 'PLDA':
                    if len(domain.user_emos) == 0:
                        # base model
                        model_path = domain.get_artifact('models.plda.' + idomain_id + '.h5')
                    else:
                        model_path = os.path.join(enrollment_dir, 'models.plda.' + idomain_id + '.h5')
                elif static_backend == 'GB':
                    if len(domain.user_emos) == 0:
                        # base model
                        model_path = domain.get_artifact('models.gb.' + idomain_id + '.h5')
                    else:
                        model_path = os.path.join(enrollment_dir, 'models.gb.' + idomain_id + '.h5')
                else:
                    self.escape_with_error("Please change 'static_backend' to one of 'PLDA' or 'GB' instead of '{}'".format(static_backend))

                if os.path.exists(model_path):
                    domain.data_hash = idt.read_data_in_hdf5(model_path, nodes=['/hash'])[0]
                    logger.info("Loaded models from hash {}, current hash {}".format(domain.data_hash, new_hash))
                    if new_hash == domain.data_hash:
                        # Load if the hash is the same
                        if static_backend == 'PLDA':
                            domain.lda[idomain_id], domain.mean[idomain_id], domain.classes[idomain_id], plda_dict, domain.enroll_seg2model[idomain_id] = \
                                idt.read_data_in_hdf5(model_path, nodes = ['/lda', '/mu', '/classes', '/plda', '/enroll_seg2model'])
                            domain.plda[idomain_id] = idt.SPLDA.load_from_dict(plda_dict)
                            if static_precompute_enroll_stats:
                                domain.plda[idomain_id].prepare_for_scoring(1, 1)
                                domain.T_Q1[idomain_id], domain.T_f[idomain_id] = idt.read_data_in_hdf5(model_path, nodes = ['/T_Q1', '/T_f'])
                            else:
                                domain.data[idomain_id], = idt.read_data_in_hdf5(model_path, nodes = ['/data', ])
                            try:
                                keyin, scitem1, scitem2, scitem3 = \
                                    idt.read_data_in_hdf5(model_path, nodes = ['/keyin', '/scitem1', '/scitem2', '/scitem3'])
                                scores = idt.Scores(scitem1, scitem2, scitem3)
                                domain.scoredict[idomain_id] = {'system0': scores}
                                mask = np.ones(scores.score_mat.shape, dtype=int) * -1
                                mask[keyin[:, 1], keyin[:, 0]] = 1
                                domain.key[idomain_id] = idt.Key(scores.train_ids, scores.test_ids, mask)
                            except:
                                pass

                        elif static_backend == 'GB':
                            domain.lda[idomain_id], domain.mean[idomain_id], domain.classes[idomain_id], domain.gb_means[idomain_id], domain.gb_wcov[idomain_id] = \
                                idt.read_data_in_hdf5(model_path, nodes = ['/lda', '/mu', '/classes', '/gbm', '/gbw'])
                            domain.key[idomain_id], domain.scoredict[idomain_id] = None, None
                            try:
                                keyin, scitem1, scitem2, scitem3 = \
                                    idt.read_data_in_hdf5(model_path, nodes = ['/keyin', '/scitem1', '/scitem2', '/scitem3'])
                                domain.key[idomain_id] = idt.LidKey.fromiter(keyin.tolist())
                                domain.scoredict[idomain_id] = {'system0': idt.Scores(scitem1, scitem2, scitem3)}
                            except:
                                pass
                else:
                    is_retraining = True

                if new_hash != domain.data_hash or is_retraining:
                    is_retraining = True
                    domain.data_hash = new_hash

                    # Load base data
                    base_data = idt.read_data_in_hdf5(idomain.get_artifact("base_data.h5"))

                    # Combine
                    if static_emotion_data_use == 'AUGMENT':
                        # Simple combination
                        all_data = np.vstack([base_data['data'][base_data['seg2model'][:, 0], :], np.array(domain.emo_subdata[idomain_id]).reshape(len(domain.emo_subdata[idomain_id]), base_data['data'].shape[1])])
                        all_map  = np.hstack([base_data['emotions'][base_data['seg2model'][:, 1]], np.array(domain.emo_submap[idomain_id])])
                        all_sources = ['base']*base_data['data'].shape[0] + ['tgt']*len(domain.emo_submap[idomain_id])
                    elif static_emotion_data_use == 'USER':
                        base_map = base_data['emotions'][base_data['seg2model'][:, 1]]
                        valid = ~np.in1d(base_map, np.unique(domain.emo_submap[idomain_id]))  # Indices of emotions ONLY in the base data
                        all_data = np.vstack([base_data['data'][base_data['seg2model'][valid, 0], :], np.array(domain.emo_subdata[idomain_id]).reshape(len(domain.emo_subdata[idomain_id]), base_data['data'].shape[1])])
                        all_map  = np.hstack([base_data['emotions'][base_data['seg2model'][valid, 1]], np.array(domain.emo_submap[idomain_id])])
                        all_sources = ['base']*valid.sum() + ['tgt']*len(domain.emo_submap[idomain_id])
                    elif static_emotion_data_use == 'USER_ONLY':
                        all_data = np.array(domain.emo_subdata[idomain_id]).reshape(len(domain.emo_subdata[idomain_id]), base_data['data'].shape[1])
                        all_map = np.array(domain.emo_submap[idomain_id])
                        all_sources = ['tgt']*len(domain.emo_submap[idomain_id])
                    else:
                        self.escape_with_error("Please set 'static_emotion_data_use' to either 'AUGMENT' or 'USER' instead of '{}'".format(static_emotion_data_use))
                    
                    emotions, emo_map = np.unique(all_map, return_inverse=True)

                    # Dump data
                    if False:
                        seg2model = np.vstack([np.arange(len(emo_map)), emo_map]).T
                        data = all_data
                        idt.save_dict_in_hdf5('base_data.h5', {'data':data, 'seg2model':seg2model, 'emotions':emotions})
                        exit()

                    # Train LDA
                    lda, mu = self.snlda_retrain(all_data, emo_map, all_sources, 'tgt', lda_dim=emotions.shape[0]-1)

                    weights = None
                    data = idt.ivector_postprocessing(all_data, lda, mu, lennorm=True)

                    if static_backend == 'PLDA':
                        plda, obj = idt.SPLDA.train(data, emo_map, None, nb_em_it=100, weights=weights, rand_init=False)
                    elif static_backend == 'GB':
                        # Train weighted GB
                        m, w = idt.GaussianBackend.train(data, emo_map, weights=weights)
#                w = np.diag(np.diag(w))  # Comment-out for full covariance
                    else:
                        self.escape_with_error("Please change 'static_backend' to one of 'PLDA' or 'GB' instead of '{}'".format(static_backend))

                    # Subset Calibration data to something reasonable
                    keep = []
                    for iemo, cnt in enumerate(np.bincount(emo_map)):
                        inds = np.where(emo_map == iemo)[0]
                        if cnt > static_max_calibration_segments_per_emotion:
                            x = np.arange(len(inds), dtype=int)
                            np.random.seed(7)
                            np.random.shuffle(x)
                            keep += inds[x[:static_max_calibration_segments_per_emotion]].tolist()
                        else:
                            keep += inds.tolist()
                    keep = np.sort(np.array(keep, dtype=int))
                    data = data[np.array(keep, dtype=int)]
                    emo_map = emo_map[np.array(keep, dtype=int)]

                    # Train Calibration
                    p = None
                    if static_backend == 'PLDA':
                        full_enroll_seg2model = np.vstack([np.arange(len(emo_map)), emo_map]).T
                        domain.data[idomain_id], domain.enroll_seg2model[idomain_id] = idt.average_data_by_model(data.copy(), full_enroll_seg2model.copy()) # Average the embeddings from each class to enroll them

                        if static_precompute_enroll_stats:
                            segsperenrol = 1
                            plda.prepare_for_scoring(segsperenrol, 1)
                            domain.T_Q1[idomain_id], domain.T_f[idomain_id] = plda.prepare_enrollments_for_scoring(domain.data[idomain_id], segsperenrol, 1, seg2model=domain.enroll_seg2model[idomain_id])
                            scores = plda.score_with_constantN_and_prepped_enrollment(data, domain.T_Q1[idomain_id], domain.T_f[idomain_id], 1, 1).T
                        else:
                            scores = idt.plda_verifier(domain.data[idomain_id], data, plda, Tseg2model=domain.enroll_seg2model[idomain_id], tseg2model=None).T
                        scitem1, scitem2, scitem3 = emotions, np.arange(data.shape[0]).astype(str).tolist(), scores.T
                        keyin = full_enroll_seg2model
                        scores = idt.Scores(emotions, np.arange(data.shape[0]).astype(str).tolist(), scores.T)
                        scoredict = {'system0': scores}
                        mask = np.ones(scores.score_mat.shape, dtype=int) * -1
                        mask[full_enroll_seg2model[:, 1], full_enroll_seg2model[:, 0]] = 1
                        key = idt.Key(scores.train_ids, scores.test_ids, mask)
                        fusion_data = [scoredict, key]
                        #models = idt.fusion.train_calibration_and_fusion(scoredict, key, p, sid=True)

                        domain.key[idomain_id] = key
                        domain.scoredict[idomain_id] = scoredict

                        # Save output
                        domain.lda[idomain_id], domain.mean[idomain_id] = lda, mu
                        domain.classes[idomain_id], domain.plda[idomain_id] = emotions, plda
                        #domain.cal_fusion_models[idomain_id] = models

                        d = {'lda': lda, 'mu': mu, 'classes': domain.classes[idomain_id], 'plda': plda.save_to_dict(), 'hash': domain.data_hash, 'enroll_seg2model': domain.enroll_seg2model[idomain_id], 'keyin': keyin, 'scitem1': scitem1, 'scitem2': scitem2, 'scitem3': scitem3}
                        if static_precompute_enroll_stats:
                            d['T_Q1'] = domain.T_Q1[idomain_id]
                            d['T_f'] = domain.T_f[idomain_id]
                        else:
                            d['data'] = domain.data[idomain_id]

                        idt.save_dict_in_hdf5(model_path, d)
                    elif static_backend == 'GB':
                        scores = idt.GaussianBackend.llks(m, w, data, addexpt=False)

                        scitem1, scitem2, scitem3 = emotions, np.arange(data.shape[0]).astype(str).tolist(), scores.T
                        scores = idt.Scores(emotions, np.arange(data.shape[0]).astype(str).tolist(), scores.T)
                        scoredict = {'system0': scores}
                        keyin = list(zip(scores.test_ids, emotions[emo_map]))
                        key = idt.LidKey.fromiter(keyin)
                        #models = idt.fusion.train_calibration_and_fusion(scoredict, key, priors=p, sid=False)
                        domain.key[idomain_id] = key
                        domain.scoredict[idomain_id] = scoredict
                        # Save output
                        domain.lda[idomain_id], domain.mean[idomain_id] = lda, mu
                        domain.classes[idomain_id], domain.gb_means[idomain_id], domain.gb_wcov[idomain_id] = emotions, m, w
                        #domain.cal_fusion_models[idomain_id] = models

                        d = {'lda': lda, 'mu': mu, 'classes': domain.classes[idomain_id], 'gbm': m, 'gbw': w, 'hash': domain.data_hash, 'keyin': keyin, 'scitem1': scitem1, 'scitem2': scitem2, 'scitem3': scitem3}
                        idt.save_dict_in_hdf5(model_path, d)
                    else:
                        self.escape_with_error("Please change 'static_backend' to one of 'PLDA' or 'GB' instead of '{}'".format(static_backend))

            # Calibration/fusion model
            if static_backend == 'PLDA':
                if len(domain.user_emos) == 0:
                    # base model
                    model_path = domain.get_artifact('models.cal.plda.h5')
                else:
                    model_path = os.path.join(enrollment_dir, 'models.cal.plda.h5')
            elif static_backend == 'GB':
                if len(domain.user_emos) == 0:
                    # base model
                    model_path = domain.get_artifact('models.cal.gb.h5')
                else:
                    model_path = os.path.join(enrollment_dir, 'models.cal.gb.h5')
            else:
                self.escape_with_error("Please change 'static_backend' to one of 'PLDA' or 'GB' instead of '{}'".format(static_backend)) 


            if not is_retraining and os.path.exists(model_path):
                # Load the model
                domain.cal_fusion_models = idt.read_data_in_hdf5(model_path)
            else:
                # Retrain the model
                # Align the two keys
                key = domain.key[static_fusion_domains[0]].align(domain.key[static_fusion_domains[1]])
                scoredict = {}
                for idomain_id in static_fusion_domains:
                # Collect and align the two score files
                    scoredict[idomain_id] = domain.scoredict[idomain_id]['system0'].align(key)

                # Train the fusion
                domain.cal_fusion_models = idt.fusion.train_calibration_and_fusion(scoredict, key, priors=None, sid=False)
                idt.save_dict_in_hdf5(model_path, domain.cal_fusion_models)
            ################ END fusion DOMAIN loading
        else:
            ################ START single DOMAIN loading
            # Check if it has changed
#            from IPython import embed
#            embed()
            if static_backend == 'PLDA':
                if len(domain.user_emos) == 0:
                    # base model
                    model_path = domain.get_artifact('lda_plda.h5')
                else:
                    model_path = os.path.join(enrollment_dir, 'models.plda.h5')
            elif static_backend == 'GB':
                if len(domain.user_emos) == 0:
                    # base model
                    model_path = domain.get_artifact('gb.h5')
                else:
                    model_path = os.path.join(enrollment_dir, 'models.gb.h5')
            else:
                self.escape_with_error("Please change 'static_backend' to one of 'PLDA' or 'GB' instead of '{}'".format(static_backend))

            if os.path.exists(model_path):
                domain.data_hash = idt.read_data_in_hdf5(model_path, nodes=['/hash'])[0]
                logger.info("Loaded models from hash {}, current hash {}".format(domain.data_hash, new_hash))
                if new_hash == domain.data_hash:
                    # Load if the hash is the same
                    if static_backend == 'PLDA':
                        domain.lda, domain.mean, domain.classes, plda_dict, domain.cal_fusion_models, domain.enroll_seg2model = \
                            idt.read_data_in_hdf5(model_path, nodes = ['/lda', '/mu', '/classes', '/plda', '/models', '/enroll_seg2model'])
                        domain.plda = idt.SPLDA.load_from_dict(plda_dict)
                        if static_precompute_enroll_stats:
                            domain.plda.prepare_for_scoring(1, 1)
                            domain.T_Q1, domain.T_f = idt.read_data_in_hdf5(model_path, nodes = ['/T_Q1', '/T_f'])
                        else:
                            domain.data, = idt.read_data_in_hdf5(model_path, nodes = ['/data', ])
                        try:
                            keyin, scitem1, scitem2, scitem3 = \
                                idt.read_data_in_hdf5(model_path, nodes = ['/keyin', '/scitem1', '/scitem2', '/scitem3'])
                            scores = idt.Scores(scitem1, scitem2, scitem3)
                            domain.scoredict = {'system0': scores}
                            mask = np.ones(scores.score_mat.shape, dtype=int) * -1
                            mask[keyin[:, 1], keyin[:, 0]] = 1
                            domain.key = idt.Key(scores.train_ids, scores.test_ids, mask)
                        except:
                            pass


                    elif static_backend == 'GB':
                        domain.lda, domain.mean, domain.classes, domain.gb_means, domain.gb_wcov, domain.cal_fusion_models = \
                            idt.read_data_in_hdf5(model_path, nodes = ['/lda', '/mu', '/classes', '/gbm', '/gbw', '/models'])
                        domain.key, domain.scoredict = None, None
                        try:
                            keyin, scitem1, scitem2, scitem3 = \
                                idt.read_data_in_hdf5(model_path, nodes = ['/keyin', '/scitem1', '/scitem2', '/scitem3'])
                            domain.key = idt.LidKey.fromiter(keyin)
                            domain.scoredict = {'system0': idt.Scores(scitem1, scitem2, scitem3)}
                        except:
                            pass

            if new_hash != domain.data_hash or return_fusion_data_forced:
                domain.data_hash = 0 # new_hash  # DEBUG

                # Load base data
                base_data = idt.read_data_in_hdf5(domain.get_artifact("base_data.h5"))

                # Combine
                if static_emotion_data_use == 'AUGMENT':
                    # Simple combination
                    all_data = np.vstack([base_data['data'][base_data['seg2model'][:, 0], :], np.array(domain.emo_data).reshape(len(domain.emo_data), base_data['data'].shape[1])])
                    all_map  = np.hstack([base_data['emotions'][base_data['seg2model'][:, 1]], np.array(domain.emo_map)])
                    all_sources = ['base']*base_data['data'].shape[0] + ['tgt']*len(domain.emo_map)
                elif static_emotion_data_use == 'USER':
                    base_map = base_data['emotions'][base_data['seg2model'][:, 1]]
                    valid = ~np.in1d(base_map, np.unique(domain.emo_map))  # Indices of emotions ONLY in the base data
                    all_data = np.vstack([base_data['data'][base_data['seg2model'][valid, 0], :], np.array(domain.emo_data).reshape(len(domain.emo_data), base_data['data'].shape[1])])
                    all_map  = np.hstack([base_data['emotions'][base_data['seg2model'][valid, 1]], np.array(domain.emo_map)])
                    all_sources = ['base']*valid.sum() + ['tgt']*len(domain.emo_map)
                elif static_emotion_data_use == 'USER_ONLY':
                    all_data = np.array(domain.emo_data).reshape(len(domain.emo_data), base_data['data'].shape[1])
                    all_map = np.array(domain.emo_map)
                    all_sources = ['tgt']*len(domain.emo_map)
                else:
                    self.escape_with_error("Please set 'static_emotion_data_use' to either 'AUGMENT' or 'USER' instead of '{}'".format(static_emotion_data_use))
                
                emotions, emo_map = np.unique(all_map, return_inverse=True)

                # Dump data
                if False:
                    seg2model = np.vstack([np.arange(len(emo_map)), emo_map]).T
                    data = all_data
                    idt.save_dict_in_hdf5('base_data.h5', {'data':data, 'seg2model':seg2model, 'emotions':emotions})
                    exit()

                # Train LDA
                lda, mu = self.snlda_retrain(all_data, emo_map, all_sources, 'tgt', lda_dim=emotions.shape[0]-1)

                weights = None
                data = idt.ivector_postprocessing(all_data, lda, mu, lennorm=True)

                if static_backend == 'PLDA':
                    plda, obj = idt.SPLDA.train(data, emo_map, None, nb_em_it=100, weights=weights, rand_init=False)
                elif static_backend == 'GB':
                    # Train weighted GB
                    m, w = idt.GaussianBackend.train(data, emo_map, weights=weights)
#                w = np.diag(np.diag(w))  # Comment-out for full covariance
                else:
                    self.escape_with_error("Please change 'static_backend' to one of 'PLDA' or 'GB' instead of '{}'".format(static_backend))

                # Subset Calibration data to something reasonable
                keep = []
                for iemo, cnt in enumerate(np.bincount(emo_map)):
                    inds = np.where(emo_map == iemo)[0]
                    if cnt > static_max_calibration_segments_per_emotion:
                        x = np.arange(len(inds), dtype=int)
                        np.random.seed(7)
                        np.random.shuffle(x)
                        keep += inds[x[:static_max_calibration_segments_per_emotion]].tolist()
                    else:
                        keep += inds.tolist()
                keep = np.sort(np.array(keep, dtype=int))
                data = data[np.array(keep, dtype=int)]
                emo_map = emo_map[np.array(keep, dtype=int)]

                # Train Calibration
                p = None
                if static_backend == 'PLDA':
                    full_enroll_seg2model = np.vstack([np.arange(len(emo_map)), emo_map]).T
                    domain.data, domain.enroll_seg2model = idt.average_data_by_model(data.copy(), full_enroll_seg2model.copy()) # Average the embeddings from each class to enroll them

                    if static_precompute_enroll_stats:
                        segsperenrol = 1
                        plda.prepare_for_scoring(segsperenrol, 1)
                        domain.T_Q1, domain.T_f = plda.prepare_enrollments_for_scoring(domain.data, segsperenrol, 1, seg2model=domain.enroll_seg2model)
                        scores = plda.score_with_constantN_and_prepped_enrollment(data, domain.T_Q1, domain.T_f, 1, 1).T
                    else:
                        scores = idt.plda_verifier(domain.data, data, plda, Tseg2model=domain.enroll_seg2model, tseg2model=None).T
                    scitem1, scitem2, scitem3 = emotions, np.arange(data.shape[0]).astype(str).tolist(), scores.T
                    scores = idt.Scores(emotions, np.arange(data.shape[0]).astype(str).tolist(), scores.T)
                    scoredict = {'system0': scores}
                    mask = np.ones(scores.score_mat.shape, dtype=int) * -1
                    mask[full_enroll_seg2model[:, 1], full_enroll_seg2model[:, 0]] = 1
                    keyin = full_enroll_seg2model
                    key = idt.Key(scores.train_ids, scores.test_ids, mask)
                    fusion_data = [scoredict, key]
                    models = idt.fusion.train_calibration_and_fusion(scoredict, key, p, sid=True)

                    # Save output
                    domain.lda, domain.mean = lda, mu
                    domain.classes, domain.plda = emotions, plda
                    domain.cal_fusion_models = models

                    d = {'lda': lda, 'mu': mu, 'classes': domain.classes, 'plda': plda.save_to_dict(), 'models': models, 'hash': domain.data_hash, 'enroll_seg2model': domain.enroll_seg2model, 'keyin': keyin, 'scitem1': scitem1, 'scitem2': scitem2, 'scitem3': scitem3}
                    if static_precompute_enroll_stats:
                        d['T_Q1'] = domain.T_Q1
                        d['T_f'] = domain.T_f
                    else:
                        d['data'] = domain.data

                    idt.save_dict_in_hdf5(model_path, d)
                elif static_backend == 'GB':
                    scores = idt.GaussianBackend.llks(m, w, data, addexpt=False)

                    scitem1, scitem2, scitem3 = emotions, np.arange(data.shape[0]).astype(str).tolist(), scores.T
                    scores = idt.Scores(emotions, np.arange(data.shape[0]).astype(str).tolist(), scores.T)
                    scoredict = {'system0': scores}
                    keyin = list(zip(scores.test_ids, emotions[emo_map]))
                    key = idt.LidKey.fromiter(keyin)
                    fusion_data = [scoredict, key]
                    models = idt.fusion.train_calibration_and_fusion(scoredict, key, priors=p, sid=False)

                    # Save output
                    domain.lda, domain.mean = lda, mu
                    domain.classes, domain.gb_means, domain.gb_wcov = emotions, m, w
                    domain.cal_fusion_models = models

                    d = {'lda': lda, 'mu': mu, 'classes': domain.classes, 'gbm': m, 'gbw': w, 'models': models, 'hash': domain.data_hash, 'keyin': keyin, 'scitem1': scitem1, 'scitem2': scitem2, 'scitem3': scitem3}
                    idt.save_dict_in_hdf5(model_path, d)
                else:
                    self.escape_with_error("Please change 'static_backend' to one of 'PLDA' or 'GB' instead of '{}'".format(static_backend))

            ################ END single DOMAIN loading




    def update_opts(self, opts, domain):

        # Copy values
        config = idt.Config(dict(self.config))

        if opts is not None:
            # Check that all passed options are valid for this plugin
            param_check = np.in1d(list(opts.keys()), self.VALID_PARAMS)
            if np.any(~param_check):
                self.escape_with_error("Unknown parameter(s) passed [%s]. Please remove from the optional parameter list." % ','.join(np.array(list(opts.keys()))[~param_check].tolist()))

            config.update(opts)

            # File-passed options are in in text format, so we need to convert these as necessary
            config.min_speech      = float(config.min_speech)
            config.sad_threshold   = float(config.sad_threshold)
            config.sad_filter      = int(config.sad_filter)
            config.sad_interpolate = int(config.sad_interpolate)
            config.sad_merge   = float(config.sad_merge)
            config.sad_padding   = float(config.sad_padding)

            config.emod_padding   = float(config.emo_padding)
            config.speaker_threshold = float(config.speaker_threshold)

            logger.debug("Using user-defined parameter options, new config is: %s" % config)

        return config


    #def run_sad(self, audio, config, speech=None):
    def run_sad(self, domain_id, audio, config, speech=None, suppress_error=False):
        domain = self.get_domains()[domain_id]
        if speech is None:
            #suppress_error = False
            if speech is None:
                interpolate = config.sad_interpolate
                filter_length = config.sad_filter

                feat = domain.sad_fc(audio)
                feat -= domain.sad_mean
                feat *= domain.sad_inv_std
                feat = torch.tensor(feat)
                feat = feat.to(config.device)

                with torch.no_grad():
                    _, logits = domain.sad_nnet(feat[0::interpolate])
                complete = feat.shape[0]
                TARGET=-1
                llrs = logit(logits[:,TARGET].detach().cpu())
                # Interpolate if needed
                if interpolate != 1:
                    w = np.r_[np.linspace(0, 1, interpolate+1), np.linspace(1, 0, interpolate+1)[1:]][1:-1]
                    llrs = upfirdn.upfirdn(llrs, w, interpolate)[interpolate - 1: interpolate - 1 + complete]

                # filt llr
                winm = np.ones(filter_length)
                llrs = np.convolve(llrs, winm/np.sum(winm), 'same')

                # For multi-fast-v2 domain: Convert 50 frames/sec to 100 frames/sec
                if domain.sad_llrframespersecond != idt.DEFAULT_FRAMES_PER_SECOND:
                    if (idt.DEFAULT_FRAMES_PER_SECOND % domain.sad_llrframespersecond) != 0:
                        self.escape_with_error("The frame rate [%d] must be divisible by vad_llrframespersecond. Chose a different vad_llrframespersecond." % idt.DEFAULT_FRAMES_PER_SECOND)
                    interpol_factor = int(idt.DEFAULT_FRAMES_PER_SECOND / domain.sad_llrframespersecond)
                    w = np.r_[np.linspace(0, 1, interpol_factor + 1), np.linspace(1, 0, interpol_factor + 1)[1:]][1:-1]
                    llrs = upfirdn.upfirdn(llrs, w, interpol_factor)[interpol_factor - 1: interpol_factor - 1 + 2*llrs.shape[0]]

                # Smoothing
                if filter_length > 0:
                    if llrs.shape[0] < float(filter_length):
                        filter_length = llrs.shape[0]
                    llrs = np.convolve(llrs, np.ones(filter_length) / float(filter_length), 'same')


                speech = llrs.ravel() > config.sad_threshold

            else:
                self.escape_with_error("Need to implement threshold/filtering for SAD of workflows!")
        else:
            suppress_error = True

        # Check duration constraints
        prelim_duration = float(speech.sum())/idt.DEFAULT_FRAMES_PER_SECOND
        if prelim_duration < config.min_speech:
            if not suppress_error:
                aid = audio.id if audio.filename is None else audio.filename
                self.escape_with_error("Insufficient speech found for [%s]. Found %.3f which is less than the required amount of \
                                    %.3f seconds." % (aid, prelim_duration, config.min_speech))
            else:
                logger.warn("Insufficient speech found for region. Found %.3f which is less than the required amount of \
                            %.3f seconds." % (prelim_duration, config.min_speech))
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
                speech = pad_speech

        duration = float(speech.sum())/idt.DEFAULT_FRAMES_PER_SECOND

        return speech.ravel(), duration


    def get_whisper_vector(self, domain, audio, speech, config):

        embeds = self.compute_embedding(audio, speech, domain, device=config.device)
        embed = np.mean(embeds,0)[:,np.newaxis].T
        return embed


    def get_audio_vector_multi(self, domain, audio, speech, config):

        embeds = []
        for idomain_id in static_fusion_domains:
            if idomain_id == 'wav2vec-v1':
                waveobject = idt.Wave(audio.data.copy(), audio.sample_rate, audio.id)
                feat = domain.feat_fc[idomain_id](waveobject, indices=speech)  # wav2vec specific line
                mean = np.mean(feat, axis=0)
                var = np.var(feat, axis=0)
                embeds.append(np.concatenate([mean,var])[np.newaxis, :])

            else:
                waveobject = idt.Wave(audio.data.copy(), audio.sample_rate, audio.id)
                feat = domain.feat_fc[idomain_id](waveobject, indices=speech)  # tdnn specific
                feat -= domain.embed_mean[idomain_id]  # tdnn specific
                feat *= domain.embed_inv_std[idomain_id]  # tdnn specific
                with torch.no_grad():
                    logits, output = domain.nnet[idomain_id](torch.tensor(feat).unsqueeze_(0).to(config.device))
                embeds.append(logits.cpu().numpy())

        return embeds

    def get_audio_wav2vec(self, domain, audio, speech, config):

        waveobject = idt.Wave(audio.data.copy(), audio.sample_rate, audio.id)
        feat = domain.feat_fc(waveobject, indices=speech)
        mean = np.mean(feat, axis=0)
        var = np.var(feat, axis=0)
        emb = np.concatenate([mean,var])[np.newaxis, :]
        return emb


    def get_audio_vector(self, domain, audio, speech, config):

        # Extract the embedding
        waveobject = idt.Wave(audio.data.copy(), audio.sample_rate, audio.id)
        feat = domain.feat_fc(waveobject, indices=speech)
        feat -= domain.embed_mean
        feat *= domain.embed_inv_std
        with torch.no_grad():
            logits, output = domain.nnet(torch.tensor(feat).unsqueeze_(0).to(config.device))
        embed = logits.cpu().numpy()

        return embed

    def get_speaker_vector(self, domain, audio, speech, config):

        # Extract the embedding
        waveobject = idt.Wave(audio.data.copy(), audio.sample_rate, audio.id)
        feat = domain.spk_feat_fc(waveobject, indices=speech)
        feat -= domain.spk_embed_mean
        feat *= domain.spk_embed_inv_std
        with torch.no_grad():
            logits, output = domain.spk_nnet(torch.tensor(feat).unsqueeze_(0).to(config.device))
        embed = logits.cpu().numpy()

        return embed

    def get_valid_speech_via_diarization(self, domain, audio, speech, speaker_name, seg_dict, spk_dict, config):
        # Diarization
        window = 150
        overlap = 75
        overlaps_per_window = int(window/overlap)
        ss = time.time()

        # Process each speech segment
        align = idt.Alignment()
        align.add_from_indices('temp', speech, 'speech')
        align.pad(config.emo_padding)
        align.pad(-1*config.emo_padding)
        align.add_from_indices('temp', speech, 'speech')
        segs = []
        for st, en in align.get_start_end('temp', unit='seconds'):
            offset = int(st*100)
            span = int((en-st)*100)
            if span < overlap:
                continue
            if span > overlap and span < window:
                seg_starts = [offset]
                seg_ends = [offset + span]
            else:
                num_segs = int((span - (overlaps_per_window-1)*overlap) / overlap)
                noverlap  = int((span - (overlaps_per_window-1)*overlap) / num_segs)
                nwindow   = overlaps_per_window * noverlap
                seg_starts = np.arange(offset, offset + span - (overlaps_per_window*noverlap-1), noverlap)
                seg_ends   = np.arange(offset + nwindow, offset + nwindow + noverlap*num_segs, noverlap)
            segs += list(zip(seg_starts, seg_ends))
        logger.warn(f"TIMIND A: {time.time()-ss}")

        # Get the embeddings
        embeds = []
        for st,en in segs:
            # Storing in the seg_dict removes repeatedly processing the same audio segment (slow)
            if en in seg_dict.keys():
                embed = seg_dict[en]
            else:
                waveobject = idt.Wave(audio.data.copy(), audio.sample_rate)
                waveobject.trim_samples([(st/100., en/100)])
                feat = domain.spk_feat_fc(waveobject, indices=np.ones(en-st).astype(bool))
                feat -= domain.spk_embed_mean
                feat *= domain.spk_embed_inv_std
                with torch.no_grad():
                    logits, output = domain.spk_nnet(torch.tensor(feat).unsqueeze_(0).to(config.device))
                embed = logits.cpu().numpy()
                seg_dict[en] = embed
            embeds.append(embed)
        logger.warn(f"TIMIND B: {time.time()-ss}")
        embeds = np.vstack(embeds)
        embeds = idt.ivector_postprocessing(embeds, domain.spk_lda, domain.spk_mu, lennorm=True)

        if embeds.shape[0] < 2:
            return  np.zeros(speech.shape, dtype=bool)

        # Diarization
        scr_mat = idt.plda_verifier(embeds, embeds, domain.spk_plda)
        #sc = scr_mat[np.triu_indices(scr_mat.shape[0], 1)]
        #gmm = mixture.GaussianMixture(n_components=1, covariance_type='tied')
        #gmm.fit(np.atleast_2d(sc).T)
        #scale = -2*gmm.means_[0][0]/gmm.covariances_[0][0]
        #scr_mat *= scale
        maxv = np.max(scr_mat)
        data = maxv - scipy.spatial.distance.squareform(scr_mat, checks=False)
        Z = fastcluster.linkage(data, metric='cosine', method='ward', preserve_input=False)
        num_clusters = 2
        indices = scipy.cluster.hierarchy.fcluster(Z, num_clusters, criterion='maxclust')

        logger.warn(f"TIMIND C: {time.time()-ss}")
        # Map to speech
        realign_idxs = np.zeros(speech.shape[0], dtype=int)
        prevseg, prevind = None, None
        for i, segwin in enumerate(segs):
            seg = np.arange(segwin[0],segwin[1]).astype(int)
            # Assumiing segs are sequential, we can remove half of the measured overlap
            if (i == 0) or (indices[i] == prevind):
                realign_idxs[seg] = indices[i]
            else:
                # Change of index, Must find mid-point of change
                splitpoint = int(np.in1d(seg, prevseg).sum()/2)  # + seg[0]
                realign_idxs[seg[splitpoint:]] = indices[i]
            prevseg, prevind = seg, indices[i]
        realign_idxs = idt.condense_realign_idxs(realign_idxs)
        logger.warn(f"TIMIND D: {time.time()-ss}")

        # NEXT BIGGEST TIME WASTER TODO
        # Get the enrollment for this speaker
        enroll_embed = idt.ivector_postprocessing(domain.spk_data[speaker_name], domain.spk_lda, domain.spk_mu, lennorm=True)
        num_seg = len(domain.spk_data[speaker_name])
        enroll_seg2model = np.array([np.arange(num_seg), [0]*num_seg]).T

        # Now we check if each of the speakers is excludable
        # OPTION could half this processing with smart selection of 1 or 2
        test_ids = [audio.id]
        spk_align_key = idt.Key([speaker_name], test_ids, np.ones((1, 1), dtype=np.int8))
        valid = []
        spk_embeds = []
        for ind in [1,2]:
            this_speech = realign_idxs == ind
            if this_speech.sum() > 2:
                skey = this_speech.sum() + np.where(realign_idxs==ind)[0][0]
                if skey in spk_dict.keys():
                    xembed = spk_dict[skey]
                else:
                    xembed = self.get_speaker_vector(domain, audio, this_speech, config)
                    spk_dict[skey] = xembed
                tembed = idt.ivector_postprocessing(xembed, domain.spk_lda, domain.spk_mu, lennorm=True)
                scores = idt.plda_verifier(enroll_embed, tembed, domain.spk_plda, Tseg2model=enroll_seg2model, tseg2model=None).T
                logger.warn(f"SPEAKER SCORE {ind} {scores[0,0]}")
                if scores[0,0] > config.speaker_threshold:
                    valid.append(ind)
                    spk_embeds.append(xembed)

        # We have the valid speech now
        valid_speech = np.zeros(speech.shape, dtype=bool)
        for vv in valid:
            valid_speech[realign_idxs == vv] = True
        logger.warn(f"TIMIND E: {time.time()-ss}")

        if len(spk_embeds) > 0:
            if len(spk_embeds) == 1:
                avg_embed = spk_embeds[0]
            else:
                avg_embed = np.mean(spk_embeds, 0)
        else:
            avg_embed = []

        return valid_speech, seg_dict, spk_dict, avg_embed

    def map_score_names(self, scores, dialect_map, fnc=np.max):
        scores.train_ids = [dialect_map[x] for x in scores.train_ids]
        Tuniq, Tindices = np.unique(scores.train_ids, return_inverse=True)
        smat = np.zeros([len(Tuniq), len(scores.test_ids)])
        for ii, ind in enumerate(np.unique(Tindices)):
            smat[ii, :] = fnc(scores.score_mat[Tindices==ind, :], 0)
        return idt.Scores(Tuniq, scores.test_ids, smat)


    def compute_embedding(self, audio, speech, domain, device):
        # Resample audio to DFA model sample rate
        audio = audio.resample(static_sample_rate)

        # Upsample SAD frames to audio sample rate
        speech_samples = np.repeat(speech, int(static_sample_rate / 100))
        speech_samples = np.pad(speech_samples, (0, len(audio) - len(speech_samples)), constant_values=False)
        if len(speech_samples) > len(audio):
            speech_samples = speech_samples[:len(audio)]
        data = audio.data[speech_samples]

        # Chunks audio into equal-sized pieces
        chunk_len_in_samples = int(static_chunk_len_in_secs * static_sample_rate)
        chunk_iterator = AudioChunkIterator(data, chunk_len_in_samples, static_normalize_chunk)

        # Dataloader for batching of chunks
        loader = torch.utils.data.DataLoader(chunk_iterator, batch_size=static_batch_size)

        # Compute embeddings for all chunks
        embeddings = []
        with torch.inference_mode():
            for batch in loader:
                inputs = domain.feature_extractor(
                    np.atleast_2d(batch),
                    sampling_rate=domain.feature_extractor.sampling_rate,
                    padding="max_length",
                    max_length=chunk_len_in_samples,
                    truncation=True,
                    return_tensors="pt",
                )
                # Move inputs to device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                embeds = domain.model(**inputs).embeddings.cpu().numpy()
                embeddings.extend(embeds)
        return embeddings


    def run_global_scoring(self, domain_id, audio, workspace, classes=None, opts=None, return_fusion_data=False):

        domain = self.get_domains()[domain_id]
        # Device to run on
        if not hasattr(domain, 'cuda_device'):
            domain.cuda_device = self.get_cuda_device(domain_id)
        if domain.cuda_device != "-1":
            device = torch.device('cuda:{}'.format(domain.cuda_device))
        else:
            device = torch.device('cpu')

        domain.sad_nnet = domain.sad_nnet.to(device)
        domain.sad_nnet.eval()

        # speaker NNet
        domain.nnet = domain.nnet.to(device)
        domain.nnet.eval()
        domain.model = domain.model.to(device)
        domain.model.eval()

        # Update options if hey are passed in
        config = self.update_opts(opts, domain)
        config.device = device

        # Check that emotions are enrolled
        available, dialect_map, dialect_map_inv = self.list_classes(domain_id, is_mapped=static_enable_emotion_mapping, return_maps=True, exclude_spks=True)
        if classes is not None:
            # Capitalize for consistency in checking
            if not np.all(np.in1d(classes, available)):
                self.escape_with_error("Requested classes that are not available. Requested: [%s]. Available: [%s]" % (','.join(classes), ','.join(available)))
        else:
            classes = available

        if static_enable_emotion_mapping and dialect_map is not None:
            # We have to expand the mapped classes if any back to the subclasses for processing
            available_unmapped = self.list_classes(domain_id, is_mapped=False, exclude_spks=True)
            classes = dialect_map_inv[np.in1d(dialect_map_inv[:,0], classes), 1].tolist()
            classes = [x for x in classes if x in available_unmapped]

        # create the alignment key, ready for calibration
        test_ids = [audio.id]
        if domain_id == 'fusion-v1':
            align_key = idt.Key(domain.classes[static_fusion_domains[0]], test_ids, np.ones((len(domain.classes[static_fusion_domains[0]]), 1), dtype=np.int8))
        else:
            align_key = idt.Key(domain.classes, test_ids, np.ones((len(domain.classes), 1), dtype=np.int8))

        # Get the whisper embeddinga
        speech, duration = self.run_sad(domain_id, audio, config, suppress_error=True)
        embed = self.get_whisper_vector(domain, audio, speech, config)

        embeds = idt.ivector_postprocessing(embed, domain.lda, domain.mean, lennorm=True)

        # Process then score the embedding
        scores = idt.GaussianBackend.llks(domain.gb_means, domain.gb_wcov, embeds, addexpt=False)
        scores = idt.Scores(domain.classes, test_ids, scores.T)
        # Calibrate the score and offset the threshold
        scores = idt.fusion.apply_calibration_and_fusion({'system0': scores}, align_key, domain.cal_fusion_models, sid=False)
        scores.score_mat = idt.loglh2detection_llr(scores.score_mat.T).T

        # Domain-specific cal offset
        scores.score_mat -= domain.cal_offset

        # Filter to the requested subset of classes and put in return format
        scores = scores.filter(classes, scores.test_ids)

        # Process dialect mapping if available and enabled
        if static_enable_emotion_mapping and dialect_map is not None:
            scores = self.map_score_names(scores, dialect_map)

        # Class-specific cal offset
        for ind, key in enumerate(scores.train_ids):
            if key in config.manual_calibration:
                scores.score_mat[ind,:] += config.manual_calibration[key]
                logger.warn(f"Added offset of {config.manual_calibration[key]} to class {key}")

        final_scores = dict([(g,scores.score_mat[i,0]) for i,g in enumerate(scores.train_ids)])
        return final_scores

    ### RegionScorerStreaming ###
    def run_region_scoring_streaming(self, domain_id, audio_stream, workspace, output_queue, classes=None, opts=None, return_fusion_data=False):
        """
        Main scoring method
        """
        logger.warn(f"OPTS: {opts}")
        domain = self.get_domains()[domain_id]
#        self.update_emotion_models(domain)

        # Device to run on
        if not hasattr(domain, 'cuda_device'):
            domain.cuda_device = self.get_cuda_device(domain_id)
        if domain.cuda_device != "-1":
            device = torch.device('cuda:{}'.format(domain.cuda_device))
        else:
            device = torch.device('cpu')

        domain.sad_nnet = domain.sad_nnet.to(device)
        domain.sad_nnet.eval()
        domain.spk_nnet = domain.spk_nnet.to(device)

        if domain_id == 'fusion-v1':
            for idomain_id in static_fusion_domains:
                if idomain_id == 'wav2vec-v1':
                    domain.feat_fc[idomain_id].cuda = domain.cuda_device != "-1"
                    domain.feat_fc[idomain_id].model = domain.feat_fc[idomain_id].model.to(device)
                else:
                    domain.nnet[idomain_id] = domain.nnet[idomain_id].to(device)
                    domain.nnet[idomain_id].eval()
        else:
            # speaker NNet
            domain.nnet = domain.nnet.to(device)
            domain.nnet.eval()
            if domain.embed_type == 'wav2vec':
                domain.feat_fc.cuda = domain.cuda_device != "-1"
                domain.feat_fc.model = domain.feat_fc.model.to(device)
            else:
                domain.model = domain.model.to(device)
                domain.model.eval()

        # audio.make_mono()
        # Update options if hey are passed in
        config = self.update_opts(opts, domain)
        config.device = device

        if opts is not None and 'speaker_id' in opts:
            classes = [opts['speaker_id']]
        #classes = ['user1111']

        # Have to deal with emo and spk classes
        if classes is not None:
            speaker_classes = [xx for xx in classes if xx[:4] == 'user']
            if len(speaker_classes) == 0:
                self.escape_with_error("This plugin must be passed a user id 'user*' to isolate their voice. Please enrol a users voice with a model name 'user...' and pass this as an option for the stream processing.")
            if len(speaker_classes) > 1:
                self.escape_with_error("This plugin must be passed a SINGLE user id 'user*' to isolate their voice. Please pass a single user name prefixed with 'user' and ensure they are first enrolled.")
            else:
                speaker_name = speaker_classes[0]
                logger.warn(f"Processing stream to analyze the voice of {speaker_name}.")
        else:
            self.escape_with_error("This plugin must be passed a user id 'user*' to isolate their voice. Please enrol a users voice with a model name 'user...' and pass this as an option for the stream processing.")

        if classes is not None:
            emo_classes = [xx for xx in classes if xx[:4] != 'user']
            if len(emo_classes) == 0:
                classes = None
            else:
                classes = emo_classes

        # Check that emotions are enrolled
        available, dialect_map, dialect_map_inv = self.list_classes(domain_id, is_mapped=static_enable_emotion_mapping, return_maps=True, exclude_spks=True)
        if classes is not None:
            # Capitalize for consistency in checking
            if not np.all(np.in1d(classes, available)):
                self.escape_with_error("Requested classes that are not available. Requested: [%s]. Available: [%s]" % (','.join(classes), ','.join(available)))
        else:
            classes = available

        if static_enable_emotion_mapping and dialect_map is not None:
            # We have to expand the mapped classes if any back to the subclasses for processing
            available_unmapped = self.list_classes(domain_id, is_mapped=False, exclude_spks=True)
            classes = dialect_map_inv[np.in1d(dialect_map_inv[:,0], classes), 1].tolist()
            classes = [x for x in classes if x in available_unmapped]

        # create the alignment key, ready for calibration
        test_ids = [audio_stream.id]
        if domain_id == 'fusion-v1':
            align_key = idt.Key(domain.classes[static_fusion_domains[0]], test_ids, np.ones((len(domain.classes[static_fusion_domains[0]]), 1), dtype=np.int8))
        else:
            align_key = idt.Key(domain.classes, test_ids, np.ones((len(domain.classes), 1), dtype=np.int8))

        last_results = []

        # NOW SAD
        counter = 0
        first_tss = -1
        last_tss = -1
        data = []
        last_duration = 0.0
        final_result = {}
        partial_results = {}
        total_sad = []
        seg_dict = {}
        spk_dict = {}
        current_sad_frame = 0
        time_since_last_speech = 0.0
        while audio_stream.is_alive:
            for chunk, tss in audio_stream:
                logger.info("Loop counter = {}, stream sample rate: {}".format(counter, audio_stream.sample_rate))
                SS = time.time()
                counter += 1
                # do something with is audio chunk….
                logger.info("Got tss: {}".format(tss))
                logger.info("Got chunk: {}".format(chunk))
                if first_tss < 0:
                    first_tss = tss[0]
                last_tss = tss[1]

                # doesn't exist
                # chunk.unnormalize()
                pcm = np.asarray(chunk * (2 ** 16 / 2), dtype=np.int16)
                data.append(pcm)
                audio = idt.Wave(np.hstack(data), audio_stream.sample_rate)
                logger.warn(f"TIMING A: {time.time()-SS}")
                if audio.duration >= last_duration + config.output_per_seconds:
                    speech, duration = self.run_sad(domain_id, audio, config, suppress_error=True)
                    if len(speech)>int(100*MAX_BUFFER_DURATION):
                        logger.warn(f"--> SADLIMIT {duration} exceeded limit of 45 seconds. Keeping last 45 seconds.")
                        speech = speech[-int(100*MAX_BUFFER_DURATION):]
                    logger.warn(f"--> SAD {duration} from {first_tss} to {last_tss} : {len(speech)}")
                    last_duration = audio.duration
                    logger.warn(f"TIMING B: {time.time()-SS}")

                    # Speech quantity check
                    enough_speech = False
                    if duration > config.min_detection_speech:
                        # Exclude HMI speech if any
                        enough_speech = True
                        if True: #domain.has_hmi:
                            speech, seg_dict, spk_dict, avg_embed = self.get_valid_speech_via_diarization(domain, audio, speech, speaker_name, seg_dict, spk_dict, config)
                            logger.warn(f"SPEECH COUNT {speech.sum()/100.}")
                            if speech.sum()/100. < config.min_detection_speech:
                                enough_speech = False  # TODO: Check that the increase in speech is meaningful

                        logger.warn(f"TIMING C: {time.time()-SS}")

                    # Keep track of time since last speech detected from user
                    if len(speech) == 0 or speech.sum() == 0:
                        time_since_last_speech = audio.duration
                    else:
                        valid_spinds = np.where(speech)
                        logger.warn(f"SPEECH INDS {np.where(speech)[0]}")
                        time_since_last_speech = min(MAX_BUFFER_DURATION, audio.duration) - np.where(speech)[0][-1]/100.
            
                    # Determined there is enough meaningful speech for a new full process of audio
                    if enough_speech:
                        user_speech_duration = speech.sum()/100.
                        # Get the embedding
                        if domain_id == 'fusion-v1':
                            embeds = self.get_audio_vector_multi(domain, audio, speech, config)
                            scoredict = {}
                            for idx, idomain_id in enumerate(static_fusion_domains):
                                embed = idt.ivector_postprocessing(embeds[idx], domain.lda[idomain_id], domain.mean[idomain_id], lennorm=True)
                                if static_backend == 'PLDA':
                                    if static_precompute_enroll_stats:
                                        scores = domain.plda.score_with_constantN_and_prepped_enrollment(embed, domain.T_Q1[idomain_id], domain.T_f[idomain_id], 1, 1).T
                                    else:
                                        scores = idt.plda_verifier(domain.data[idomain_id], embed, domain.plda[idomain_id], Tseg2model=domain.enroll_seg2model[idomain_id], tseg2model=None).T
                                    scores = idt.Scores(domain.classes[idomain_id], test_ids, scores.T)
                                elif static_backend == 'GB':
                                    scores = idt.GaussianBackend.llks(domain.gb_means[idomain_id], domain.gb_wcov[idomain_id], embed, addexpt=False)
                                    scores = idt.Scores(domain.classes[idomain_id], test_ids, scores.T)

                                scoredict[idomain_id] = scores


                            if static_backend == 'PLDA':
                                # PLDA
                                scores = idt.fusion.apply_calibration_and_fusion(scoredict, align_key, domain.cal_fusion_models, sid=True)
                            elif static_backend == 'GB':
                                # GB
                                scores = idt.fusion.apply_calibration_and_fusion(scoredict, align_key, domain.cal_fusion_models, sid=False)
                                scores.score_mat = idt.loglh2detection_llr(scores.score_mat.T).T

                        else:
                            if domain.embed_type == 'wav2vec':
                                embeds = self.get_audio_wav2vec(domain, audio, speech, config)
                            else:
                                #embeds = self.get_audio_vector(domain, audio, speech, config)
                                #embeds = avg_embed
                                embeds = self.get_whisper_vector(domain, audio, speech, config)

                            embeds = idt.ivector_postprocessing(embeds, domain.lda, domain.mean, lennorm=True)

                            # Process then score the embedding
                            if static_backend == 'PLDA':
                                if static_precompute_enroll_stats:
                                    scores = domain.plda.score_with_constantN_and_prepped_enrollment(embeds, domain.T_Q1, domain.T_f, 1, 1).T
                                else:
                                    scores = idt.plda_verifier(domain.data, embeds, domain.plda, Tseg2model=domain.enroll_seg2model, tseg2model=None).T
                                scores = idt.Scores(domain.classes, test_ids, scores.T)
                                if return_fusion_data:
                                    return scores, align_key, classes, dialect_map
                                scores = idt.fusion.apply_calibration_and_fusion({'system0': scores}, align_key, domain.cal_fusion_models, sid=True)
                            elif static_backend == 'GB':
                                scores = idt.GaussianBackend.llks(domain.gb_means, domain.gb_wcov, embeds, addexpt=False)
                                scores = idt.Scores(domain.classes, test_ids, scores.T)
                                if return_fusion_data:
                                    return scores, align_key, classes, dialect_map
                                # Calibrate the score and offset the threshold
                                scores = idt.fusion.apply_calibration_and_fusion({'system0': scores}, align_key, domain.cal_fusion_models, sid=False)
                                scores.score_mat = idt.loglh2detection_llr(scores.score_mat.T).T
                            logger.warn(f"TIMING E: {time.time()-SS}")

                        # Domain-specific cal offset
                        scores.score_mat -= domain.cal_offset

                        # Filter to the requested subset of classes and put in return format
                        scores = scores.filter(classes, scores.test_ids)

                        # Process dialect mapping if available and enabled
                        if static_enable_emotion_mapping and dialect_map is not None:
                            scores = self.map_score_names(scores, dialect_map)

                        # Class-specific cal offset
                        for ind, key in enumerate(scores.train_ids):
                            if key in config.manual_calibration:
                                scores.score_mat[ind,:] += config.manual_calibration[key]
                                logger.warn(f"Added offset of {config.manual_calibration[key]} to class {key}")
                        
                        # Convert speech indices to timestamped regions
                        #################### STATS
                        segments = []
                        for ii, emo in enumerate(scores.train_ids):
                            segments.append((first_tss, last_tss, emo, scores.score_mat[ii, 0]))
                        segments.append((first_tss, last_tss, 'user_speech', user_speech_duration))
                        segments.append((first_tss, last_tss, 'time_since_speech', time_since_last_speech))
                        segments.append((first_tss, last_tss, 'sufficient_speech', int(user_speech_duration >= config.ok_detection_speech)))

                    else:
                        # Not enough speech
                        segments = []
                        for ii, emo in enumerate(available):
                            segments.append((first_tss, last_tss, emo, 0.0))
                        segments.append((first_tss, last_tss, 'user_speech', 0.0))
                        segments.append((first_tss, last_tss, 'time_since_speech', last_tss))
                        segments.append((first_tss, last_tss, 'sufficient_speech', 0))

                    # Partial result
                    frames = [first_tss, last_tss]
                    partial_results = {'partial_result': segments}
                    final_result = {'final_result': segments}
                    if output_queue != None and audio_stream.is_alive:
                        #output_queue.put(( {'speech': segments}, {}, frames))
                        output_queue.put(( partial_results, {}, frames))
                    logger.warn(f"TIMING F: {time.time()-SS}")

        if output_queue != None:
            frames = [first_tss, last_tss]
            output_queue.put(( final_result, {}, frames))


    ### ClassEnroller/ClassModifier ###
    def add_class_audio(self, domain_id, audio, class_id, enrollspace, opts=None):

        domain = self.get_domains()[domain_id]
        if not hasattr(domain, 'cuda_device'):
            domain.cuda_device = self.get_cuda_device(domain_id)
        if domain.cuda_device != "-1":
            device = torch.device('cuda:{}'.format(domain.cuda_device))
        else:
            device = torch.device('cpu')

        domain.sad_nnet = domain.sad_nnet.to(device)
        domain.sad_nnet.eval()
        domain.spk_nnet = domain.spk_nnet.to(device)
        domain = self.get_domains()[domain_id]
        if domain_id == 'fusion-v1':
            for idomain_id in static_fusion_domains:
                if idomain_id == 'wav2vec-v1':
                    domain.feat_fc[idomain_id].cuda = domain.cuda_device != "-1"
                    domain.feat_fc[idomain_id].model = domain.feat_fc[idomain_id].model.to(device)
                else:
                    domain.nnet[idomain_id] = domain.nnet[idomain_id].to(device)
                    domain.nnet[idomain_id].eval()
        else:
            # speaker NNet
            domain.nnet = domain.nnet.to(device)
            domain.nnet.eval()
            if domain.embed_type == 'wav2vec':
                domain.feat_fc.cuda = domain.cuda_device != "-1"
                domain.feat_fc.model = domain.feat_fc.model.to(device)
            else:
                domain.model = domain.model.to(device)
                domain.model.eval()

        config = self.update_opts(opts, domain)
        config.device = device
        enrollment_dir = self.get_enrollment_storage(domain.get_id())

        # Check that class_id is OK for enrollment without a class
        if False: #os.path.exists(domain.get_artifact('emotion.map')):
            if not hasattr(domain, 'map_from_file'):
                map_from_file = dict(np.atleast_2d(np.loadtxt(domain.get_artifact('emotion.map'), dtype=str)))
            else:
                map_from_file = dict(domain.map_from_file)
            if class_id in set(map_from_file.values()):
                self.escape_with_error("Cannot enroll emotion with name '{}' as it exists in the domain file 'emotion.map' as a value to be mapped to. Either remove this mapping entry from the emotion.map or enroll under a different name. If the second option is taken, ensure that the emotion does not already exist in the plugin as discrimination will be hindered.".format(class_id))

        if opts is not None and 'region' in opts:
            audio.trim_samples(opts['region'])
            
        duration = audio.duration

        audio_dir = os.path.join(enrollspace, "staging-%s" % domain_id, audio.id)
        utils.mkdirs(audio_dir)

        # OPTIONAL: Add chunkng of enrollment data for calibration adaptation
        embed_filename = os.path.join(enrollment_dir, class_id, audio.id, audio.id + '.vecs.h5')
        out_embed_filename = os.path.join(audio_dir, audio.id + '.vecs.h5')
        if not os.path.exists(embed_filename):
            # Generate vecs for 10, 20, 30, 60, 120, 240 durations AND the duration of each in embed_data
#            speech, duration = self.run_sad(audio, config)
            speech, duration = self.run_sad(domain_id, audio, config, suppress_error=True)
            if class_id[:4] == 'user':
                embeds = self.get_speaker_vector(domain, audio, speech, config)
            else:
                if domain_id == 'fusion-v1':
                    embeds = self.get_audio_vector_multi(domain, audio, speech, config)
                else:
                    if domain.embed_type == 'wav2vec':
                        embeds = self.get_audio_wav2vec(domain, audio, speech, config)
                    else:
                        embeds = self.get_whisper_vector(domain, audio, speech, config)

                # Save vecs and meta
            embed_dict = dict([(audio.id + '_' + str(i), v) for i, v in enumerate(embeds)])
            idt.save_dict_in_hdf5(out_embed_filename, embed_dict)
        else:
            shutil.copy(embed_filename, out_embed_filename)

        # Return location for the purpose of audiovectors (not needed for audio enrollment)
        return audio_dir, duration


    def remove_class_audio(self, domain_id, audio, class_id, enrollspace):
        if 'fusion-v1' == domain_id:
            for idomain_id in static_fusion_domains:
                output = self.domain_remove_class_audio(idomain_id, audio, class_id, enrollspace)
        else:
            self.domain_remove_class_audio(domain_id, audio, class_id, enrollspace)


    def domain_remove_class_audio(self, domain_id, audio, class_id, enrollspace):
        removal_dir = os.path.join(enrollspace, "removals-%s" % domain_id)

        utils.mkdirs(removal_dir)

        with open(os.path.join(removal_dir, audio.id), "w") as f:
            f.write("please")


    def obsolete_finalize_class(self, domain_id, class_id, enrollspace):

        if 'fusion-v1' == domain_id:
            for idomain_id in static_fusion_domains:
                self.domain_finalize_class(idomain_id, class_id, enrollspace)
        else:
            self.domain_finalize_class(domain_id, class_id, enrollspace)


    def finalize_class(self, domain_id, class_id, enrollspace):

        final_enrollment_dir = self.get_enrollment_storage(domain_id, class_id)

        removal_dir = os.path.join(enrollspace, "removals-%s" % domain_id)
        if os.path.isdir(removal_dir):
            for file in os.listdir(removal_dir):
                target = os.path.join(final_enrollment_dir, file)
                if os.path.isdir(target):
                    shutil.rmtree(target)
            shutil.rmtree(removal_dir)
        
        for file in glob.glob(os.path.join(enrollspace, "staging-%s" % domain_id, "*")):
            if len(os.listdir(file))>0:
                dest = os.path.join(final_enrollment_dir, os.path.basename(file))
                if os.path.exists(dest):
                    shutil.rmtree(dest)
                logger.warn("Finalizing %s" % (file))
                shutil.move(file, dest)
            else:
                logger.warn("Audio id [%s] for class_id [%s] failed to enroll" % (file, class_id))


    def obsolete_remove_class(self, domain_id, class_id, workspace):
        if 'fusion-v1' == domain_id:
            for idomain_id in static_fusion_domains:
                output = self.domain_remove_class(idomain_id, class_id, workspace)
        else:
            self.domain_remove_class(domain_id, class_id, workspace)

    def remove_class(self, domain_id, class_id, workspace):
        class_dir = self.get_enrollment_storage(domain_id, class_id)
        unpopulated = len(os.listdir(class_dir)) == 0  # Indicates this is a base class with no user data to unenroll
        shutil.rmtree(class_dir)
        if unpopulated:
            self.escape_with_error("Class '{}' is a built-in emotion and there is no user-enrolled data for this class to remove.".format(class_id))


    def snlda_retrain(self, full_data, full_spk_ids, full_source_idxs, source_id, lda_dim=150):
        # LDA retraining
        #lda = idt.estimate_lda(full_data, full_spk_ids, lda_dim, whiten=True) #, full_source_idxs, whiten=True)
        lda = idt.estimate_lda_robust(full_data, full_spk_ids, lda_dim, whiten=True)
        # lda = idt.estimate_snlda(full_data, full_spk_ids, lda_dim, full_source_idxs, whiten=True)
        if lda.dtype.type == np.complex128:
            lda = idt.estimate_lda_robust(full_data, full_spk_ids, lda_dim, whiten=True)
        #uniqsrc, idxssrc = np.unique(full_source_idxs, return_inverse=True)
        #if np.sum(uniqsrc==source_id) > 0:
        #    indomidx = np.where(uniqsrc==source_id)[0][0]
        #else:
        #    indomidx = np.arange(len(uniqsrc), dtype=int)
        #mu = np.mean(full_data[idxssrc==indomidx], axis=0).dot(lda)
        mu = np.mean(full_data, axis=0).dot(lda)

        return lda, mu

    def get_region_scoring_opts(self):
        return self.get_global_scoring_opts()

    def get_global_scoring_opts(self):
        """
        These options are used in the OLIVE GUI and may be configured on the commandline by passing a file to --options
        """
        trait_options = [
            TraitOption('sad_threshold', "Threshold to determine speech frames", "Higher value results in less speech from processing (default 1.0)", TraitType.CHOICE_TRAIT, "", self.config.sad_threshold),
            TraitOption('min_speech', "Amount of speech needed to process audio", "Higher value results in less scores being output, but higher confidence (default 2.0)", TraitType.CHOICE_TRAIT, "", self.config.min_speech),
        ]
        return trait_options


# This line is very important! Every plugin should have one
plugin = CustomPlugin()
