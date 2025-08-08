import glob
import os
import re
import numpy as np
import importlib
import idento3 as idt
import idento3.engines as idte
from idento3.features import FeatProcess
from olive.plugins import logger, Plugin, GlobalScorer, ClassModifier, utils, TraitOption, TraitType, shutil
import torch
from nn import NET
from utils import mvn
from eval_buffer import FileLoad_EVAL

try:
    x = idt.DEFAULT_FRAMES_PER_SECOND
except AttributeError:
    idt.DEFAULT_FRAMES_PER_SECOND = 100

default_config = idt.Config(dict(
##################################################
# Configurable [global scoring] parameters
# WARNING - these should be changed in plugin_config.py, not here

# DETECTION OPTIONS
min_speech = 2.0,

# SAD
sad_threshold   = 1.0,
sad_filter      = 31,
sad_interpolate = 4,
sad_merge       = 0.3,
sad_padding     = 0.1,

# Define the languages enabled for each domain in ./domains/$domain/domain_config.txt

##################################################
))

# Dialect mapping to base language if available
static_enable_dialect_to_language_mapping = True
static_precompute_enroll_stats = True  # Used with PLDA for faster scoring
static_max_calibration_segments_per_lang = 1000  # Subset limit when enrolling/calibrating. Full set is used for backend model training.
static_backend = 'PLDA'  # 'PLDA' or 'GB'
# For instances of when a user enrolls data for which a pre-enrolled language exists. USER: Ignore pre-enrolled data; AUGMENT: Combine the data sources.
static_language_data_use = 'USER'  # 'USER' or 'AUGMENT'
static_audio_length_before_chunking = 300  # seconds

# match the beginning of the string or a space, followed by a non-space
def cap_sentence(s):
    return ' '.join([x.capitalize() for x in s.split(' ' )])

class CustomPlugin(Plugin, GlobalScorer, ClassModifier):


    def __init__(self):
        self.task  = "LID"
        self.label = "LID SmOlive Embeddings with PLDA (Commercial)" 
        self.description = "Language embeddings framework with PLDA"
        self.vendor = "SRI"
        self.version = '1.0.0'
        self.minimum_runtime_version = '5.7.1'
        self.minimum_olive_version = '5.7.1'
        self.group = "Language"
        self.create_date = "2025-03-28"
        self.revision_date = "2025-03-28"
        self.loaded_domains = []
        self.config          = default_config
        loader               = importlib.machinery.SourceFileLoader('plugin_config', os.path.join(os.path.dirname(os.path.realpath(__file__)), 'plugin_config.py'))
        spec                 = importlib.util.spec_from_loader(loader.name, loader)
        mod                  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.config.update(mod.user_config)
        self.VALID_PARAMS    = ['region'] + list(self.config.keys()) # For checking user inputs and flagging unknown paramters. Region is passed with 5-column enrollment


    ### Method implementations required of a Plugin ###
    def list_classes(self, domain_id, enrolled=True, included=True, is_mapped=True, return_maps=False):

        if not enrolled and not included:
            self.escape_with_error("At least one of enrolled or included must be True")

        # User enrollments
        user_langs = []
        if enrolled:
            enrollment_dir = self.get_enrollment_storage(domain_id)
            user_langs = [x for x in os.listdir(enrollment_dir) if x[:6] != 'models']

        # Pre-enrolled languages
        domain_langs = []
        domain = self.get_domains()[domain_id]
        if included:
            domain_config_file = domain.get_artifact('domain_config.txt')
            enabled_pre_enrolled_languages = [x.strip() for x in open(domain_config_file) if '#' not in x]
            cal_domain_langs = idt.read_data_in_hdf5(domain.get_artifact("base_data_f32.h5"), nodes=['/languages'])[0].tolist()
            # Check that the user has only set valid languages:
            if not np.all(np.in1d(enabled_pre_enrolled_languages, cal_domain_langs)):
                self.escape_with_error("Config contains languages in file %s that are not available. Requested: [%s]. Available: [%s]" % (domain_config_file, ','.join(enabled_pre_enrolled_languages), ','.join(cal_domain_langs)))
            domain_langs = [x for x in cal_domain_langs if x in enabled_pre_enrolled_languages]

        dom_classes = np.unique(user_langs + domain_langs).tolist()

        # Handle mapping if available
        dialect_map, dialect_map_inv = None, None
        if static_enable_dialect_to_language_mapping and is_mapped and os.path.exists(domain.get_artifact('dialect_language.map')):
            dialect_map = dict(zip(dom_classes, dom_classes))
            # Ensure user-enrolled languages are not mapped as we assume they are by definition important as is
            if not hasattr(domain, 'map_from_file'):
                map_from_file = dict(np.atleast_2d(np.loadtxt(domain.get_artifact('dialect_language.map'), delimiter='\t', dtype=str)))
            else:
                map_from_file = dict(domain.map_from_file)
            for lang in user_langs:
                if lang in map_from_file.keys():
                    del map_from_file[lang]
                if lang in set(map_from_file.values()):
                    mapped = ','.join([k for k,n in map_from_file.items() if n == lang and k in domain_langs])
                    if len(mapped) > 0:
                        self.escape_with_error("There is a clash between the enrolled class '{}' and classes being mapped to it in the file 'dialect_language.map' in the domain folder which currently maps '{}' to '{}'. To fix this, (1) enroll '{}' under a different name, (2) remove the dialect-to-language mapping(s) for '{}' in dialect_language.map, or (3) disable the dialects '{}' in the domain.config of the domain folder.".format(lang, mapped, lang, lang, lang, mapped))
            dialect_map.update(map_from_file)
            dialect_map_inv = np.array(list(zip(dialect_map.values(), dialect_map.keys())))
            # Map the languages to the output labels
            dom_classes = np.unique([dialect_map[x] for x in dom_classes]).tolist()

        if return_maps:
            return dom_classes, dialect_map, dialect_map_inv
        else:
            return dom_classes


    def load(self, domain_id, device=None):

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

        # Domain-dependent components: EmbeddingDNN, PLDA, CAL
        if domain_id not in self.loaded_domains:
            domain = self.get_domains()[domain_id]

            # Senone BN feature extractor
            bn_config_path    = domain.get_artifact("embed_config.py")
            bn_config       = idt.Config(dict(feature_name='bottleneck'))
            bn_config.update_with_file(bn_config_path)
            domain.bn_layer = bn_config['layer']
            domain.bn_linear_out = bn_config['linearout']
            domain.bn_ppfunc = bn_config['featprocess_func']

            # BN extractor
            domain.feat_config = idt.Config()
            domain.feat_config.update_with_file(domain.get_artifact(os.path.join("bn_config.py")))

            # initialization
            bn_mvn = np.load(domain.get_artifact(os.path.join("bn_mvn.npz")))
            domain.bn_mean = bn_mvn['mean']
            domain.bn_inv_std = bn_mvn['inv_std']

            try:
                domain.bn_extractor = NET.load_from_file(domain.get_artifact(os.path.join("bn_nnet_int8.npz")))
                domain.embed_nnet    = NET.load_from_file(domain.get_artifact("embed_nn_int8.pnn"))
                domain.quan = True
            except:
                domain.bn_extractor = NET.load_from_file(domain.get_artifact(os.path.join("bn_nnet_f32.npz")))
                domain.embed_nnet    = NET.load_from_file(domain.get_artifact("embed_nn_f32.pnn"))
                domain.quan = False

            inds = np.where(np.array(domain.bn_extractor.layer_index)==domain.bn_layer-1)[0]
            domain.bn_extractor.model = torch.nn.Sequential(*list(domain.bn_extractor.model.children())[:inds[0]+1])
            domain.bn_extractor.eval()

            domain.evalfeatiter = FileLoad_EVAL(None, domain.feat_config, mean=domain.bn_mean, inv_std=domain.bn_inv_std, target_dim=domain.bn_extractor.in_shape[-1])

            # Embedding Extractor 
            embed_mvn     = np.load(domain.get_artifact("embed_mvn.npz"))
            domain.embed_mu      = embed_mvn['mean']
            domain.embed_inv_std = embed_mvn['inv_std']
            embed_layer   = np.loadtxt(domain.get_artifact("embed_layer.txt"), dtype=int).ravel()[0]

            # Trim to embeddings layer
            inds = np.where(np.array(domain.embed_nnet.layer_index)==embed_layer-1)[0]
            domain.embed_nnet.model = torch.nn.Sequential(*list(domain.embed_nnet.model.children())[:inds[0]+1])
            domain.embed_nnet.eval()

            # Domain-specific cal offset
            domain.cal_offset = 0.0
            if os.path.exists(domain.get_artifact("cal.offset")):
                domain.cal_offset = float(open(domain.get_artifact("cal.offset")).readlines()[0].strip())

            # Load all enrolments on load and subsample during score_plda we also load SI
            domain.data_hash = -1000.
            self.update_language_models(domain)

            # Dialect map
            domain.map_from_file = None
            if os.path.exists(domain.get_artifact('dialect_language.map')):
                domain.map_from_file = dict(np.atleast_2d(np.loadtxt(domain.get_artifact('dialect_language.map'), delimiter='\t', dtype=str)))

            self.loaded_domains.append(domain_id)

            logger.info("Loading of plugin '%s' domain '%s' complete." % (self.label, domain_id))


    def update_classes(self, domain_id):
        # Function that is called by the server after finalizing all enrollments
        domain = self.get_domains()[domain_id]
        if domain_id not in self.loaded_domains:
            self.load(domain_id)        
        self.update_language_models(domain)


    def update_language_models(self, domain):
        
        domain.lang_data = []
        domain.lang_map  = []

        enrollment_dir = self.get_enrollment_storage(domain.get_id())
        domain.user_langs = self.list_classes(domain.get_id(), included=False, enrolled=True, is_mapped=False)

        # Load the user-defined languages
        for language_id in domain.user_langs:
            language_dir = os.path.join(enrollment_dir, language_id)
            for session_id in os.listdir(language_dir):
                iv_path = os.path.join(enrollment_dir, language_id, session_id, session_id +'.vecs.h5')
                try:
                    vecs = idt.read_data_in_hdf5(iv_path)
                    for audioid in list(vecs.keys()):
                        audioid = str(audioid)
                        domain.lang_data.append(vecs[audioid][np.newaxis, :])
                        domain.lang_map.append(language_id)
                except:
                    self.escape_with_error("Corrupt language audio file path [%s]. Please remove and try again" % os.path.dirname(iv_path))

        # Retrain LDA, GB, and Calibration with enrolled data if it has changed since last update
        new_hash = np.abs(np.sum(np.array(domain.lang_data)/1000.))

        # Check if it has changed
        if static_backend == 'PLDA':
            if len(domain.user_langs) == 0 and domain.quan:
                # base model
                model_path = domain.get_artifact('lda_plda_int8.h5')
            elif len(domain.user_langs) == 0 and not domain.quan:
                model_path = domain.get_artifact('lda_plda_f32.h5')
            else:
                model_path = os.path.join(enrollment_dir, 'models.plda.h5')
        elif static_backend == 'GB':
            if len(domain.user_langs) == 0 and domain.quan:
                # base model
                model_path = domain.get_artifact('gb_int8.h5')
            elif len(domain.user_langs) == 0 and not domain.quan:
                model_path = domain.get_artifact('gb_f32.h5')
            else:
                model_path = os.path.join(enrollment_dir, 'models.gb.h5')
        else:
            self.escape_with_error("Please change 'static_backend' to one of 'PLDA' or 'GB' instead of '{}'".format(static_backend))

        if os.path.exists(model_path):
            try:
                domain.data_hash = idt.read_data_in_hdf5(model_path, nodes=['/hash'])[0]
#                domain.data_hash = -1
            except idt.utils.IdentoException:
                domain.data_hash = -1

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

                elif static_backend == 'GB':
                    domain.lda, domain.mean, domain.classes, domain.gb_means, domain.gb_wcov, domain.cal_fusion_models = \
                        idt.read_data_in_hdf5(model_path, nodes = ['/lda', '/mu', '/classes', '/gbm', '/gbw', '/models'])

        if new_hash != domain.data_hash:
            domain.data_hash = new_hash

            # Load base data
            if domain.quan:
                base_data = idt.read_data_in_hdf5(domain.get_artifact("base_data_int8.h5"))
            else:
                base_data = idt.read_data_in_hdf5(domain.get_artifact("base_data_f32.h5"))

            # Combine
            if static_language_data_use == 'AUGMENT':
                # Simple combination
                all_data = np.vstack([base_data['data'][base_data['seg2model'][:, 0], :], np.array(domain.lang_data).reshape(len(domain.lang_data), base_data['data'].shape[1])])
                all_map  = np.hstack([base_data['languages'][base_data['seg2model'][:, 1]], np.array(domain.lang_map)])
                all_sources = ['base']*base_data['data'].shape[0] + ['tgt']*len(domain.lang_map)
            elif static_language_data_use == 'USER':
                base_map = base_data['languages'][base_data['seg2model'][:, 1]]
                valid = ~np.in1d(base_map, np.unique(domain.lang_map))  # Indices of languages ONLY in the base data
                all_data = np.vstack([base_data['data'][base_data['seg2model'][valid, 0], :], np.array(domain.lang_data).reshape(len(domain.lang_data), base_data['data'].shape[1])])
                all_map  = np.hstack([base_data['languages'][base_data['seg2model'][valid, 1]], np.array(domain.lang_map)])
                all_sources = ['base']*valid.sum() + ['tgt']*len(domain.lang_map)
            else:
                self.escape_with_error("Please set 'static_language_data_use' to either 'AUGMENT' or 'USER' instead of '{}'".format(static_language_data_use))
             
            languages, lang_map = np.unique(all_map, return_inverse=True)

            # Train LDA
            lda, mu = self.snlda_retrain(all_data, lang_map, all_sources, 'tgt', lda_dim=languages.shape[0]-1)

            weights = len(lang_map)/np.bincount(lang_map)
            weights = weights[lang_map]
            data = idt.ivector_postprocessing(all_data, lda, mu, lennorm=True)

            if static_backend == 'PLDA':
                plda, obj = idt.SPLDA.train(data, lang_map, None, nb_em_it=100, weights=weights, rand_init=False)
            elif static_backend == 'GB':
                # Train weighted GB
                m, w = idt.GaussianBackend.train(data, lang_map, weights=weights)
                w = np.diag(np.diag(w))
            else:
                self.escape_with_error("Please change 'static_backend' to one of 'PLDA' or 'GB' instead of '{}'".format(static_backend))

            # Subset Calibration data to something reasonable
            keep = []
            for ilang, cnt in enumerate(np.bincount(lang_map)):
                inds = np.where(lang_map == ilang)[0]
                if cnt > static_max_calibration_segments_per_lang:
                    x = np.arange(len(inds), dtype=int)
                    np.random.seed(7)
                    np.random.shuffle(x)
                    keep += inds[x[:static_max_calibration_segments_per_lang]].tolist()
                else:
                    keep += inds.tolist()
            keep = np.sort(np.array(keep, dtype=int))
            data = data[np.array(keep, dtype=int)]
            lang_map = lang_map[np.array(keep, dtype=int)]

            # Train Calibration
            p = None
            if static_backend == 'PLDA':
                full_enroll_seg2model = np.vstack([np.arange(len(lang_map)), lang_map]).T
                domain.data, domain.enroll_seg2model = idt.average_data_by_model(data.copy(), full_enroll_seg2model.copy()) # Average the embeddings from each class to enroll them

                if static_precompute_enroll_stats:
                    segsperenrol = 1
                    plda.prepare_for_scoring(segsperenrol, 1)
                    domain.T_Q1, domain.T_f = plda.prepare_enrollments_for_scoring(domain.data, segsperenrol, 1, seg2model=domain.enroll_seg2model)
                    scores = plda.score_with_constantN_and_prepped_enrollment(data, domain.T_Q1, domain.T_f, 1, 1).T
                else:
                    scores = idt.plda_verifier(domain.data, data, plda, Tseg2model=domain.enroll_seg2model, tseg2model=None).T

                scores = idt.Scores(languages, np.arange(data.shape[0]).astype(str).tolist(), scores.T)
                scoredict = {'system0': scores}
                mask = np.ones(scores.score_mat.shape, dtype=int) * -1
                mask[full_enroll_seg2model[:, 1], full_enroll_seg2model[:, 0]] = 1
                key = idt.Key(scores.train_ids, scores.test_ids, mask)
                models = idt.fusion.train_calibration_and_fusion(scoredict, key, p, sid=True)

                # Save output
                domain.lda, domain.mean = lda, mu
                domain.classes, domain.plda = languages, plda
                domain.cal_fusion_models = models

                d = {'lda': lda, 'mu': mu, 'classes': domain.classes, 'plda': plda.save_to_dict(), 'models': models, 'hash': domain.data_hash, 'cal_fusion_models': domain.cal_fusion_models, 'enroll_seg2model': domain.enroll_seg2model}
                if static_precompute_enroll_stats:
                    d['T_Q1'] = domain.T_Q1
                    d['T_f'] = domain.T_f
                else:
                    d['data'] = domain.data

                idt.save_dict_in_hdf5(model_path, d)
            elif static_backend == 'GB':
                scores = idt.GaussianBackend.llks(m, w, data, addexpt=False)

                scores = idt.Scores(languages, np.arange(data.shape[0]).astype(str).tolist(), scores.T)
                scoredict = {'system0': scores}
                keyin = list(zip(scores.test_ids, languages[lang_map]))
                key = idt.LidKey.fromiter(keyin)
                models = idt.fusion.train_calibration_and_fusion(scoredict, key, priors=p, sid=False)

                # Save output
                domain.lda, domain.mean = lda, mu
                domain.classes, domain.gb_means, domain.gb_wcov = languages, m, w
                domain.cal_fusion_models = models

                d = {'lda': lda, 'mu': mu, 'classes': domain.classes, 'gbm': m, 'gbw': w, 'models': models, 'hash': domain.data_hash, 'cal_fusion_models': domain.cal_fusion_models}
                idt.save_dict_in_hdf5(model_path, d)
            else:
                self.escape_with_error("Please change 'static_backend' to one of 'PLDA' or 'GB' instead of '{}'".format(static_backend))


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

            logger.debug("Using user-defined parameter options, new config is: %s" % config)

        return config


    def run_sad(self, audio, config, speech=None):
        if speech is None:
            suppress_error = False
            if speech is None:
                # Overwrite embed system SAD config with the plugin specs. This can be updated on the fly with update_opts
                sad_opts = {'filter_length': config.sad_filter, 'interpolate': config.sad_interpolate, 'threshold': config.sad_threshold}
                speech   = idte.dnn_vad(self.sad_engine.feat_fc(audio), self.sad_engine.embedextractor,
                                        return_llrs=False, **sad_opts)
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

        return speech, duration


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
            feats_inp = domain.evalfeatiter(tmpaudio)
            splice = (domain.bn_extractor.model[0].in_shape / feats_inp.shape[-1])
            splice_indices = np.arange(splice).astype(int) - (splice-1) / 2
            feats_inp = FeatProcess.splice_by_indices(feats_inp, splice_indices)
            feats_inp = torch.tensor(feats_inp).float()
            logits, output = domain.bn_extractor(feats_inp.unsqueeze_(0))

            if domain.bn_linear_out:
                feats_BN = logits.squeeze_(0).detach().numpy()
            else:
                feats_BN = output.squeeze_(0).detach().numpy()

            feats = domain.bn_ppfunc(feats_BN, indices=speech)
            feats = mvn(feats, domain.embed_mu, domain.embed_inv_std)
            feats = torch.tensor(feats).float()

            with torch.no_grad():
                logits, output = domain.embed_nnet(feats.unsqueeze_(0))
                embed = logits.numpy()
            embeds.append(embed)
        embed = np.mean(embeds,0)

        return embed

    def map_score_names(self, scores, dialect_map, fnc=np.max):
        scores.train_ids = [dialect_map[x] for x in scores.train_ids]
        Tuniq, Tindices = np.unique(scores.train_ids, return_inverse=True)
        smat = np.zeros([len(Tuniq), len(scores.test_ids)])
        for ii, ind in enumerate(np.unique(Tindices)):
            smat[ii, :] = fnc(scores.score_mat[Tindices==ind, :], 0)
        return idt.Scores(Tuniq, scores.test_ids, smat)

    ### GlobalScorer ###
    def run_global_scoring(self, domain_id, audio, workspace, classes=None, opts=None):
        """
        Main scoring method
        """
        # Deal with region/channel opts
        if opts is not None and 'region' in opts:
            audio.trim_samples(opts['region'])

        domain = self.get_domains()[domain_id]
        audio.make_mono()

        # Update options if hey are passed in
        config = self.update_opts(opts, domain)

        # Check that languages are enrolled
        available, dialect_map, dialect_map_inv = self.list_classes(domain_id, is_mapped=static_enable_dialect_to_language_mapping, return_maps=True)
        if classes is not None:
            # Capitalize for consistency in checking
            classes = [cap_sentence(s) for s in classes]
            if not np.all(np.in1d(classes, available)):
                self.escape_with_error("Requested classes that are not available. Requested: [%s]. Available: [%s]" % (','.join(classes), ','.join(available)))
        else:
            classes = available

        if static_enable_dialect_to_language_mapping and dialect_map is not None:
            # We have to expand the mapped classes if any back to the subclasses for processing
            available_unmapped = self.list_classes(domain_id, is_mapped=False)
            classes = dialect_map_inv[np.in1d(dialect_map_inv[:,0], classes), 1].tolist()
            classes = [x for x in classes if x in available_unmapped]

        # create the alignment key, ready for calibration
        test_ids = [audio.id]
        align_key = idt.Key(domain.classes, test_ids, np.ones((len(domain.classes), 1), dtype=np.int8))

        speech, duration = self.run_sad(audio, config)

        # Get the embedding
        embeds = self.get_audio_vector(domain, audio, speech, config)

        # Process then score the embedding
        embeds = idt.ivector_postprocessing(embeds, domain.lda, domain.mean, lennorm=True)
        if static_backend == 'PLDA':
            if static_precompute_enroll_stats:
                scores = domain.plda.score_with_constantN_and_prepped_enrollment(embeds, domain.T_Q1, domain.T_f, 1, 1).T
            else:
                scores = idt.plda_verifier(domain.data, embeds, domain.plda, Tseg2model=domain.enroll_seg2model, tseg2model=None).T
            scores = idt.Scores(domain.classes, test_ids, scores.T)
            scores = idt.fusion.apply_calibration_and_fusion({'system0': scores}, align_key, domain.cal_fusion_models, sid=True)
        elif static_backend == 'GB':
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
        if static_enable_dialect_to_language_mapping and dialect_map is not None:
            scores = self.map_score_names(scores, dialect_map)

        final_scores = {lang: scores.score_mat[ii, 0] for ii, lang in enumerate(scores.train_ids)}

        return final_scores


    ### ClassEnroller/ClassModifier ###
    def add_class_audio(self, domain_id, audio, class_id, enrollspace, opts=None):

        domain = self.get_domains()[domain_id]
        config = self.update_opts(opts, domain)
        enrollment_dir = self.get_enrollment_storage(domain.get_id())

        class_id = cap_sentence(class_id)

        # Check that class_id is OK for enrollment without a class
        if os.path.exists(domain.get_artifact('dialect_language.map')):
            if not hasattr(domain, 'map_from_file'):
                map_from_file = dict(np.atleast_2d(np.loadtxt(domain.get_artifact('dialect_language.map'), delimiter='\t', dtype=str)))
            else:
                map_from_file = dict(domain.map_from_file)
            if class_id in set(map_from_file.values()):
                self.escape_with_error("Cannot enroll language with name '{}' as it exists in the domain file 'dialect_language.map' as a value to be mapped to. Either remove this mapping entry from the dialect_language.map or enroll under a different name. If the second option is taken, ensure that the language does not already exist in the plugin as discrimination will be hindered.".format(class_id))

        if opts is not None and 'region' in opts:
            audio.trim_samples(opts['region'])
            
        duration = audio.duration

        audio_dir = os.path.join(enrollspace, "staging", audio.id)
        utils.mkdirs(audio_dir)

        # OPTIONAL: Add chunkng of enrollment data for calibration adaptation
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

        class_id = cap_sentence(class_id)
        final_enrollment_dir = self.get_enrollment_storage(domain_id, class_id)

        removal_dir = os.path.join(enrollspace, "removals")
        if os.path.isdir(removal_dir):
            for file in os.listdir(removal_dir):
                target = os.path.join(final_enrollment_dir, file)
                if os.path.isdir(target):
                    shutil.rmtree(target)
            shutil.rmtree(removal_dir)
        
        for file in glob.glob(os.path.join(enrollspace, "staging", "*")):
            if len(os.listdir(file))>0:
                dest = os.path.join(final_enrollment_dir, os.path.basename(file))
                if os.path.exists(dest):
                    shutil.rmtree(dest)
                shutil.move(file, dest)
            else:
                logger.warn("Audio id [%s] for class_id [%s] failed to enroll" % (file, class_id))


    def remove_class(self, domain_id, class_id, workspace):
        class_id = cap_sentence(class_id)
        class_dir = self.get_enrollment_storage(domain_id, class_id)
        unpopulated = len(os.listdir(class_dir)) == 0  # Indicates this is a base class with no user data to unenroll
        shutil.rmtree(class_dir)
        if unpopulated:
            self.escape_with_error("Class '{}' is a built-in language and there is no user-enrolled data for this class to remove.".format(class_id))

    def snlda_retrain(self, full_data, full_spk_ids, full_source_idxs, source_id, lda_dim=150):
        # LDA retraining
        lda = idt.estimate_lda(full_data, full_spk_ids, lda_dim, whiten=True) #, full_source_idxs, whiten=True)
        # lda = idt.estimate_snlda(full_data, full_spk_ids, lda_dim, full_source_idxs, whiten=True)
        if lda.dtype.type == np.complex128:
            lda = idt.estimate_lda_robust(full_data, full_spk_ids, lda_dim, whiten=True)
        # uniqsrc, idxssrc = np.unique(full_source_idxs, return_inverse=True)
        # if np.sum(uniqsrc==source_id) > 0:
        #     indomidx = np.where(uniqsrc==source_id)[0][0]
        # else:
        #     indomidx = np.arange(len(uniqsrc), dtype=int)
        # mu = np.mean(full_data[idxssrc==indomidx], axis=0).dot(lda)
        mu = np.mean(full_data, axis=0).dot(lda)

        return lda, mu

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
