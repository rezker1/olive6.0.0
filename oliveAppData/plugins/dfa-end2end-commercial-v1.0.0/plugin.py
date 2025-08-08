import os
import importlib
from pprint import pformat
from typing import Union

import upfirdn
import numpy as np
import torch
from transformers import AutoModelForAudioXVector

import idento3 as idt
from nn import NET
from olive.plugins import (
    Plugin,
    GlobalScorer,
    RegionScorer,
    TraitOption,
    TraitType,
    logger,
)

##################################################
# CONFIG - DO NOT CHANGE THESE UNLESS TOLD TO DO SO BY A DEVELOPER
default_config = idt.Config(
    dict(
        # DETECTION OPTIONS
        min_speech=0.3,
        threshold=0.0,
        score_offset=0.0,
        min_region=1.0,
        # SAD
        sad_threshold=0.0,
        sad_llr_offset=0.0,
        sad_filter=51,
        sad_interpolate=4,
        sad_merge=0.2,
        sad_padding=0.1,
    )
)
##################################################
# Static variables for this plugin
OFFSET = -1.25
chunk_len_in_secs = 4
sample_rate = 16000
batch_size = 1
min_length_seconds = 0.5
overlap_seconds = 3.5
tile_to_max_length = False
USE_BUFFERING = True
MAX_BUFFER = 1000000
MAX_CACHE_SEC = 300  # Memory reduction through cached processing
MIN_DUR = 0.31  # Mininmum duration in seconds

class AudioChunkIterator(torch.utils.data.IterableDataset):
    """A simple iterator class to return successive chunks of samples"""
    def __init__(
        self,
        samples: Union[np.ndarray, torch.Tensor],
        chunk_len_in_samples: int,
        min_length_samples: int = 0,
        overlap_samples: int = 0,
        normalize: bool = True,
    ):
        self._samples = samples
        self._chunk_len = chunk_len_in_samples
        self._min_len = min_length_samples
        self._start = 0
        self.output = True
        self.overlap = overlap_samples
        self.normalize = normalize

    def __iter__(self):
        return self

    def __len__(self):
        if len(self._samples) <= self._chunk_len:
            return 1
        else:
            return 2 + (len(self._samples) - self._chunk_len) // (
                self._chunk_len - self.overlap
            )

    def __next__(self):
        if not self.output:
            raise StopIteration
        last = int(self._start + self._chunk_len)
        if last <= len(self._samples):
            chunk = self._samples[self._start : last]
            section = (self._start, last)
            if self.normalize:
                chunk = (chunk - chunk.mean()) / np.sqrt(chunk.var() + 1e-7)
            self._start = last - self.overlap
        else:
            samp_len = len(self._samples) - self._start
            if samp_len < self._min_len:
                raise StopIteration
            if tile_to_max_length:
                chunk = self._samples[self._start : len(self._samples)]
                num_repeats = int(self._chunk_len / samp_len) + 1
                chunk = np.tile(chunk, (1, num_repeats))[0, : self._chunk_len]
            else:
                chunk = np.zeros([int(self._chunk_len)], dtype='float32')
                chunk[0:samp_len] = self._samples[self._start:len(self._samples)]
            section = (self._start, len(self._samples))
            if self.normalize:
                chunk = (chunk - chunk.mean()) / np.sqrt(chunk.var() + 1e-7)
            self.output = False
        return chunk, section


class CustomPlugin(Plugin, RegionScorer, GlobalScorer):
    def __init__(self):
        self.task = "DFA"
        self.label = "DeepFake Audio Detection (End-to-End)"
        self.description = "Fake/Generated Speech Detector"
        self.vendor = "SRI"
        self.version = "1.0.0"
        self.minimum_runtime_version = "5.7.1"
        self.minimum_olive_version = "5.7.1"
        self.create_date = "2025-03-25"
        self.revision_date = "2025-04-07"
        self.group = "Speech"
        self.loaded_domains = []
        self.loaded_base = False
        # These are the valid keys to check that users don’t pass rubbish keys in opts.
        self.VALID_PARAMS = set(["region", "speech_regions", "speech_frames"] + list(default_config.keys()))
        # Load the user_config in memory
        self.config = default_config
        loader = importlib.machinery.SourceFileLoader(
            "plugin_config",
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "plugin_config.py"
            ),
        )
        spec = importlib.util.spec_from_loader(loader.name, loader)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.config.update(mod.user_config)
        # Import end2end model definition (PyTorch modules etc)
        loader = importlib.machinery.SourceFileLoader(
            "dfamodel",
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "dfamodel", "dfamodel.py"
            ),
        )
        spec = importlib.util.spec_from_loader(loader.name, loader)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

    ### Method implementations required of a Plugin ###
    def list_classes(self, domain_id):
        return ["synthetic"]

    def get_cuda_device(self, domain_id):
        domain = self.get_domains()[domain_id]
        device_conf = (
            domain.config.get("domain", "device")
            if domain.device is None
            else domain.device
        )
        cuda_device = "-1"
        if not ("gpu" == device_conf[:3] or "cpu" == device_conf[:3]):
            self.escape_with_error(
                "'device' parameter in meta.conf of domain [{}] should be 'cpu' or 'gpuN' where N "
                "is the index of the GPU to use. Instead, it is set to '{}'".format(
                    domain_id, device_conf
                )
            )
        if "gpu" == device_conf[:3]:
            try:
                # Make sure gpu index can be extracted as int
                gpu_index = int(device_conf[3:])
            except ValueError:
                self.escape_with_error(
                    "'device' parameter in meta.conf of domain [{}] should be 'cpu' or 'gpuN' where N is the index of the GPU to use. Instead, it is set to '{}'".format(
                        domain_id, device_conf
                    )
                )
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                if os.environ["CUDA_VISIBLE_DEVICES"] == "":
                    self.escape_with_error(
                        "Requested gpu use in meta.conf of domain, but environment variable 'CUDA_VISIBLE_DEVICES' is empty. Either unset this variable or set it apprioriately to GPUs to be used"
                    )
                else:
                    cvd = np.array(
                        os.environ["CUDA_VISIBLE_DEVICES"].split(","), dtype=int
                    )
                    cvd_map = dict(zip(cvd, np.arange(len(cvd)).astype(int)))
                    if gpu_index not in cvd_map:
                        self.escape_with_error(
                            "Requested gpu {} in meta.conf of domain {} but this GPU was not listed in environment variable CUDA_VISIBLE_DEVICES.".format(
                                gpu_index, os.environ["CUDA_VISIBLE_DEVICES"]
                            )
                        )
                    else:
                        gpu_index = cvd_map[gpu_index]
            cuda_device = "{}".format(gpu_index)
            logger.info(
                "Allocated GPU {} to plugin/domain {}/{}".format(
                    cuda_device, self.label, domain_id
                )
            )
        return cuda_device

    def load(self, domain_id, device=None):
        domain = self.get_domains()[domain_id]
        domain.device = device
        # Load the base components (domain-independemt)
        if not self.loaded_base:
            # Base modules
            # SAD DNN
            dnn_file = self.get_artifact("nn.pnn")
            self.sad_mvn_data = np.load(self.get_artifact("mvn.npz"))
            self.sad_nnet = NET.load_from_file(dnn_file)
            # SAD feature config
            sadconfig = idt.Config()
            sadconfig.update_with_file(self.get_artifact("sad.config"))
            self.sadconfig = sadconfig
            self.sad_fc = idt.Frontend(sadconfig.featclass)(sadconfig)
            self.sad_llrframespersecond = idt.DEFAULT_FRAMES_PER_SECOND
            if 'vad_llrframespersecond' in self.sad_fc.config.keys():
                self.sad_llrframespersecond = self.sad_fc.config.vad_llrframespersecond

            # Fixed parameters determining the audio chunk length, batch size, and minimum
            # speech duration.
            self.chunk_len_in_samples = int(chunk_len_in_secs * sample_rate)
            self.min_length_samples = int(min_length_seconds * sample_rate)
            self.overlap_samples = int(overlap_seconds * sample_rate)

            # Load and initialize models
            domain.cuda_device = self.get_cuda_device(domain_id)
            domain.model_dir = self.get_artifact("dfamodel")
            if domain.cuda_device != "-1":
                logger.info("deferring cuda calls to processing thread")
            else:
                device = torch.device("cpu")
                domain.model = AutoModelForAudioXVector.from_pretrained(
                    domain.model_dir,
                    local_files_only=True,
                ).to(device)
                for p in domain.model.parameters():
                    p.requires_grad = False
                domain.model.eval()
            domain.device = device

            self.loaded_base = True

        # Domain dependent components
        if domain_id not in self.loaded_domains:
            domain = self.get_domains()[domain_id]

            # PLDA Model and LDA
            plda_model = domain.get_artifact("lda_plda.h5")
            plda_dict = idt.read_data_in_hdf5(plda_model)
            domain.lda = plda_dict["IvecTransform"]["LDA"]
            domain.mu = plda_dict["IvecTransform"]["Mu"]
            if ["U"] in list(plda_dict["PLDA"].keys()):
                domain.plda = idt.JPLDA.load_from_dict(plda_dict)
                domain.is_jplda = True
            else:
                domain.is_jplda = False
                domain.plda = idt.SPLDA.load_from_dict(plda_dict["PLDA"])

            # The calibration model
            cal_model = domain.get_artifact("bin_global.fusion.h5")
            domain.calibration_model = idt.read_data_in_hdf5(cal_model)
            # Load enrol data
            domain.enrol_data, domain.enrol_ids = idt.read_data_in_hdf5(
                domain.get_artifact("enrol.h5"), nodes=["/data", "/ids"]
            )
            self.loaded_domains.append(domain_id)
            logger.info(
                "Loading of plugin '%s' domain '%s' complete." % (self.label, domain_id)
            )

    def update_opts(self, opts, domain):
        # Copy values
        config = idt.Config(dict(self.config))

        if opts is not None:
            # Check that all passed options are valid for this plugin
            unk_params = [p for p in opts.keys() if p not in self.VALID_PARAMS]
            if len(unk_params) > 0:
                self.escape_with_error(
                    f"Unknown parameter(s) passed [{','.join(unk_params)}]."
                    "Please remove from the optional parameter list."
                )
            config.update(opts)
            config.min_speech = float(config.min_speech)
            config.sad_threshold = float(config.sad_threshold)
            config.score_offset = float(config.score_offset)
            config.sad_filter = int(config.sad_filter)
            config.sad_interpolate = int(config.sad_interpolate)
            config.sad_merge = float(config.sad_merge)
            config.sad_padding = float(config.sad_padding)
            config.threshold = float(config.threshold)
            config.min_region = float(config.min_region)
            logger.debug(
                "Using user-defined parameter options, new config is: %s" % config
            )

        return config

    def get_embeddings(
        self, model, audio, chunk_len_in_samples, overlap_samples, min_length_samples
    ):
        # Chunks audio into equal-sized pieces
        chunk_iterator = AudioChunkIterator(
            samples=audio,
            chunk_len_in_samples=chunk_len_in_samples,
            min_length_samples=min_length_samples,
            overlap_samples=overlap_samples,
        )
        # Dataloader for batching of chunks
        loader = torch.utils.data.DataLoader(chunk_iterator, batch_size=batch_size)
        # Compute embeddings
        embeddings = []
        sections = []
        # Compute sum of chunk embeddings
        with torch.inference_mode():
            for batch, secs in loader:
                # batch = torch.from_numpy(np.expand_dims(np.cos(np.linspace(-1,1, 16000).repeat(4)), axis=0).astype(np.float32))
                batch = batch.to(model.device)
                outputs = model(batch)
                embeds = outputs.embeddings.cpu().numpy()
                embeddings.extend([e for e in embeds])
                for st, en in zip(secs[0], secs[1]):
                    sections.append((float(st), float(en)))
        return embeddings, sections

    def buffered_call(self, fnc, data, buffer_size, overlap, feat_shift, *args, **kwargs):
        """
        Allows fnc to be called multiple times by chunking data, then stitch
        the result together
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
        if shift % feat_shift > 0:
            shift = int(shift - shift % feat_shift + feat_shift)
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
            if etrim == res.shape[0]:
                break  # In case batch size is smaller than overlap
        return np.hstack(result)

    #def run_sad(self, audio, config, speech=None):
    def run_sad(self, domain_id, audio, config, speech=None, suppress_error=False):
        domain = self.get_domains()[domain_id]
        # Set SAD to correct device
        self.sad_nnet = self.sad_nnet.to(domain.device)
        self.sad_nnet.eval()

        def get_nnet_llrs(batch_data, device, interpolate=4, filter_length=1):
            # Compute features
            batch_audio = idt.Wave(batch_data, audio.sample_rate)
            feat = self.sad_fc(batch_audio)
            feat -= self.sad_mvn_data['mean']
            feat *= self.sad_mvn_data['inv_std']
            feat = torch.tensor(feat)
            feat = feat.to(device)
            with torch.no_grad():
                _, logits = self.sad_nnet(feat[0::interpolate])
            complete = feat.shape[0]
            llrs = idt.utils.logit(logits[:,-1].detach().cpu())
            # Interpolate if needed
            if interpolate != 1:
                w = np.r_[np.linspace(0, 1, interpolate+1), np.linspace(1, 0, interpolate+1)[1:]][1:-1]
                llrs = upfirdn.upfirdn(llrs, w, interpolate)[interpolate - 1: interpolate - 1 + complete]
            # filt llr
            winm = np.ones(filter_length)
            llrs = np.convolve(llrs, winm/np.sum(winm), 'same')
            return llrs

        if speech is None:
            # A buffered call to produce the features
            batch_size = 1 * MAX_CACHE_SEC * audio.sample_rate  # MAX_CACHE_SEC seconds of audio
            overlap = audio.sample_rate * 2  # overlap of two seconds

            if not USE_BUFFERING:
                llrs = get_nnet_llrs(audio.data, domain.device, interpolate=self.sad_fc.config.vad_interpolate, filter_length=config.sad_filter)
            else:
                audio = self.sad_fc.audio_func(audio, channels=None, bounds=None, **self.sad_fc.config)
                def passthrough(wave, **kwargs):
                    return wave
                oldfnc = self.sad_fc.audio_func
                self.sad_fc.audio_func = passthrough
                llrs = self.buffered_call(get_nnet_llrs, audio.data, batch_size, overlap, self.sad_fc.config.window - self.sad_fc.config.overlap,
                    domain.device, interpolate=self.sad_fc.config.vad_interpolate, filter_length=config.sad_filter)
                self.sad_fc.audio_func = oldfnc

            # For multi-fast-v2 domain: Convert 50 frames/sec to 100 frames/sec
            if self.sad_llrframespersecond != idt.DEFAULT_FRAMES_PER_SECOND:
                if (idt.DEFAULT_FRAMES_PER_SECOND % self.sad_llrframespersecond) != 0:
                    self.escape_with_error("The frame rate [%d] must be divisible by vad_llrframespersecond. Chose a different vad_llrframespersecond." % idt.DEFAULT_FRAMES_PER_SECOND)
                interpol_factor = int(idt.DEFAULT_FRAMES_PER_SECOND / self.sad_llrframespersecond)
                w = np.r_[np.linspace(0, 1, interpol_factor + 1), np.linspace(1, 0, interpol_factor + 1)[1:]][1:-1]
                llrs = upfirdn.upfirdn(llrs, w, interpol_factor)[interpol_factor - 1: interpol_factor - 1 + 2*llrs.shape[0]]

            filter = config.sad_filter
            # Smoothing
            if filter > 0:
                if llrs.shape[0] < float(filter):
                    filter = llrs.shape[0]
                llrs = np.convolve(llrs, np.ones(filter) / float(filter), 'same')

            # LLR calibration for threshold 0.0
            llrs = llrs - config.sad_llr_offset

            # Map to correct values and pad ends so that the full audio duration is annotated
            target_shape = int(audio.duration * idt.DEFAULT_FRAMES_PER_SECOND)
            missing = int(target_shape - llrs.shape[0])
            add = missing // 2
            extra = missing - add
            llrs = np.hstack([[llrs[0]]*add, llrs, [llrs[-1]]*extra])
            speech = llrs > np.float64(config.sad_threshold)

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

    ### Global scorer ###
    def run_global_scoring(self, domain_id, audio, workspace, classes=None, opts=None):
        """
        Main scoring method
        """
        domain = self.get_domains()[domain_id]
        # Update options if hey are passed in
        config = self.update_opts(opts, domain)

        # Load and initialize models
        if not hasattr(domain, "model") and domain.cuda_device != "-1":
            device = torch.device("cuda:{}".format(domain.cuda_device))
            domain.model_dir = self.get_artifact("dfamodel")
            domain.model = AutoModelForAudioXVector.from_pretrained(
                domain.model_dir,
                local_files_only=True,
            ).to(device)
            for p in domain.model.parameters():
                p.requires_grad = False
            domain.model.eval()
        else:
            device = torch.device("cpu")
        domain.device = device
        audio = audio.resample(sample_rate)
        audio.make_mono()

        timing_map = None
        if opts is not None:
            if "speech_regions" in opts and "speech_frames" in opts:
                logger.warn(
                    "Both 'speech_regions' and 'speech_frames' passed. Using only speech_regions."
                )
            if "region" in opts and "speech_regions" in opts:
                logger.warn(
                    "Both 'region' and 'speech_regions' passed. Using only speech_regions."
                )
            if "region" in opts:
                timing_map = opts["region"]
                del opts["region"]  # So frame scoring doesn't see it
            if "speech_regions" in opts:
                timing_map = opts["speech_regions"].get_result()["speech"]
                del opts["speech_regions"]  # So frame scoring doesn't see it
        if timing_map is None:
            timing_map = [(0, audio.duration)]

        all_scores = []
        for reg in timing_map:
            if (reg[1] - reg[0]) < MIN_DUR:
                logger.warn(
                    "Region [{}, {}] shorter than minimum duration; moving to next region.".format(
                        reg[0], reg[1]
                    )
                )
                continue
            # If there is only one region/selection, reuse audio, otherwise create a copy
            if len(timing_map) == 1:
                audio_selection = audio
            else:
                audio_selection = idt.Wave(
                    audio.data.copy(), audio.sample_rate, audio.id
                )
            audio_selection.trim_samples([reg])
            # Get SAD indices
            speech, duration = self.run_sad(domain_id, audio_selection, config)
            if duration < config.min_speech:
                logger.warn(
                    "Speech content in region [{}, {}] shorter than minimum duration ({}); moving to next region.".format(
                        reg[0], reg[1], config.min_speech
                    )
                )
                continue

            audio_chunk = idt.Wave(
                audio_selection.data.copy(), audio_selection.sample_rate, audio.id
            )
            regS = [
                (idt.frame_to_seconds(i[0]), idt.frame_to_seconds(i[1]))
                for i in idt.get_speech_dur_segments(speech)
            ]
            audio_chunk.trim_samples(regS)

            # Compute embeddings
            embeddings, _ = self.get_embeddings(
                model=domain.model,
                audio=audio_chunk.data,
                chunk_len_in_samples=self.chunk_len_in_samples,
                overlap_samples=0,
                min_length_samples=self.min_length_samples,
            )
            embeddings = np.mean(np.stack(embeddings), axis=0, keepdims=True)
            # Scoring part
            enrol_ivs = idt.ivector_postprocessing(
                domain.enrol_data, domain.lda, domain.mu, lennorm=True
            )
            test_ivs = idt.ivector_postprocessing(
                embeddings, domain.lda, domain.mu, lennorm=True
            )

            Tseg2mod = np.arange(enrol_ivs.shape[0])
            Tseg2mod = np.c_[Tseg2mod, np.zeros(enrol_ivs.shape[0], dtype=int)].astype(int)
            tseg2mod = np.arange(test_ivs.shape[0])
            tseg2mod = np.c_[tseg2mod, tseg2mod].astype(int)

            score_mat = idt.plda_verifier(
                enrol_ivs,
                test_ivs,
                domain.plda,
                Tseg2model=Tseg2mod,
                tseg2model=tseg2mod,
            )
            test_ids = ["test_" + str(i) for i in range(test_ivs.shape[0])]
            scores = idt.Scores(["real"], test_ids, score_mat)
            score_map = {"system0": scores}
            align_key = idt.scoring.Key(scores.train_ids, scores.test_ids, scores.mask)
            cal_scores = idt.fusion.apply_calibration_and_fusion(
                score_map, align_key, domain.calibration_model
            )
            final_score = np.mean(
                -1.0 * cal_scores.score_mat - OFFSET - config.score_offset,
                axis=0,
                keepdims=True,
            )
            all_scores.append(final_score)
        overall_score = float(np.hstack(all_scores).mean())

        return {"synthetic": overall_score}

    def get_global_scoring_opts(self):
        """
        These options are used in the OLIVE GUI and may be configured on the commandline by passing a file to --options
        """
        global_scoring_trait_options = [
            TraitOption(
                "sad_threshold",
                "Threshold to determine speech frames",
                "Higher value results in less speech for processing",
                TraitType.CHOICE_TRAIT,
                "",
                self.config.sad_threshold,
            ),
            TraitOption(
                "sad_merge",
                "Duration in seconds to merge SAD sections.",
                "Higher value results in more merged speech sections for processing",
                TraitType.CHOICE_TRAIT,
                "",
                self.config.sad_merge,
            ),
            TraitOption(
                "sad_padding",
                "Duration in seconds to pad SAD sections.",
                "Higher value results in longer/padded speech sections for processing",
                TraitType.CHOICE_TRAIT,
                "",
                self.config.sad_padding,
            ),
            TraitOption(
                "min_speech",
                "Amount of speech needed to process audio",
                "Higher value results in less scores being output, but higher confidence (default 0.3)",
                TraitType.CHOICE_TRAIT,
                "",
                self.config.min_speech,
            ),
            TraitOption(
                "score_offset",
                "Offset final DF score",
                "Higher value results in less DF detection, but higher confidence (default 0.3)",
                TraitType.CHOICE_TRAIT,
                "",
                self.config.score_offset,
            ),
        ]
        return global_scoring_trait_options

    def run_region_scoring(self, domain_id, audio, workspace, classes=None, opts=None):
        """Segmentation-by-Classification DFA detection approach"""
        domain = self.get_domains()[domain_id]
        # Update options if they are passed in
        config = self.update_opts(opts, domain)
        # Load and initialize models
        if not hasattr(domain, "model") and domain.cuda_device != "-1":
            device = torch.device("cuda:{}".format(domain.cuda_device))
            domain.model_dir = self.get_artifact("dfamodel")
            domain.model = AutoModelForAudioXVector.from_pretrained(
                domain.model_dir,
                local_files_only=True,
            ).to(device)
            domain.model.eval()
        else:
            device = torch.device("cpu")
        domain.device = device
        enrol_embeds = idt.ivector_postprocessing(
            domain.enrol_data, domain.lda, domain.mu, lennorm=True
        )
        enrol_ids = np.arange(enrol_embeds.shape[0], dtype=int)
        enrol_seg2model = np.c_[enrol_ids, np.zeros(enrol_embeds.shape[0], dtype=int)].astype(int)

        audio = audio.resample(sample_rate)
        audio.make_mono()

        timing_map = None
        if opts is not None:
            if "speech_regions" in opts and "speech_frames" in opts:
                logger.warn(
                    "Both 'speech_regions' and 'speech_frames' passed. Using only speech_regions."
                )
            if "region" in opts and "speech_regions" in opts:
                logger.warn(
                    "Both 'region' and 'speech_regions' passed. Using only speech_regions."
                )
            if "region" in opts:
                timing_map = opts["region"]
                del opts["region"]  # So frame scoring doesn't see it
            if "speech_regions" in opts:
                timing_map = opts["speech_regions"].get_result()["speech"]
                del opts["speech_regions"]  # So frame scoring doesn't see it
        if timing_map is None:
            timing_map = [(0, audio.duration)]

        outputs = []
        regoffset = 0.0
        for reg in timing_map:
            if (reg[1] - reg[0]) < MIN_DUR:
                logger.warn(
                    "Region [{}, {}] shorter than minimum duration; moving to next region.".format(
                        reg[0], reg[1]
                    )
                )
                continue
            regoffset = reg[0] * 1000.0
            # If there is only one region/selection, reuse audio, otherwise create a copy
            if len(timing_map) == 1:
                audio_selection = audio
            else:
                audio_selection = idt.Wave(
                    audio.data.copy(), audio.sample_rate, audio.id
                )
            audio_selection.trim_samples([reg])
            # Get SAD indices
            speech, duration = self.run_sad(domain_id, audio_selection, config)
            if duration < config.min_speech:
                logger.warn(
                    "Speech content in region [{}, {}] shorter than minimum duration ({}); moving to next region.".format(
                        reg[0], reg[1], config.min_speech
                    )
                )
                continue
            all_embeds = []
            align_seconds = []
            for seg in idt.get_speech_dur_segments(speech):
                regS = (idt.frame_to_seconds(seg[0]), idt.frame_to_seconds(seg[1]))
                audio_chunk = idt.Wave(
                    audio_selection.data.copy(), audio_selection.sample_rate, audio.id
                )
                audio_chunk.trim_samples([regS])
                # Compute embeddings
                embeddings, startend_indices = self.get_embeddings(
                    model=domain.model,
                    audio=audio_chunk.data,
                    chunk_len_in_samples=self.chunk_len_in_samples,
                    overlap_samples=self.overlap_samples,
                    min_length_samples=self.min_length_samples,
                )
                all_embeds.extend(embeddings)
                for st, en in startend_indices:
                    align_seconds.append(
                        (regS[0] + (st / sample_rate), regS[0] + (en / sample_rate))
                    )
            all_embeds = np.stack(all_embeds)
            # Scoring part
            test_embeds = idt.ivector_postprocessing(
                all_embeds, domain.lda, domain.mu, lennorm=True
            )
            test_ids = np.arange(test_embeds.shape[0], dtype=int)
            test_seg2model = np.vstack([test_ids, test_ids]).T
            score_mat = idt.plda_verifier(
                enrol_embeds,
                test_embeds,
                domain.plda,
                Tseg2model=enrol_seg2model,
                tseg2model=test_seg2model,
            )
            scores = idt.Scores(["real"], test_ids.tolist(), score_mat)
            score_map = {"system0": scores}
            align_key = idt.scoring.Key(scores.train_ids, scores.test_ids, scores.mask)
            cal_scores = idt.fusion.apply_calibration_and_fusion(
                score_map, align_key, domain.calibration_model
            )
            final_score_mat = np.mean(
                -1.0 * cal_scores.score_mat - OFFSET - config.score_offset, axis=0
            )
            score_array = np.ones(int(audio.duration * 100 + 1), dtype=np.float32) * -1.0

            align = idt.Alignment()
            regpad = 0.2
            for score, (st, en) in zip(final_score_mat, align_seconds):
                if score > config.threshold:
                    align.add(
                        audio.id,
                        1,
                        max(st - regpad, 0.0),
                        min(en + regpad, audio.duration),
                        "synthetic",
                    )
                    score_array[int(st * 100) : int(en * 100)] = score
                    logger.info(f"Found synthetic from {st} to {en} (score: {score})")

            if audio.id in align.keys():
                result = []
                for st, en, topic_id in align.get_start_end_label(audio.id):
                    if en - st >= config.min_region * 100:  # Apply minimum constraints
                        result += [
                            (
                                regoffset + int(st * 10.0),
                                regoffset + int(en * 10.0),
                                topic_id,
                                score_array[st:en].max(),
                            )
                        ]
                outputs.extend(result)

        if len(outputs) > 0:
            outputs = sorted(outputs, key=lambda x: x[0])
            # Convert timestamps from ms to s
            outputs = [
                (
                    np.float32(res[0] / 1000.0),
                    np.float32(res[1] / 1000.0),
                    res[2],
                    res[3],
                )
                for res in outputs
            ]
            logger.info(pformat(outputs))
        else:
            logger.info(f"No synthetic sections found in audio '{audio.id}'.")

        return {"synthetic": outputs}

    def get_region_scoring_opts(self):
        """
        These options are used in the OLIVE GUI and may be configured on the commandline by passing a file to --options
        """
        region_scoring_trait_options = [
            TraitOption(
                "threshold",
                "Detection threshold",
                "Higher value results in less detections being output (default %0.1f)"
                % self.config.threshold,
                TraitType.CHOICE_TRAIT,
                "",
                self.config.threshold,
            ),
            TraitOption(
                "min_speech",
                "Minimum speech for detection",
                "Required amount of speech in file to perform deepfake audio detection (default %0.1f)"
                % self.config.min_speech,
                TraitType.CHOICE_TRAIT,
                "",
                self.config.min_speech,
            ),
            TraitOption(
                "sad_threshold",
                "Threshold to determine speech frames",
                "Higher value results in less speech fro processing",
                TraitType.CHOICE_TRAIT,
                "",
                self.config.sad_threshold,
            ),
            TraitOption(
                "score_offset",
                "Offset final DF score",
                "Higher value results in less DF detection, but higher confidence (default 0.3)",
                TraitType.CHOICE_TRAIT,
                "",
                self.config.score_offset,
            ),
            TraitOption(
                "min_region",
                "Minimum duration of synthetic region",
                "Required amount of detected synthetic speech for a region.",
                TraitType.CHOICE_TRAIT,
                "",
                self.config.min_region,
            ),
        ]
        return region_scoring_trait_options


# This line is very important! Every plugin should have one
plugin = CustomPlugin()
