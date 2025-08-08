#!/usr/bin/env python
from olive.plugins import Plugin, BoundingBoxScorer, TraitType, TraitOption, logger
import os
from facenet_pytorch import MTCNN, PNet, RNet, ONet
import torch
from idento3 import Image, Config
import re
import numpy as np
import importlib

default_config = Config(dict(
##################################################
# Configurable parameters

# DETECTION OPTIONS
threshold  = 0.95,
frame_interval = 1.0,

))


class loadableMTCNN(MTCNN):

    def __init__(
        self, image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        select_largest=True, keep_all=False, device=None
    ):
        super().__init__()

        self.image_size = image_size
        self.margin = margin
        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.factor = factor
        self.post_process = post_process
        self.select_largest = select_largest
        self.keep_all = keep_all

        # Avoid loading models from runtime
        self.pnet = PNet(pretrained=False)
        self.rnet = RNet(pretrained=False)
        self.onet = ONet(pretrained=False)

        self.device = torch.device('cpu')
        if device is not None:
            self.device = device
            self.to(device)

    def populate_models(self, pnet_file, rnet_file, onet_file):
        # Load instead the models that are in this plugin
        self.pnet.load_state_dict(torch.load(pnet_file))
        self.rnet.load_state_dict(torch.load(rnet_file))
        self.onet.load_state_dict(torch.load(onet_file))


class CustomPlugin(Plugin, BoundingBoxScorer):

    # Not usually included in a plugin.py file
    def __init__(self):
        self.task = "FDV"
        self.label = "Face Detection Video (Commercial)"
        self.description = "A video-based face detection system using facenet embeddings"
        self.vendor = "SRI"
        self.group = "Olive"

        self.version = '1.0.0'
        self.minimum_runtime_version = "6.0.0"
        self.minimum_olive_version = "6.0.0"
        self.create_date = "2025-4-14"
        self.revision_date = "2025-4-14"
        self.loaded = False

        self.config   = default_config
        loader        = importlib.machinery.SourceFileLoader('plugin_config', os.path.join(os.path.dirname(os.path.realpath(__file__)), 'plugin_config.py'))
        spec          = importlib.util.spec_from_loader(loader.name, loader)
        mod           = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.config.update(mod.user_config)
        self.VALID_PARAMS  = ['region'] + list(default_config.keys())

    def load(self, domain_id, device=None):
        domain = self.get_domains()[domain_id]
        domain.device = device

        if not self.loaded:
            self.mtcnn = loadableMTCNN(image_size=160, margin=0, min_face_size=20,
                                       thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
                                       keep_all=True, select_largest=True).eval()
            pnet_file = self.get_artifact('pnet.pt')
            rnet_file = self.get_artifact('rnet.pt')
            onet_file = self.get_artifact('onet.pt')
            self.mtcnn.populate_models(pnet_file, rnet_file, onet_file)

            self.loaded = True

    def get_bounding_boxes(self, image):
        boxes, probs = self.mtcnn.detect(image.convert('RGB'), landmarks=False)
        return boxes, probs

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

    def list_classes(self, domain):
        return ['face']

    # Helper function to process user inputs for run-time options
    def update_opts(self, opts, domain):

        # Copy values
        config = Config(dict(self.config))

        if opts is not None:
            # Check that all passed options are valid for this plugin
            param_check = np.in1d(list(opts.keys()), self.VALID_PARAMS)
            if np.any(~param_check):
                raise Exception("Unknown parameter(s) passed [%s]. Please remove from the optional parameter list." % ','.join(np.array(list(opts.keys()))[param_check==False].tolist()))

            config.update(opts)

            # File-passed options are in in text format, so we need to convert these as necessary
            config.threshold   = float(config.threshold)
            config.frame_interval   = float(config.frame_interval)

            logger.debug("Using user-defined parameter options, new config is: %s" % config)

        return config

    def is_motion_imagery(self):
        return True

    def run_bounding_box_scoring(self, domain_id, DATA, workspace, classes=None, opts=None):

        domain = self.get_domains()[domain_id]

        if classes is not None and type(classes) == int: classes = None
        if opts is not None and type(opts) == str: opts = None

        if not hasattr(domain, 'cuda_device'):
            domain.cuda_device = self.get_cuda_device(domain_id)
        if domain.cuda_device != "-1":
            device = torch.device('cuda:{}'.format(domain.cuda_device))
        else:
            device = torch.device('cpu')
        self.mtcnn.device = device
        self.mtcnn.to(device)
        self.mtcnn.pnet.to(device)
        self.mtcnn.rnet.to(device)
        self.mtcnn.onet.to(device)

        config = self.update_opts(opts, domain)

        if (1 / DATA.fps) > config.frame_interval:
            logger.warn("Target frame_interval of [{}] may result in duplicate outputs for video with FPS of [{}]. Either increase frame_interval, or increase target_fps in the meta.conf of the domain (default 4)".format(config.frame_interval, DATA.fps))

        images_and_timestamps = DATA.generate_frames_by_interval(config.frame_interval)
        boxes, probs, ts_index, timestamps = [], [], [], []
        for ii, (image, timestamp) in enumerate(images_and_timestamps):
            # Process each image and return results
            timestamps += [timestamp]
            boxs, prbs = self.get_bounding_boxes(image.as_PIL_Image())
            #boxs, prbs = self.get_bounding_boxes(image)
            if boxs is not None:
                valid = np.where(np.array(prbs) > config.threshold)[0]
                if len(valid)>0:
                    boxes.append([boxs[idx] for idx in valid])
                    probs += [prbs[idx] for idx in valid]
                    ts_index += [ii]*len(valid)

        if len(boxes) == 0:
            return {"faces": []}

        boxes  = np.vstack(boxes)
        boxes[boxes<0] = 0
        probs  = np.vstack(probs).ravel()

        # get the timestamp range by averaging the timestamps between frames as they are not always consistent
        ts_index = np.array(ts_index)
        timestamps = np.array(timestamps + [DATA.duration])  # Add duration incase last frame has detection
        ts_index_s = np.clip(ts_index-1, 0, None)
        ts_s = np.mean([timestamps[ts_index], timestamps[ts_index_s]],0)
        ts_index_e = ts_index + 1
        ts_e = np.mean([timestamps[ts_index], timestamps[ts_index_e]],0)
        timestamps_range = np.vstack([ts_s, ts_e]).T

        bbox_scale = 1.0
        if DATA.original_frame_size is not None:
            bbox_scale = max(DATA.original_frame_size) / max(DATA.size)
        boxes = boxes*bbox_scale
        # We expect label: [(class, score, (bbox), (region))]
        results = [('face', probs[ibox], (int(box[0]), int(box[1]), int(box[2]), int(box[3])), (timestamps_range[ibox,0], timestamps_range[ibox,1])) for ibox, box in enumerate(boxes)]
        return {"faces": results}


    def get_bounding_box_scoring_opts(self):
        """
        These options are used in the OLIVE GUI and may be configured on the commandline by passing a file to --options
        """
        region_scoring_trait_options = [
            TraitOption('threshold', "Detection threshold", "Higher value results in less detections being output (default %0.1f)" % 0.0, TraitType.CHOICE_TRAIT, "", ""),
            TraitOption('frame_interval', "Frame interval", "Time in seconds between consecutive frames to be processed (default %0.1f)" % 1.0, TraitType.CHOICE_TRAIT, "", ""),
            ]

        return region_scoring_trait_options

# This line is very important! Every plugin should have one
plugin = CustomPlugin()
