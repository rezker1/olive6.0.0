#!/usr/bin/env python
import os, sys, re, glob
from olive.plugins import Plugin, BoundingBoxScorer, TraitType, TraitOption, logger, ClassModifier, utils
import idento3 as idt
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, extract_face
import cv2
import torch
import importlib
import shutil

default_config = idt.Config(dict(
##################################################
# Configurable parameters

# DETECTION OPTIONS
threshold  = 0.0,
face_detection_threshold = 0.95,

# PROCESSING OPTIONS
keep_only_one_score_per_enrollee = True,
keep_only_highest_score_per_face = True,
remove_overlapping_faces_from_highest_scoring_face = True,

))

OUTPUT_ALL_FACES_PER_IMAGE = True

class CustomPlugin(Plugin, BoundingBoxScorer, ClassModifier):

    # Not usually included in a plugin.py file
    def __init__(self):
        self.task = "FRI" #TODO check if scenic can read this class
        self.label = "Face Recognition Image (Commercial)"
        self.description = "A face recognition system based on facenet embeddings"
        self.vendor = "SRI"
        self.version = '1.0.0'
        self.minimum_runtime_version = '6.0.0'
        self.minimum_olive_version = '6.0.0'
        self.create_date = "2025-4-14"
        self.revision_date = "2025-5-14"
        self.group = "Imagery"
        self.loaded = False
        self.loaded_domains = []
        self.config          = default_config
        loader               = importlib.machinery.SourceFileLoader('plugin_config', os.path.join(os.path.dirname(os.path.realpath(__file__)), 'plugin_config.py'))
        spec                 = importlib.util.spec_from_loader(loader.name, loader)
        mod                  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.config.update(mod.user_config)
        self.VALID_PARAMS    = ['region'] + list(default_config.keys()) # For checking user inputs and flagging unknown paramters.

    def is_motion_imagery(self):
        return False

    def get_folder_update_timestamp(self, dir):
        ts = max(map(lambda x: os.path.getmtime(x[0]), os.walk(dir)))
        return ts

    def load(self, domain_id, device=None):
        domain = self.get_domains()[domain_id]
        domain.device = device

        if not self.loaded:
            self.mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20,
                            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
                            keep_all=OUTPUT_ALL_FACES_PER_IMAGE,select_largest=False).eval()
              
            self.loaded=True        

        if 'domain_id' not in self.loaded_domains:
            nnet_feats_file = domain.get_artifact("vggface.features.pt")
            nnet_logits_file = domain.get_artifact("vggface.logits.pt")
            domain.resnetVGG = InceptionResnetV1(num_classes=8631)
            try:
                state_dict = {}
                state_dict.update(torch.load(nnet_feats_file))
                state_dict.update(torch.load(nnet_logits_file))
                # Embedding models
                domain.resnetVGG.load_state_dict(state_dict)
            except:
                # Handles case of newer facenet that doesn't know about logits.weights and logits.bias
                state_dict = {}
                state_dict.update(torch.load(nnet_feats_file))
                domain.resnetVGG.load_state_dict(state_dict)
            domain.resnetVGG = domain.resnetVGG.eval()

            plda_params = idt.read_data_in_hdf5(domain.get_artifact("plda.h5"))
            domain.lda = plda_params['IvecTransform']['LDA']
            domain.mu  = plda_params['IvecTransform']['Mu']
            domain.plda_model = idt.SPLDA.load_from_dict(plda_params['PLDA'])

            # Load calibration models
            domain.cal_model = idt.read_data_in_hdf5(domain.get_artifact("fid_cal.h5"))
            domain.cal_offset = float(open(domain.get_artifact("cal.offset")).readlines()[0].strip())

            self.update_classes(domain_id)

            self.loaded_domains.append(domain_id)

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

    ### LIST CLASS
    def list_classes(self, domain_id):
        # List all the classes in the enrollments directory
        domain = self.get_domains()[domain_id]
        enrollments_dir = self.get_enrollment_storage(domain.get_id())
        retVal = [x for x in os.listdir(enrollments_dir) if x != '__data__']
        return retVal

    # Helper function to process user inputs for run-time options
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
            config.threshold   = float(config.threshold)
            config.face_detection_threshold   = float(config.face_detection_threshold)

            if type(config.keep_only_one_score_per_enrollee) == str:
                config.keep_only_one_score_per_enrollee  = True if config.keep_only_one_score_per_enrollee == 'True' else False
            if type(config.keep_only_highest_score_per_face) == str:
                config.keep_only_highest_score_per_face  = True if config.keep_only_highest_score_per_face == 'True' else False
            if type(config.remove_overlapping_faces_from_highest_scoring_face) == str:
                config.remove_overlapping_faces_from_highest_scoring_face  = True if config.remove_overlapping_faces_from_highest_scoring_face == 'True' else False

            logger.debug("Using user-defined parameter options, new config is: %s" % config)

        return config

    def extract_img_embedding(self, domain, image, config, single_face=False, bbox=None, allow_no_faces=False, error_if_multi_face=False, device='cpu'):
        # Detects multiple faces
        # Use passed bbox if available, otherwise detect the faces
        if bbox is not None and len(bbox)>0:
            boxes = bbox

            # Check bbox form left, top, right, bottom
            if len(boxes)!=4 or np.any(boxes) < 0 or boxes[0] >= boxes[2] or boxes[1] >= boxes[3] or boxes[2] > image.width or boxes[3] > image.height:
                self.escape_with_error("Enrollment bounding box is not in a valid format (4 numbers [left, top, right, bottom] within image size [0, 0, %d, %d]" % (image.width, image.height))

            face = extract_face(image, boxes, self.mtcnn.image_size, self.mtcnn.margin, None)
            if self.mtcnn.post_process:
                face = fixed_image_standardization(face)
            face = face[np.newaxis,:,:,:]
            
        else:
            boxes, probs = self.mtcnn.detect(image, landmarks=False)
            if boxes is None:
                valid = None
            else:
                valid = np.where(np.array(probs) > config.face_detection_threshold)[0]
            if valid is None or len(valid)==0:
                if allow_no_faces:
                    return None, None
                else:
                    self.escape_with_message("No faces detected in image")
            if error_if_multi_face and len(valid)>1:
                self.escape_with_message("Multiple faces ({}) detected in image, but expected single face. Either crop the image or set a higher face detection threshold.".format(len(boxes)))

            face = self.mtcnn(image, return_prob=False)

            # Subset to those with face detection above the threshold
            boxes = [boxes[idx] for idx in valid]
            probs = [probs[idx] for idx in valid]
            face  = face[valid]
            if not OUTPUT_ALL_FACES_PER_IMAGE:
                boxes = boxes[0] # Just the highest
                boxes = boxes[np.newaxis,:]
                face = face[np.newaxis,:,:,:]
            if (single_face and len(boxes)>1):
                boxes = boxes[0]
                face = face[0]
                face = face[np.newaxis,:,:,:]
        face = face.to(device)
        embedding = domain.resnetVGG(face)
        embeds = embedding.detach().cpu().numpy()
        return embeds, boxes

    def run_bounding_box_scoring(self, domain_id, DATA, workspace, classes=None, opts=None):
        # Process each audio file and return results
        
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
        domain.resnetVGG.to(device)

        config = self.update_opts(opts, domain)

        # Check sync of disk vs memory models
        ts = self.get_folder_update_timestamp(self.get_enrollment_storage(domain.get_id()))
        if ts != domain.enroll_ts:
            logger.warn("Synchronizing face models on disk with those in memory.")
            self.update_classes(domain)

        available = self.list_classes(domain_id)
        if classes is None:
            # print("--> run_region_scoring is about to call list_classes") # DEBUG
            classes = available
        else:
            if not np.all(np.in1d(classes, available)):
                self.escape_with_error("Requested classes that are not available. Requested: [%s]. Available: [%s]" % (','.join(classes), ','.join(available)))
        if len(classes)==0:
            raise Exception("No face models enrolled for plugin")

        is_video = False
        
        # For single image
        image = DATA
        embeds, boxes = self.extract_img_embedding(domain, image.as_PIL_Image().convert('RGB'), config, device=device)

        nfaces = embeds.shape[0]
        test_seg2model = np.vstack([np.arange(nfaces), np.arange(nfaces)]).T

        enr_data = idt.ivector_postprocessing(domain.enroll_data,  domain.lda, domain.mu, lennorm=True)
        embeds = idt.ivector_postprocessing(embeds, domain.lda, domain.mu, lennorm=True)
        
        score_mat = idt.plda_verifier(
            enr_data, embeds, domain.plda_model, Tseg2model=domain.enroll_seg2model,
            tseg2model=test_seg2model)

        scores = idt.Scores(domain.enroll_ids, np.arange(nfaces).astype(str).tolist(), score_mat)
        score_dict = {}
        score_dict['system0'] = scores

        scr = score_dict.values()
        align_key = idt.scoring.Key(scores.train_ids, scores.test_ids, scores.mask)

        calscores=idt.fusion.apply_calibration_and_fusion(score_dict, align_key, domain.cal_model, sid=True)
        calscores.score_mat += domain.cal_offset

        # For each enrolled speaker, report a single result as the highest scoring speaker cluster from diarization
        boxes = np.array(boxes)
        boxes[boxes<0] = 0
        boxids = ['-'.join((str(int(boxes[ii][0])), str(int(boxes[ii][1])), str(int(boxes[ii][2])), str(int(boxes[ii][3])))) for ii in np.arange(nfaces).astype(int)]

        if config.keep_only_one_score_per_enrollee:
            inds = calscores.score_mat.argmax(1)
            scrs = calscores.score_mat.max(1)
            calscores.score_mat[:] = config.threshold - 1.0 # So they can't be selected in the return list
            calscores.score_mat[np.arange(len(inds)).astype(int), inds] = scrs # Re-enter the top scores

        if config.keep_only_highest_score_per_face:
            inds = calscores.score_mat.argmax(0)
            scrs = calscores.score_mat.max(0)
            calscores.score_mat[:] = config.threshold - 1.0 # So they can't be selected in the return list
            calscores.score_mat[inds, np.arange(len(inds)).astype(int)] = scrs # Re-enter the top scores

        if config.remove_overlapping_faces_from_highest_scoring_face:
            # Iterate over the test faces, prioritizing the higest scores
            unfiltered = np.ones(len(calscores.test_ids)).astype(bool)
            bboxarr_lr = [np.linspace(int(bbox[0]),int(bbox[2]),int(bbox[2]-bbox[0]+2)).astype(int) for bbox in boxes]
            bboxarr_ud = [np.linspace(int(bbox[1]),int(bbox[3]),int(bbox[3]-bbox[1]+2)).astype(int) for bbox in boxes]

            to_remove = []
            calscores_orig = calscores.score_mat.copy()
            while np.any(unfiltered) and calscores.score_mat.max()>=config.threshold:
                max_tst_ind = np.where(calscores.score_mat==calscores.score_mat.max())[1][0]
                calscores.score_mat[:, max_tst_ind] = config.threshold - 1.0 # Suppress from further analysis in this loop
                unfiltered[max_tst_ind] = False
                for ind in np.where(unfiltered)[0]:
                    overlap = np.any(np.in1d(bboxarr_lr[ind],bboxarr_lr[max_tst_ind])) and np.any(np.in1d(bboxarr_ud[ind],bboxarr_ud[max_tst_ind]))
                    if overlap:
                        # remove from the score set
                        print(boxids[ind], boxids[max_tst_ind])
                        to_remove.append(ind)
                        calscores.score_mat[:, ind] = config.threshold - 1.0 # Suppress from further analysis in this loop
                        unfiltered[ind] = False
            # Revert scores
            calscores.score_mat = calscores_orig
            boxids = [x for i,x in enumerate(boxids) if i not in to_remove]
            boxes = [x for i,x in enumerate(boxes) if i not in to_remove]
            calscores = calscores.filter(calscores.train_ids, [x for i,x in enumerate(calscores.test_ids) if i not in to_remove])

        scorelist = []
        for idx, clsid in enumerate(calscores.train_ids):
            for tidx in np.arange(len(calscores.test_ids)).astype(int):
                tst_scr = calscores.score_mat[idx,tidx]
                if tst_scr >= config.threshold:
                    box = boxes[tidx]
                    scorelist.append((clsid, tst_scr, (int(box[0]), int(box[1]), int(box[2]), int(box[3])), ())) 
        results = {'speakers': scorelist}
        return results


    ### ENROLLMENT and CLASS UPDATES
    def update_classes(self, domain_id):
        # Function that is called by the server after finalizing all enrollments
        domain = self.get_domains()[domain_id]
        enrollment_dir = self.get_enrollment_storage(domain.get_id())
        speaker_ids = self.list_classes(domain.get_id())
        domain.enroll_ts        = self.get_folder_update_timestamp(enrollment_dir)
        domain.speaker_data={}
        domain.speaker_logids={}
        speaker_map = []
        for speaker_id in speaker_ids:
            iv_path = os.path.join(enrollment_dir, speaker_id, "__data__", "data.h5")
            try:
                logids, embeds = idt.read_data_in_hdf5(iv_path, nodes=['/ids', '/data'])
                domain.speaker_data[speaker_id] = embeds
                domain.speaker_logids[speaker_id] = logids
                speaker_map.append(speaker_id)
            except:
                raise Exception("Corrupt face model file path [%s]. Please remove and try again"
                                % os.path.dirname(iv_path))

        if len(speaker_map) > 0:
            domain.enroll_data = np.vstack(list(domain.speaker_data.values()))
            domain.speaker_map = np.empty([domain.enroll_data.shape[0], 2], dtype=object)
            cnt = 0
            for speaker_id in domain.speaker_data.keys():
                logids = domain.speaker_logids[speaker_id]
                new_cnt = len(logids)
                domain.speaker_map[cnt:cnt+new_cnt, 0] = logids
                domain.speaker_map[cnt:cnt+new_cnt, 1] = speaker_id
                cnt += new_cnt
            domain.enroll_ids, spk_ids = np.unique(domain.speaker_map[:, 1], return_inverse=True)
            domain.enroll_seg2model = np.vstack([np.arange(len(spk_ids)), spk_ids]).T

    def add_class_data(self, domain_id, image, class_id, bounding_boxes,  enrollspace, opts=None):
        domain = self.get_domains()[domain_id]
        config = self.update_opts(opts, domain)

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
        domain.resnetVGG.to(device)
        data_dir = os.path.join(enrollspace, "staging", image.id)

        # Save the enrollment model/data
        if not os.path.exists(os.path.join(data_dir, image.id)+'.iv'):
            try:
                # Expects a single face for enrollment and error if multiple or zero faces found
                vec, _ = self.extract_img_embedding(domain, image.as_PIL_Image().convert('RGB'), config, single_face=True, bbox=bounding_boxes, error_if_multi_face=True, device=device)
                utils.mkdirs(data_dir)
                idt.save_vectors_in_dir([{'iv': vec}],
                                    [image.id], data_dir, ext=".iv", with_meta=True)
            except:
                self.escape_with_error(f"Failed to extract embedding or create enrollment folder {data_dir}")

        # Return location for the purpose of datavectors (not needed for data enrollment)
        return data_dir


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

        # Consolidate embeddings for rapid speaker retrieval
        embeds = {}
        for session_id in os.listdir(final_enrollment_dir):
            if session_id != '__data__':
                iv_path = os.path.join(final_enrollment_dir, session_id, session_id + '.iv')
                try:
                    embeds[str(session_id)] = idt.read_data_in_hdf5(iv_path, nodes=['/iv', ])[0]
                except:
                    raise Exception("Corrupt speaker data file path [%s]. Please remove and try again"
                                    % os.path.dirname(iv_path))

        if not os.path.exists(os.path.join(final_enrollment_dir, '__data__')):
            utils.mkdirs(os.path.join(final_enrollment_dir, '__data__'))
        outfile = os.path.join(final_enrollment_dir, '__data__', 'data.h5')
        idt.save_consolidated_vectors(outfile, list(embeds.keys()), np.vstack(list(embeds.values())))

    def remove_class(self, domain_id, class_id, workspace):
        class_dir = self.get_enrollment_storage(domain_id, class_id)
        shutil.rmtree(class_dir)

    def remove_class_data(self, domain_id, data, class_id, enrollspace):
        removal_dir = os.path.join(enrollspace, "removals")
        utils.mkdirs(removal_dir)
        with open(os.path.join(removal_dir, data.id), "w") as f:
            f.write("please")

    def get_bounding_box_scoring_opts(self):
        """
        These options are used in the OLIVE GUI and may be configured on the commandline by passing a file to --options
        """
        region_scoring_trait_options = [
            TraitOption('threshold', "Detection threshold for face similarity", "Higher value results in less detections being output (default %0.1f)" % self.config.threshold, TraitType.CHOICE_TRAIT, "", ""),
            TraitOption('face_detection_threshold', "Detection threshold to locate faces to score", "Higher value results in less detections being output (default %0.1f)" % self.config.face_detection_threshold, TraitType.CHOICE_TRAIT, "", ""),
            TraitOption('keep_only_one_score_per_enrollee', "Keep only highest scoring face per enrollee", "Removes other faces scored by the enrollee except the maximum scoring face (default %0.1f)" % self.config.keep_only_one_score_per_enrollee, TraitType.BOOLEAN_TRAIT, "", ""),
            TraitOption('keep_only_highest_score_per_face', "Keep only highest scoring enrollee per face", "Removes scores from non-max scoring enrollee per face (default %0.1f)" % self.config.keep_only_one_score_per_enrollee, TraitType.BOOLEAN_TRAIT, "", ""),
            TraitOption('remove_overlapping_faces_from_highest_scoring_face', "Keep only highest scoring face from overlapping faces", "Removes overlapping faces from the most face-like face (default %0.1f)" % self.config.remove_overlapping_faces_from_highest_scoring_face, TraitType.BOOLEAN_TRAIT, "", ""),
            ]

        return region_scoring_trait_options

    def get_enrollment_opts(self):
        """
        These options are used in the OLIVE GUI and may be configured on the commandline by passing a file to --options
        """
        enrollment_trait_options = [
            TraitOption('face_detection_threshold', "Detection threshold to locate faces to score", "Higher value results in less detections being output (default %0.1f)" % self.config.face_detection_threshold, TraitType.CHOICE_TRAIT, "", ""),
            ]

        return enrollment_trait_options

#This line is very important! Every plugin should have one
plugin=CustomPlugin()
