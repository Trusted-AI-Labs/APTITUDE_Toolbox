"""
Copyright (c) 2021-2022 UCLouvain, ICTEAM
Licensed under GPL-3.0 [see LICENSE for details]
Written by Jonathan Samelson and Benoît Gérin (2022)
"""

from pytb.detection.bboxes.bboxes_2d_detector.bboxes_2d_detector import BBoxes2DDetector
from pytb.output.bboxes_2d import BBoxes2D

from timeit import default_timer

import torch
import torchvision
import numpy as np
import logging

log = logging.getLogger("aptitude-toolbox")

class FASTERRCNN(BBoxes2DDetector):

    def __init__(self, detector_parameters: dict):
        """Initializes the detector with the given parameters.

        Args:
            detector_parameters (dict): A dictionary containing the FasterRCNN parameters.
        """
        super().__init__(detector_parameters)
        # Whether to use the default weights available on PyTorch
        self.use_coco = detector_parameters["FASTERRCNN"].get("use_coco_weights", True)
        
        # Whether to use the GPU if available.
        self.gpu = detector_parameters["FASTERRCNN"].get("GPU", False)

        log.debug("GPU set to {}.".format(self.gpu))

        log.debug("Faster-RCNN {} implementation selected.".format(self.pref_implem))

        if self.pref_implem == "torch-resnet50":
            if self.use_coco:
                self.net = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            else:
                self.net = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
                # Use weights from the provided path 
                self.net.load_state_dict(torch.load(self.model_path))
            if self.gpu:
                self.net.cuda()
            else:
                self.net.cpu()

            # Change the mode of the network to eval, for inference.
            self.net.eval()

        else:
            assert False, "[ERROR] Unknown implementation of Faster-RCNN: {}".format(self.pref_implem)

    def detect(self, frame: np.ndarray) -> BBoxes2D:
        """Performs a Faster-RCNN inference on the given frame.

        Args:
            frame (np.ndarray): The frame to infer Faster-RCNN detections

        Returns:
            BBoxes2D: A set of 2D bounding boxes identifying the detected objects.
        """
        if self.pref_implem == "torch-resnet50":
            # Obtain values between 0 and 1 instead of 0 and 255
            frame = frame.astype('float32') / 255.0

            # Permute the channels of the image
            frame = torch.from_numpy(frame).permute(2, 0, 1)
            if self.gpu:
                frame = frame.cuda()
            output = self._detect_torch_resnet50_pretrained(frame)

        else:
            assert False, "[ERROR] Unknown implementation of Faster-RCNN: {}".format(self.pref_implem)

        return output

    def _detect_torch_resnet50_pretrained(self, org_frame) -> BBoxes2D:
        """
        Performs the inference using the implementation PyTorch Resnet50.

        Args:
            org_frame (np.array): The frame to infer Faster-RCNN detections.
        """
        start = default_timer()
        with torch.no_grad():
            predictions = self.net([org_frame])
            boxes = predictions[0]['boxes'].to('cpu').numpy()
            labels = predictions[0]['labels'].to('cpu').numpy()
            scores = predictions[0]['scores'].to('cpu').numpy()
            #masks = predictions[0]['masks'].to('cpu').numpy()
        end = default_timer()

        bboxes = BBoxes2D((end - start), boxes, labels.astype(int), scores,
                          self.input_width, self.input_height, "x1_y1_x2_y2")
        bboxes.to_xt_yt_w_h()
        return bboxes
