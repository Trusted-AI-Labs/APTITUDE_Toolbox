from pytb.detection.bboxes.bboxes_2d_detector.bboxes_2d_detector import BBoxes2DDetector
from pytb.output.bboxes_2d import BBoxes2D

from timeit import default_timer

import cv2
import torch
import torchvision
import numpy as np
import logging

log = logging.getLogger("aptitude-toolbox")

class MRCNN(BBoxes2DDetector):

    def __init__(self, detector_parameters: dict):
        """Initializes the detectors with the given parameters.

        Args:
            detector_parameters (dict): A dictionary containing the YOLO detector parameters
        """
        super().__init__(detector_parameters)
        self.use_coco = detector_parameters["MRCNN"].get("use_coco_weights", True)
        self.gpu = detector_parameters["MRCNN"].get("GPU", False)

        log.debug("GPU set to {}.".format(self.gpu))

        log.debug("Mask-RCNN {} implementation selected.".format(self.pref_implem))

        if self.pref_implem == "torch-resnet50":
            if self.use_coco:
                self.net = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
            else:
                self.net = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
                self.net.load_state_dict(torch.load(self.model_path))
            if self.gpu:
                self.net.cuda()
            else:
                self.net.cpu()
            self.net.eval()

        else:
            assert False, "[ERROR] Unknown implementation of Mask-RCNN: {}".format(self.pref_implem)

    def detect(self, frame: np.ndarray) -> BBoxes2D:
        """Performs a Mask-RCNN inference on the given frame.

        Args:
            frame (np.ndarray): The frame to infer Mask-RCNN detections

        Returns:
            BBoxes2D: A set of 2DBBoxes of the detected objects.
        """
        if self.pref_implem == "torch-resnet50":
            frame = frame.astype('float32') / 255.0
            frame = torch.from_numpy(frame).permute(2, 0, 1)
            if self.gpu:
                frame = frame.cuda()
            output = self._detect_torch_resnet50_pretrained(frame)

        else:
            assert False, "[ERROR] Unknown implementation of Mask-RCNN: {}".format(self.pref_implem)

        return output

    def _detect_torch_resnet50_pretrained(self, org_frame) -> BBoxes2D:
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
