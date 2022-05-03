from pytb.detection.bboxes.bboxes_2d_detector.bboxes_2d_detector import BBoxes2DDetector
from pytb.output.bboxes_2d import BBoxes2D

from timeit import default_timer

import cv2
import torch
import torchvision
import numpy as np
import logging

log = logging.getLogger("aptitude-toolbox")

class YOLO(BBoxes2DDetector):

    def __init__(self, detector_parameters: dict):
        """Initializes the detectors with the given parameters.

        Args:
            detector_parameters (dict): A dictionary containing the YOLO detector parameters
        """
        super().__init__(detector_parameters)
        self.conf_thresh = detector_parameters["MRCNN"].get("conf_thresh", 0)
        self.nms_thresh = detector_parameters["MRCNN"].get("nms_thresh", 0)
        self.nms_across_classes = detector_parameters["MRCNN"].get("nms_across_classes", True)
        self.gpu = detector_parameters["MRCNN"].get("GPU", False)
        self.half_precision = detector_parameters["MRCNN"].get("half_precision", False)

        log.debug("GPU set to {} and half precision set to {}."
                  .format(self.gpu, self.half_precision))

        log.debug("Mask-RCNN {} implementation selected.".format(self.pref_implem))

        if self.pref_implem == "torch-resnet50_pretrained":
            self.net = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
            if self.gpu:
                self.net.cuda()
            else:
                self.net.cpu()
            self.net.conf = self.conf_thresh
            self.net.iou = self.nms_thresh
            self.net.agnostic = self.nms_across_classes

        else:
            assert False, "[ERROR] Unknown implementation of Mask-RCNN: {}".format(self.pref_implem)

    def detect(self, frame: np.ndarray) -> BBoxes2D:
        """Performs a Mask-RCNN inference on the given frame.

        Args:
            frame (np.ndarray): The frame to infer Mask-RCNN detections

        Returns:
            BBoxes2D: A set of 2DBBoxes of the detected objects.
        """
        if self.pref_implem == self.pref_implem == "torch-resnet50_pretrained":
            output = self._detect_torch_resnet50_pretrained(frame)

        else:
            assert False, "[ERROR] Unknown implementation of Mask-RCNN: {}".format(self.pref_implem)

        return output

    def _setup_cv2(self):
        if self.gpu:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            if self.half_precision:
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
                log.debug("OpenCV with DNN_BACKEND_CUDA target CUDAFP16.")
            else:
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                log.debug("OpenCV with DNN_BACKEND_CUDA target CUDA.")
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            log.debug("OpenCV with DNN_BACKEND_OPENCV and target CPU.")

    def _detect_torch_resnet50_pretrained(self, org_frame) -> BBoxes2D:
        start = default_timer()
        with torch.no_grad():
            predictions = self.net([org_frame])
            boxes = predictions[0]['boxes'].to('cpu')
            labels = predictions[0]['labels'].to('cpu')
            scores = predictions[0]['scores'].to('cpu')
        end = default_timer()

        #results = np.array(output.xyxy[0].cpu())

        bboxes = BBoxes2D((end - start), results[:, 0:4], results[:, 5].astype(int), results[:, 4],
                          self.input_width, self.input_height, "x1_y1_x2_y2")
        bboxes.to_xt_yt_w_h()
        return bboxes
