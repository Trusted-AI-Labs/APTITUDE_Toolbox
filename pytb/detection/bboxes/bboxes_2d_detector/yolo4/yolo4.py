from pytb.detection.bboxes.bboxes_2d_detector.bboxes_2d_detector import BBoxes2DDetector
from pytb.output.bboxes_2d import BBoxes2D

from timeit import default_timer

import cv2
import numpy as np
import logging

log = logging.getLogger("aptitude-toolbox")


class YOLO4(BBoxes2DDetector):

    def __init__(self, proc_parameters: dict):
        """This class can be used for YOLO v2, v3, v4 models from Darknet.

        Initializes the detector with the given parameters.

        Args:
            proc_parameters (dict): A dictionary containing the YOLO detector parameters
        """
        super().__init__(proc_parameters)

        # The minimum confidence threshold of the detected objects if the implementation allows to provide one.
        self.conf_thresh = proc_parameters["params"].get("conf_thresh", 0)
        
        # The minimum non-max suppression threshold of the detected objects if the implementation allows to provide one.
        # The non-max suppression can be implemented in multiple ways, results can vary.
        self.nms_thresh = proc_parameters["params"].get("nms_thresh", 0)
        
        # Whether to perform the NMS algorithm across the different classes of object or separately.
        self.nms_across_classes = proc_parameters["params"].get("nms_across_classes", True)
        
        # Whether to use the GPU if available.
        self.gpu = proc_parameters["params"].get("GPU", False)
        
        # Whether to use the half precision capability of the recent GPU cards. 
        self.half_precision = proc_parameters["params"].get("half_precision", False)

        log.debug("GPU set to {} and half precision set to {}.".format(self.gpu, self.half_precision))
        log.debug("YOLOv2-3-4 {} implementation selected.".format(self.pref_implem))
        
        # Implementation for YOLOv2-3-4 from OpenCV.
        # This implementation is slightly faster than cv2-Readnet but is a bit more 'blackbox'.
        if self.pref_implem == "cv2-DetectionModel":
            self.net = cv2.dnn_DetectionModel(self.model_path, self.config_path)
            self.net.setInputSize(self.input_width, self.input_height)
            self.net.setInputScale(1.0 / 255)
            self.net.setInputSwapRB(True)
            self.net.setNmsAcrossClasses(self.nms_across_classes)
            self._setup_cv2()

        # Implementation for YOLOv2-3-4 from OpenCV.
        # This implementation is slightly slower than cv2-DetectionModel
        # but outputs a bit more details about the predictions.
        elif self.pref_implem == "cv2-ReadNet":
            self.net = cv2.dnn.readNet(self.model_path, self.config_path)
            self._setup_cv2()

        else:
            assert False, "[ERROR] Unknown implementation of YOLO: {}".format(self.pref_implem)

    def detect(self, frame: np.array) -> BBoxes2D:
        """Performs a YOLO inference on the given frame. 

        Args:
            frame (np.array): The frame to infer YOLO detections.

        Returns:
            BBoxes2D: A set of 2D bounding boxes identifying the detected objects.
        """
        if self.pref_implem == "cv2-DetectionModel":
            if frame.shape[:2] != (self.input_height, self.input_width):
                frame = cv2.resize(frame, (self.input_width, self.input_height), interpolation=cv2.INTER_AREA)
            output = self._detect_cv2_detection_model(frame)

        elif self.pref_implem == "cv2-ReadNet":
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (self.input_width, self.input_height),
                                         swapRB=True, crop=False)
            output = self._detect_cv2_read_net(blob)

        else:
            assert False, "[ERROR] Unknown implementation of YOLO: {}".format(self.pref_implem)

        return output

    def _setup_cv2(self):
        """
        Setup OpenCV framework with the required backend.
        """
        if self.gpu:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            # Half precision is for recent GPU cards that had such capability.
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

    def _detect_cv2_detection_model(self, cv2_org_frame: np.array) -> BBoxes2D:
        """Performs a YOLOv2-3-4 inference on the given frame using cv2-DetectionModel of openCV.

        Args:
            frame (np.array): The frame to infer YOLOv2-3-4 detections.

        Returns:
            BBoxes2D: A set of 2D bounding boxes identifying the detected objects.
        """
        start = default_timer()
        classes, confidences, boxes = self.net.detect(cv2_org_frame, confThreshold=self.conf_thresh,
                                                      nmsThreshold=self.nms_thresh)
        end = default_timer()

        # Format results
        if len(classes) > 0:
            classes = classes.flatten()
            confidences = confidences.flatten()

        output = BBoxes2D((end - start), np.array(boxes), np.array(classes), np.array(confidences),
                          self.input_width, self.input_height)
        return output

    def _detect_cv2_read_net(self, blob_org_frame) -> BBoxes2D:
        """Performs a YOLOv2-4 inference on the given frame using cv2-ReadNet of openCV.

        Args:
            frame (Any): The frame to infer YOLOv2-3-4 detections.

        Returns:
            BBoxes2D: A set of 2D bounding boxes identifying the detected objects.
        """
        # Detect objects
        self.net.setInput(blob_org_frame)
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        # Inference
        start = default_timer()
        outputs = self.net.forward(output_layers)
        end = default_timer()

        classes = []
        confidences = []
        boxes = []

        # Get the output of each yolo layers
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                conf = scores[scores > self.conf_thresh]
                if len(conf) != 0:
                    box = detection[:4] * np.array(
                        [self.input_width, self.input_height, self.input_width, self.input_height])
                    box -= np.array([box[2] / 2, box[3] / 2, 0, 0])  # to xt, yt, w, h
                    classes.append(scores.argmax())
                    confidences.append(np.max(conf))
                    boxes.append(box)

        return BBoxes2D((end - start), np.array(boxes), np.array(classes), np.array(confidences),
                        self.input_width, self.input_height)
