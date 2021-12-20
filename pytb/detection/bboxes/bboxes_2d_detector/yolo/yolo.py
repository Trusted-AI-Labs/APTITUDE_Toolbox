from pytb.detection.bboxes.bboxes_2d_detector.bboxes_2d_detector import BBoxes2DDetector
from pytb.output.bboxes_2d import BBoxes2D

from timeit import default_timer
import cv2
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
        self.conf_thresh = detector_parameters["YOLO"].get("conf_thresh", 0)

        self.ocv_gpu = detector_parameters["OpenCV"].get("GPU", False)
        self.ocv_half_precision = detector_parameters["OpenCV"].get("half_precision", False)
        log.debug("OpenCV selected with GPU set to {} and half precision set to {}."
                  .format(self.ocv_gpu, self.ocv_half_precision))

        log.debug("YOLO {} implementation selected.".format(self.pref_implem))
        if self.pref_implem == "cv2-DetectionModel":
            self.nms_thresh = detector_parameters["YOLO"].get("nms_thresh", 0)
            self.nms_across_classes = detector_parameters["YOLO"].get("nms_across_classes", True)

            self.net = cv2.dnn_DetectionModel(self.model_path, self.config_path)
            self.net.setInputSize(self.input_width, self.input_height)
            self.net.setInputScale(1.0 / 255)
            self.net.setInputSwapRB(True)
            self.net.setNmsAcrossClasses(self.nms_across_classes)

        elif self.pref_implem == "cv2-ReadNet":
            self.net = cv2.dnn.readNet(self.model_path, self.config_path)
        else:
            assert False, "[ERROR] Unknown implementation of YOLO: {}".format(self.pref_implem)

        if self.ocv_gpu:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            if self.ocv_half_precision:
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
                log.debug("OpenCV with DNN_BACKEND_CUDA target CUDAFP16.")
            else:
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                log.debug("OpenCV with DNN_BACKEND_CUDA target CUDA.")
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            log.debug("OpenCV with DNN_BACKEND_OPENCV and target CPU.")

    def detect(self, frame: np.ndarray) -> BBoxes2D:
        """Performs a YOLO inference on the given frame. 

        Args:
            frame (np.ndarray): The frame to infer YOLO detections

        Returns:
            BBoxes2D: A set of 2DBBoxes of the detected objects.
        """
        output = None
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

    def _detect_cv2_detection_model(self, cv2_org_frame: np.ndarray) -> BBoxes2D:
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
        # detect objects
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
                    boxes.append(box.astype("int"))

        return BBoxes2D((end - start), np.array(boxes), np.array(classes), np.array(confidences),
                        self.input_width, self.input_height)
