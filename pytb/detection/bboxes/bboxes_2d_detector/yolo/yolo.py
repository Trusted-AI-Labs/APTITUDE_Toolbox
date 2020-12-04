from pytb.detection.bboxes.bboxes_2d_detector.bboxes_2d_detector import BBoxes2DDetector

import cv2
import numpy as np

class YOLO(BBoxes2DDetector):

    def __init__(self, detector_parameters):
        """Initiliazes the detectors with the given parameters.

        Args:
            detector_parameters (dict): A dictionary containing the YOLO detector parameters
        """
        super().__init__(detector_parameters)
        self.conf_thresh = detector_parameters["YOLO"]["conf_thresh"]
        self.nms_thresh = detector_parameters["YOLO"]["nms_thresh"]

        if self.pref_implem == "cv2-DetectionModel":
            self.net = cv2.dnn_DetectionModel(self.model_path, self.config_path)
            self.net.setInputSize(self.input_width, self.input_height)
            self.net.setInputScale(1.0 / 255)
            self.net.setInputSwapRB(True)

        elif self.pref_implem == "cv2-ReadNet":
            self.net = cv2.dnn.readNet(self.model_path, self.config_path)

        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def detect(self, org_frame):
        """Performs a YOLO inference on the given frame. 
        Returns a set of 2DBBox detections of the detected objects.

        Args:
            org_frame (PIL Image): The frame in PIL format to infer YOLO detections
        """
        res = None 
        
        cv2_org_frame = cv2.cvtColor(np.array(org_frame), cv2.COLOR_RGB2BGR)
        if self.pref_implem == "cv2-DetectionModel":
            res = self._detect_cv2_detection_model(cv2_org_frame)

        elif self.pref_implem == "cv2-ReadNet":
            resized = cv2.resize(cv2_org_frame, (self.input_width, self.input_height),  interpolation=cv2.INTER_AREA)
            blob = cv2.dnn.blobFromImage(resized, 1/255.0, (self.input_width, self.input_height), swapRB=True, crop=False)
            res = self._detect_cv2_read_net(blob)

        return res

    def _detect_cv2_detection_model(self, cv2_org_frame):
        classes, confidences, boxes = self.net.detect(cv2_org_frame, confThreshold=self.conf_thresh, nmsThreshold=self.nms_thresh)

        # Format results
        if len(classes) > 0: 
            classes = classes.flatten()
        if len(confidences) > 0: 
            confidences = confidences.flatten()

        return classes, confidences, boxes

    def _detect_cv2_read_net(self, blob_org_frame):
        # detect objects
        self.net.setInput(blob_org_frame)
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        # Inference
        outputs = self.net.forward(output_layers)
        classes = []
        confidences = []
        boxes = []

        # Get the output of each yolo layers
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.conf_thresh:
                    w = int(detection[2] * self.input_width)
                    h = int(detection[3] * self.input_height)
                    x = int(detection[0] * self.input_width - w/2)
                    y = int(detection[1] * self.input_height - h/2)
                    classes.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        # TODO Move to post-process
        # Remove overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_thresh, self.nms_thresh)

        _bounding_boxes = []
        _classes = []
        _confidences = []
        for i in indices:
            i = i[0]
            _bounding_boxes.append(boxes[i])
            _classes.append(classes[i])
            _confidences.append(confidences[i])

        return _classes, _confidences, _bounding_boxes





