from pytb.detection.detector import Detector
from pytb.output.detection import Detection
import pytb.utils.transformation as tfm

from timeit import default_timer
import numpy as np
import cv2

class DetectionManager:

    def __init__(self, detector: Detector, preprocess_parameters: dict, postprocess_parameters: dict):

        # _validate_preprocess_parameters(preprocess_parameters)
        # _validate_postprocess_parameters(postprocess_parameters)
        
        self.detector = detector
        self.preprocess_parameters = preprocess_parameters
        self.postprocess_parameters = postprocess_parameters

    @staticmethod
    def _validate_preprocess_parameters(preprocess_parameters: dict):
        # TODO
        pass

    @staticmethod
    def _validate_postprocess_parameters(postprocess_parameters: dict):
        #TODO
        pass

    def detect(self, org_frame: np.ndarray) -> Detection:
        start = default_timer()
        edit_frame = tfm.pre_process(self.preprocess_parameters, org_frame)
        preproc_time = default_timer()-start

        # call the concrete method of the detector
        start = default_timer()
        detections = self.detector.detect(edit_frame)
        detections.processing_time = default_timer()-start

        detections.preprocessing_time = preproc_time

        # Post process
        start = default_timer()
        if detections.number_detections != 0:
            detections = tfm.post_process(self.postprocess_parameters, detections)
        detections.postprocessing_time = default_timer()-start

        # Display results
        # ratio = max(edit_frame.shape)
        # for b in detections.bboxes:
        #     (x, y, w, h) = b
        #     x = int(x * (ratio/detections.dim_width))
        #     w = int(w * (ratio/detections.dim_width))
        #     y = int(y * (ratio/detections.dim_height))
        #     h = int(h * (ratio/detections.dim_height))
        #     color = (255,0,0)
        #     cv2.rectangle(edit_frame, (x, y), (x + w, y + h), color, 2)
        # cv2.imshow("res", edit_frame)
        # cv2.waitKey(0)

        return detections