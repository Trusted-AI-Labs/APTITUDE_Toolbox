from timeit import default_timer

import numpy as np

import pytb.utils.transformation as tfm
from pytb.detection.detector import Detector
from pytb.output.detection import Detection


class DetectionManager:

    def __init__(self, detector: Detector, preprocess_parameters: dict, postprocess_parameters: dict):
        # _validate_preprocess_parameters(preprocess_parameters)
        # _validate_postprocess_parameters(postprocess_parameters)

        self.detector = detector
        self.preprocess_parameters = preprocess_parameters
        self.postprocess_parameters = postprocess_parameters

        self.roi = None

    @staticmethod
    def _validate_preprocess_parameters(preprocess_parameters: dict):
        # TODO
        pass

    @staticmethod
    def _validate_postprocess_parameters(postprocess_parameters: dict):
        # TODO
        pass

    def detect(self, org_frame: np.ndarray) -> Detection:
        start = default_timer()
        edit_frame, self.roi = tfm.pre_process(self.preprocess_parameters, org_frame, self.roi)
        preproc_time = default_timer() - start

        # call the concrete method of the detector
        start = default_timer()
        detection = self.detector.detect(edit_frame)
        detection.processing_time = default_timer() - start

        detection.preprocessing_time = preproc_time

        # Post process
        start = default_timer()
        if detection.number_objects != 0:
            detection = tfm.post_process(self.postprocess_parameters, detection)
        detection.postprocessing_time = default_timer() - start

        # print("--------------")
        # print(detection.preprocessing_time)
        # print(detection.processing_time)
        # print(detection.postprocessing_time)
        # print(1/(detection.processing_time+detection.processing_time+detection.postprocessing_time))
        return detection
