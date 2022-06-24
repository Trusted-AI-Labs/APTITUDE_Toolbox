from timeit import default_timer
import logging
import numpy as np

import pytb.utils.transformation as tfm
import pytb.utils.validator as val
from pytb.detection.detector import Detector
from pytb.output.detection import Detection

log = logging.getLogger("aptitude-toolbox")


class DetectionManager:

    def __init__(self, detector: Detector, preprocess_parameters: dict, postprocess_parameters: dict):
        """
        This class is used to handle all the steps of the detection in one single `detect` method.
        Specifically, it applies the preprocess parameters, infer the detections 
        and filters out the results using the postprocess parameters.
        This constructor initialize a DetectionManager provided that the preprocess and post process
        parameters are valid. Otherwise, an error message is given indicating the invalid parameter(s).


        Args:
            detector (Detector): A concrete implementation of a Detector initialized using the `detector_factory`.
            preprocess_parameters (dict): A dictionary containing the image transformation before proceeding to the detection.
            postprocess_parameters (dict): A dictionary containing the thresholds to filter out the results of the detection. 

        Returns:
            Detection: An initalized DetectionManager that can be called on each frame using `detect` method.
        """
        assert val.validate_preprocess_parameters(preprocess_parameters), \
            "[ERROR] Invalid Preproc parameter(s) detected, check any error reported above for details."
        assert val.validate_postprocess_parameters(postprocess_parameters), \
            "[ERROR] Invalid Postproc parameter(s) detected, check any error reported above for details."

        self.detector = detector
        self.preprocess_parameters = preprocess_parameters
        self.postprocess_parameters = postprocess_parameters

        # Those two attributes are used to keep the value of the ROI to be applied on an image sequence.
        # to avoid repeated readings.
        self.preproc_roi = None
        self.postproc_roi = None

        # TODO define a method to handle the proper update of the parameters

    def detect(self, org_frame: np.ndarray) -> Detection:
        """
        Proceeds to the detection on the given frame.
        It first transforms the image using the preprocess parameters of this instance of `DetectionManager`.
        Then, it infers the results of the detection (see the concrete implementation of the provided detector). 
        Finally, it filters out the result using the postprocess parameters of this instance of `DetectionManager`.
        """
        start = default_timer()
        edit_frame, self.preproc_roi, border_px, _ = tfm.pre_process(self.preprocess_parameters,
                                                                     org_frame, self.preproc_roi)
        preproc_time = default_timer() - start
        log.debug("Preprocessing done.")

        if border_px is not None:
            self.postprocess_parameters["borders_detection"] = border_px

        # call the concrete method of the detector
        start = default_timer()
        detection = self.detector.detect(edit_frame)
        detection.processing_time = default_timer() - start

        detection.preprocessing_time = preproc_time
        log.debug("Actual detection done.")

        # Post process
        start = default_timer()
        if detection.number_objects != 0:
            detection, self.postproc_roi = tfm.post_process(self.postprocess_parameters, detection, self.postproc_roi)
        detection.postprocessing_time = default_timer() - start
        log.debug("Postprocessing done.")

        return detection
