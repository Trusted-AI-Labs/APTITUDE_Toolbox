import numpy as np
from abc import ABC, abstractmethod

from pytb.detection.detector import Detector
from pytb.output.bboxes_2d import BBoxes_2D

class BBoxes2DDetector(Detector, ABC):

    def __init__(self, detector_parameters: dict):
        """Initiliazes the detectors with the given parameters.

        Args:
            detector_parameters (dict): A dictionary containing the related detector's parameters
        """
        super().__init__()
        self.pref_implem = detector_parameters["BBoxes2DDetector"]["pref_implem"]
        self.model_path = detector_parameters["BBoxes2DDetector"]["model_path"]
        self.config_path = detector_parameters["BBoxes2DDetector"]["config_path"]
        self.input_width = detector_parameters["BBoxes2DDetector"]["input_width"]
        self.input_height = detector_parameters["BBoxes2DDetector"]["input_height"]

    @abstractmethod
    def detect(self, org_frame: np.ndarray) -> BBoxes_2D:
        """Performs an inference on the given frame. 

        Args:
            org_frame (np.ndarray): The given frame to infer detections

        Returns:
            BBoxes_2D:A set of 2DBBoxes detections of the detected objects
        """
        pass