import numpy as np
from abc import ABC, abstractmethod

from pytb.detection.detector import Detector
from pytb.output.bboxes_2d import BBoxes2D


class BBoxes2DDetector(Detector, ABC):

    def __init__(self, detector_parameters: dict):
        """
        This class encompasses the attributes that are common to most detectors of 2D bounding boxes.
        Initiliazes the detector with the given parameters.

        Args:
            detector_parameters (dict): A dictionary containing the parameters of the desired detector.
        """
        super().__init__()

        # A detector can have multiple implementations (e.g. in different frameworks),
        # this parameter allows to choose one (required).
        self.pref_implem = detector_parameters["BBoxes2DDetector"]["pref_implem"]
        
        # A detector usually comes with its weight and a configuration file.
        # Those are the path to those files (required in some implementations).
        self.model_path = detector_parameters["BBoxes2DDetector"].get("model_path", "")
        self.config_path = detector_parameters["BBoxes2DDetector"].get("config_path", "")
        
        # The input path of the image in the detector.
        # This allows to setup the first layers of the network to match the image shape.
        self.input_width = detector_parameters["BBoxes2DDetector"].get("input_width", 416)
        self.input_height = detector_parameters["BBoxes2DDetector"].get("input_height", 416)

    @abstractmethod
    def detect(self, org_frame: np.ndarray) -> BBoxes2D:
        """Performs an inference on the given frame. 

        Args:
            org_frame (np.ndarray): The given frame to infer detections

        Returns:
            BBoxes2D: A set of 2D bounding boxes identifying detected objects of the detected objects
        """
        pass
