"""
Copyright (c) 2021-2022 UCLouvain, ICTEAM
Licensed under GPL-3.0 [see LICENSE for details]
Written by Jonathan Samelson (2021-2022)
"""

import numpy as np
from abc import ABC, abstractmethod

from pytb.detection.detector import Detector
from pytb.output.bboxes_2d import BBoxes2D


class BBoxes2DDetector(Detector, ABC):

    def __init__(self, proc_parameters: dict):
        """
        This class encompasses the attributes that are common to most detectors of 2D bounding boxes.
        Initializes the detector with the given parameters.

        Args:
            proc_parameters (dict): A dictionary containing the parameters of the desired detector.
        """
        super().__init__()

        # A detector can have multiple implementations (e.g. in different frameworks),
        # this parameter allows to choose one (required).
        self.pref_implem = proc_parameters["pref_implem"]
        
        # A detector usually comes with its weight and a configuration file.
        # Those are the path to those files (required in some implementations).
        self.model_path = proc_parameters["params"].get("model_path", "")
        self.config_path = proc_parameters["params"].get("config_path", "")
        
        # The input path of the image in the detector.
        # This allows to setup the first layers of the network to match the image shape.
        self.input_width = proc_parameters["params"].get("input_width", 416)
        self.input_height = proc_parameters["params"].get("input_height", 416)

    @abstractmethod
    def detect(self, org_frame: np.array) -> BBoxes2D:
        """Performs an inference on the given frame. 

        Args:
            org_frame (np.array): The given frame to infer detections

        Returns:
            BBoxes2D: A set of 2D bounding boxes identifying detected objects of the detected objects
        """
        pass
