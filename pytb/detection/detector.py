"""
Copyright (c) 2021-2022 UCLouvain, ICTEAM
Licensed under GPL-3.0 [see LICENSE for details]
Written by Jonathan Samelson (2021-2022)
"""

from abc import ABC, abstractmethod
import numpy as np

from pytb.output.detection import Detection


class Detector(ABC):

    @abstractmethod
    def detect(self, org_frame: np.array) -> Detection:
        """
        Performs an inference on the given frame. 

        Args:
            org_frame (np.array): The given frame to infer detections

        Returns:
            Detection: A set of detections of the detected objects
        """
        pass
