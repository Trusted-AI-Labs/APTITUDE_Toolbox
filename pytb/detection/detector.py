from abc import ABC, abstractmethod
import numpy as np

from pytb.output.detection import Detection


class Detector(ABC):

    @abstractmethod
    def detect(self, org_frame: np.ndarray) -> Detection:
        """
        Performs an inference on the given frame. 

        Args:
            org_frame (np.ndarray): The given frame to infer detections

        Returns:
            Detection: A set of detections of the detected objects
        """
        pass
