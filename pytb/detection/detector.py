from abc import ABC, abstractmethod

class Detector(ABC):

    @abstractmethod
    def detect(self, org_frame):
        """
        Performs an inference on the given frame. 
        Returns a set of detections.
        :param org_frame: The given frame to infer detections
        :return: An object of the class Detection with the inference result
        """
        pass
