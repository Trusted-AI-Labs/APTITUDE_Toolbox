from abc import ABC, abstractmethod

class BBoxes2DDetector(ABC):

    def __init__(self, detector_parameters):
        """
        Initiliazes the detectors with the given parameters.
        :param detector_parameters: A dictionary containing the related detector's parameters
        :return: boolean
        """
        super().__init__()
        self.pref_implem = detector_parameters["BBoxes2DDetector"]["pref_implem"]
        self.model_path = detector_parameters["BBoxes2DDetector"]["model_path"]
        self.config_path = detector_parameters["BBoxes2DDetector"]["config_path"]
        self.input_width = detector_parameters["BBoxes2DDetector"]["input_width"]
        self.input_height = detector_parameters["BBoxes2DDetector"]["input_height"]

    # TODO This function could be part of the parent class "Detector"
    @abstractmethod
    def detect(self, org_frame):
        """
        Performs an inference on the given frame. 
        Returns a set of 2DBox detections of the detected objects.
        :param org_frame: The given frame to infer detections
        :return: An object of the class 2DBoxes with the inference result
        """
        pass