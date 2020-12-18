from pytb.detection.bboxes.bboxes_2d_detector.yolo.yolo import YOLO
from pytb.detection.detector import Detector
from pytb.detection.bboxes.bboxes_2d_detector.bboxes_2d_detector import BBoxes2DDetector

class DetectorFactory:

    @staticmethod
    def create_detector(detector_parameters: dict) -> Detector:
        # _validate_parameters(detector_parameters)
        det_type = detector_parameters["Detector"]["type"]
        
        if det_type == "BBoxes2DDetector":
            return DetectorFactory._bboxes_2d_detector(detector_parameters)
        
        elif det_type == "PoseDetector":
            return DetectorFactory._pose_detector(detector_parameters)


    @staticmethod
    def _bboxes_2d_detector(detector_parameters: dict) -> BBoxes2DDetector:
        model_type = detector_parameters["BBoxes2DDetector"]["model"]

        if model_type == "YOLO":
            return YOLO(detector_parameters)

    @staticmethod
    def _pose_detector(detector_parameters: dict) -> None:
        pass

    @staticmethod
    def _validate_parameters(detector_parameters: dict) -> bool:
        """Check compatibility between provided parameters.

        Args:
            detector_parameters (dict): the dictionary containing detector parameters.

        Returns:
            bool: whether it is a valid configuration
        """
        #TODO validate parameters from create_detector
        pass

