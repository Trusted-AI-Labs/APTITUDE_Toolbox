from pytb.detection.bboxes.bboxes_2d_detector.yolo.yolo import YOLO

class DetectorFactory:

    @staticmethod
    def create_detector(detector_parameters):
        # _validate_parameters(detector_parameters)
        det_type = detector_parameters["Detector"]["type"]
        
        if det_type == "BBoxes2DDetector":
            return DetectorFactory._bboxes_2d_detector(detector_parameters)
        
        elif det_type == "PoseDetector":
            return DetectorFactory._pose_detector(detector_parameters)


    @staticmethod
    def _bboxes_2d_detector(detector_parameters):
        model_type = detector_parameters["BBoxes2DDetector"]["model"]

        if model_type == "YOLO":
            return YOLO(detector_parameters)

    @staticmethod
    def _pose_detector(detector_parameters):
        pass

    @staticmethod
    def _validate_parameters(detector_parameters):
        """Check compatibility between provided parameters.

        Args:
            detector_parameters (dict): the dictionary containing detector parameters.
        """
        #TODO validate parameters from create_detector
        pass

