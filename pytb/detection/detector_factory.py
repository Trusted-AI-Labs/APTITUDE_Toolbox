import logging
import pytb.utils.validator as val
from pytb.detection.detector import Detector
from pytb.detection.bboxes.bboxes_2d_detector.bboxes_2d_detector import BBoxes2DDetector

log = logging.getLogger("aptitude-toolbox")


class DetectorFactory:

    @staticmethod
    def create_detector(detector_parameters: dict) -> Detector:
        """
        Creates a detector given a dictionary of parameters 
        provided that the defined parameters are valid. 
        Otherwise, an error message is given indicating the invalid parameters.
        It first branches on the detector type (e.g. `BBoxes2DDetector`) 
        and then follow the chain to initialize the required detector.

        Args:
            detector_parameters (dict): A dictionary describing the detector to initialize.

        Returns:
            Detection: A concrete implementation of a Detector (e.g YOLO).
        """
        assert val.validate_detector_parameters(detector_parameters), \
            "[ERROR] Invalid Proc (detector) parameter(s) detected, check the above for details."
        det_type = detector_parameters["Detector"]["type"]

        if det_type == "BBoxes2DDetector":
            return DetectorFactory._bboxes_2d_detector(detector_parameters)

        elif det_type == "PoseDetector":
            return DetectorFactory._pose_detector(detector_parameters)

    @staticmethod
    def _bboxes_2d_detector(detector_parameters: dict) -> BBoxes2DDetector:
        """
        Creates a `BBoxes2DDetector` given a dictionary of parameters.
        It then branches on the model type to initialize a concrete detector implementation (e.g YOLO).
        """
        model_type = detector_parameters["BBoxes2DDetector"]["model_type"]

        log.info("Model type {} selected.".format(model_type))
        if model_type == "YOLO":
            from pytb.detection.bboxes.bboxes_2d_detector.yolo.yolo import YOLO
            return YOLO(detector_parameters)
        if model_type == "Detectron2":
            from pytb.detection.bboxes.bboxes_2d_detector.detectron2.detectron2 import Detectron2
            return Detectron2(detector_parameters)
        if model_type == "MRCNN":
            from pytb.detection.bboxes.bboxes_2d_detector.mask_rcnn.mrcnn import MRCNN
            return MRCNN(detector_parameters)
        if model_type == "FASTERRCNN":
            from pytb.detection.bboxes.bboxes_2d_detector.faster_rcnn.faster_rcnn import FASTERRCNN
            return FASTERRCNN(detector_parameters)
        if model_type == "BackgroundSubtractor":
            from pytb.detection.bboxes.bboxes_2d_detector.background_subtractor.background_subtractor \
                import BackgroundSubtractor
            return BackgroundSubtractor(detector_parameters)

    @staticmethod
    def _pose_detector(detector_parameters: dict) -> None:
        """
        TODO. No implementation of pose detector yet.
        Creates a PoseDetector given a dictionary of parameters.
        It then branches on the model type to initialize a concrete detector implementation (e.g XXX).
        """
        pass
