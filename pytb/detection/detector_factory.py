"""
Copyright (c) 2021-2022 UCLouvain, ICTEAM
Licensed under GPL-3.0 [see LICENSE for details]
Written by Jonathan Samelson (2021-2022), Arthur Pisvin (2023)
"""

import logging
import pytb.utils.validator as val
from pytb.detection.detector import Detector
from pytb.detection.bboxes.bboxes_2d_detector.bboxes_2d_detector import BBoxes2DDetector

log = logging.getLogger("aptitude-toolbox")


class DetectorFactory:

    @staticmethod
    def create_detector(proc_parameters: dict) -> Detector:
        """
        Creates a detector given a dictionary of parameters 
        provided that the defined parameters are valid. 
        Otherwise, an error message is given indicating the invalid parameters.
        Based on a model type and an implementation, it initializes the appropriate detector with the given parameters.

        Args:
            proc_parameters (dict): A dictionary describing the detector to initialize.

        Returns:
            Detection: A concrete implementation of a Detector (e.g YOLO).
        """
        assert val.validate_detector_parameters(proc_parameters), \
            "[ERROR] Invalid Proc (detector) parameter(s) detected, check the above for details."
        output_type = proc_parameters["output_type"]

        if output_type == "bboxes2D":
            return DetectorFactory._bboxes_2d_detector(proc_parameters)

        elif output_type == "pose":
            return DetectorFactory._pose_detector(proc_parameters)

    @staticmethod
    def _bboxes_2d_detector(proc_parameters: dict) -> BBoxes2DDetector:
        """
        Creates a `BBoxes2DDetector` given a dictionary of parameters.
        It then branches on the model type to initialize a concrete detector implementation (e.g YOLO).
        """
        model_type = proc_parameters["model_type"]

        log.info("Model type {} selected.".format(model_type))
        if model_type == "YOLO4":
            from pytb.detection.bboxes.bboxes_2d_detector.yolo4.yolo4 import YOLO4
            return YOLO4(proc_parameters)
        if model_type == "YOLO5":
            from pytb.detection.bboxes.bboxes_2d_detector.yolo5.yolo5 import YOLO5
            return YOLO5(proc_parameters)
        if model_type == "YOLO8":
            from pytb.detection.bboxes.bboxes_2d_detector.yolo8.yolo8 import YOLO8
            return YOLO8(proc_parameters)
        if model_type == "Detectron2":
            from pytb.detection.bboxes.bboxes_2d_detector.detectron2.detectron2 import Detectron2
            return Detectron2(proc_parameters)
        if model_type == "MaskRCNN":
            from pytb.detection.bboxes.bboxes_2d_detector.mask_rcnn.mask_rcnn import MaskRCNN
            return MaskRCNN(proc_parameters)
        if model_type == "FasterRCNN":
            from pytb.detection.bboxes.bboxes_2d_detector.faster_rcnn.faster_rcnn import FasterRCNN
            return FasterRCNN(proc_parameters)
        if model_type == "BackgroundSubtractor":
            from pytb.detection.bboxes.bboxes_2d_detector.background_subtractor.background_subtractor \
                import BackgroundSubtractor
            return BackgroundSubtractor(proc_parameters)

    @staticmethod
    def _pose_detector(proc_parameters: dict) -> None:
        """
        TODO. No implementation of pose detector yet.
        Creates a PoseDetector given a dictionary of parameters.
        It then branches on the model type to initialize a concrete detector implementation (e.g XXX).
        """
        pass
