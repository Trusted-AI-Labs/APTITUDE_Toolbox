"""
Copyright (c) 2021-2022 UCLouvain, ICTEAM
Licensed under GPL-3.0 [see LICENSE for details]
Written by Jonathan Samelson (2021-2022)
"""

from pytb.detection.bboxes.bboxes_2d_detector.bboxes_2d_detector import BBoxes2DDetector

import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from timeit import default_timer

from pytb.output.bboxes_2d import BBoxes2D
import logging

log = logging.getLogger("aptitude-toolbox")


class Detectron2(BBoxes2DDetector):

    def __init__(self, proc_parameters: dict):
        """Initializes the detector with the given parameters.

        Args:
            proc_parameters (dict): A dictionary containing the Detectron2 detector parameters
        """
        super().__init__(proc_parameters)
        
        # The minimum confidence threshold of the detected objects if the implementation allows to provide one.
        self.conf_thresh = proc_parameters["params"].get("conf_thresh", 0)

        # The minimum non-max suppression threshold of the detected objects if the implementation allows to provide one.
        # The non-max suppression can be implemented in multiple ways, results can vary.
        self.nms_thresh = proc_parameters["params"].get("nms_thresh", 0)

        # Whether to use the GPU if available.
        self.gpu = proc_parameters["params"].get("GPU", False)

        log.debug("Detectron2 {} implementation selected.".format(self.pref_implem))
        if self.pref_implem == "Default":
            cfg = get_cfg()
            cfg.merge_from_file(self.config_path)
            cfg.MODEL.WEIGHTS = self.model_path
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.conf_thresh
            cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = self.nms_thresh
            cfg.INPUT.FORMAT = "BGR"
            if self.gpu:
                cfg.MODEL.DEVICE = "cuda"
                log.debug("Device CUDA selected.")
            else:
                cfg.MODEL.DEVICE = "cpu"
                log.debug("Device CPU selected.")

            # Use the DefaultPredictor, which is the most generic
            self.predictor = DefaultPredictor(cfg)
        else:
            assert False, "[ERROR] Unknown implementation of Detectron2: {}".format(self.pref_implem)

    def detect(self, org_frame: np.array) -> BBoxes2D:
        """Performs a Detectron2 inference on the given frame.

        Args:
            frame (np): The frame to infer Detectron2 detections

        Returns:
            BBoxes2D: A set of 2D bounding boxes identifying the detected objects.
        """
        start = default_timer()
        if self.pref_implem == "Default":
            detections = self.predictor(org_frame)
        else:
            assert False, "[ERROR] Unknown implementation of Detectron2: {}".format(self.pref_implem)
        end = default_timer()

        # Transfer the results to the CPU
        bboxes = detections["instances"].pred_boxes.tensor.cpu().detach().numpy()
        classes = detections["instances"].pred_classes.cpu().detach().numpy()
        confs = detections["instances"].scores.cpu().detach().numpy()

        # Seemingly, Detectron2 uses the original image dimension for inference,
        # no specific dimensions are required
        output = BBoxes2D(end-start, bboxes, classes, confs, org_frame.shape[1], org_frame.shape[0],
                          bboxes_format="x1_y1_x2_y2")
        output.to_xt_yt_w_h()
        return output
