from detectron2.structures import BoxMode

from pytb.detection.bboxes.bboxes_2d_detector.bboxes_2d_detector import BBoxes2DDetector

import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from timeit import default_timer

from pytb.output.bboxes_2d import BBoxes2D


class Detectron2(BBoxes2DDetector):

    def __init__(self, detector_parameters: dict):
        """Initializes the detectors with the given parameters.

        Args:
            detector_parameters (dict): A dictionary containing the Detectron2 detector parameters
        """
        super().__init__(detector_parameters)
        self.conf_thresh = detector_parameters["Detectron2"].get("conf_thresh", 0)
        self.nms_thresh = detector_parameters["Detectron2"].get("nms_thresh", 0)
        self.gpu = detector_parameters["Detectron2"].get("GPU", False)

        if self.pref_implem == "Default":
            cfg = get_cfg()
            cfg.merge_from_file(self.config_path)
            cfg.MODEL.WEIGHTS = self.model_path
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.conf_thresh
            cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = self.nms_thresh
            cfg.INPUT.FORMAT = "BGR"
            if self.gpu:
                cfg.DEVICE = "cuda"
            else:  # It may have no effect if PyTorch is compiled with CUDA
                cfg.DEVICE = "cpu"

            self.predictor = DefaultPredictor(cfg)

    def detect(self, org_frame: np.ndarray) -> BBoxes2D:
        start = default_timer()
        detections = self.predictor(org_frame)
        end = default_timer()

        bboxes = detections["instances"].pred_boxes.tensor.cpu().detach().numpy()
        classes = detections["instances"].pred_classes.cpu().detach().numpy()
        confs = detections["instances"].scores.cpu().detach().numpy()

        # Seemingly, Detectron2 uses the original image dimension for inference,
        # no specific dimensions are required
        output = BBoxes2D(end-start, bboxes, classes, confs, org_frame.shape[1], org_frame.shape[0],
                          bboxes_format="x1_y1_x2_y2")
        output.to_xt_yt_w_h()
        return output
