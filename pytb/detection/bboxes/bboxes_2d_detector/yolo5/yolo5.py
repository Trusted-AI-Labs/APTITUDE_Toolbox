from pytb.detection.bboxes.bboxes_2d_detector.bboxes_2d_detector import BBoxes2DDetector
from pytb.output.bboxes_2d import BBoxes2D

from timeit import default_timer

import torch
import numpy as np
import logging

log = logging.getLogger("aptitude-toolbox")


class YOLO5(BBoxes2DDetector):

    def __init__(self, proc_parameters: dict):
        """Initializes the detector with the given parameters.

        Args:
            proc_parameters (dict): A dictionary containing the YOLO detector parameters
        """
        super().__init__(proc_parameters)

        # The minimum confidence threshold of the detected objects if the implementation allows to provide one.
        self.conf_thresh = proc_parameters["params"].get("conf_thresh", 0)

        # The minimum non-max suppression threshold of the detected objects if the implementation allows to provide one.
        # The non-max suppression can be implemented in multiple ways, results can vary.
        self.nms_thresh = proc_parameters["params"].get("nms_thresh", 0)

        # Whether to perform the NMS algorithm across the different classes of object or separately.
        self.nms_across_classes = proc_parameters["params"].get("nms_across_classes", True)

        # Whether to use the GPU if available.
        self.gpu = proc_parameters["params"].get("GPU", False)

        log.debug("GPU set to {}" .format(self.gpu))
        log.debug("YOLOv5 {} implementation selected.".format(self.pref_implem))

        # implementation for YOLOv5. YOLOv5 is not the actual successor of YOLOv4
        if self.pref_implem == "torch-Ultralytics":
            self.net = torch.hub.load('ultralytics/yolov5:v6.0', 'custom', path=self.model_path, verbose=False)
            if self.gpu:
                self.net.cuda()
            else:
                self.net.cpu()
            self.net.conf = self.conf_thresh
            self.net.iou = self.nms_thresh
            self.net.agnostic = self.nms_across_classes

        else:
            assert False, "[ERROR] Unknown implementation of YOLO: {}".format(self.pref_implem)

    def detect(self, frame: np.array) -> BBoxes2D:
        """Performs a YOLOv5 inference on the given frame using ultralytics on PyTorch.

        Args:
            frame (np.array): The frame to infer YOLOv5 detections.

        Returns:
            BBoxes2D: A set of 2D bounding boxes identifying the detected objects.
        """
        org_frame = frame[..., ::-1]  # BGR to RGB

        start = default_timer()
        output = self.net(org_frame, size=self.input_width)
        end = default_timer()

        # Get results on CPU
        results = np.array(output.xyxy[0].cpu())

        bboxes = BBoxes2D((end - start), results[:, 0:4], results[:, 5].astype(int), results[:, 4],
                          self.input_width, self.input_height, "x1_y1_x2_y2")
        bboxes.to_xt_yt_w_h()
        return bboxes
