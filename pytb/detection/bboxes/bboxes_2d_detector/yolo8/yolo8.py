from pytb.detection.bboxes.bboxes_2d_detector.bboxes_2d_detector import BBoxes2DDetector
from pytb.output.bboxes_2d import BBoxes2D

from timeit import default_timer
from ultralytics import YOLO
from ultralytics.nn.autobackend import AutoBackend
import numpy as np
import logging

log = logging.getLogger("aptitude-toolbox")


class YOLO8(BBoxes2DDetector):

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
        self.gpu =  proc_parameters["params"].get("GPU", False)
        self.device = 0 if self.gpu else "cpu"

        log.debug("GPU set to {}" .format(proc_parameters["params"].get("GPU", False)))
        log.debug("YOLOv8 {} implementation selected.".format(self.pref_implem))


        # implementation for YOLOv8.
        if self.pref_implem == "torch-Ultralytics":
            self.net = YOLO(self.model_path)
            if self.gpu:
                self.net.to("cuda")
            else:
                self.net.to("cpu")

        else:
            assert False, "[ERROR] Unknown implementation of YOLO: {}".format(self.pref_implem)


        

    def detect(self, frame: np.array) -> BBoxes2D:
        """Performs a YOLOv8 inference on the given frame using ultralytics on PyTorch.

        Args:
            frame (np.array): The frame to infer YOLOv8 detections.

        Returns:
            BBoxes2D: A set of 2D bounding boxes identifying the detected objects.
        """
        org_frame = frame[..., ::-1]  # BGR to RGB
        start = default_timer()
        output = self.net.predict(org_frame, verbose=False, conf = self.conf_thresh, iou= self.nms_thresh, agnostic_nms= self.nms_across_classes, device=self.device)
        end = default_timer()
        # Get results on CPU

        results = output[0].boxes.cpu().boxes.numpy()

        bboxes = BBoxes2D((end - start), results[:, 0:4], results[:,5].astype(int), results[:,4],
                          self.input_width, self.input_height, "x1_y1_x2_y2")
        bboxes.to_xt_yt_w_h()
        return bboxes
