from pytb.tracking.bboxes.bboxes_2d_tracker.bboxes_2d_tracker import BBoxes2DTracker
from pytb.output.bboxes_2d import BBoxes2D
from pytb.output.bboxes_2d_track import BBoxes2DTrack

from .deep_sort_leonlok.tracker import Tracker
from .deep_sort_leonlok.detection import Detection
from .deep_sort_leonlok import nn_matching
from . import generate_detections as gdet

import numpy as np
from timeit import default_timer
import logging

log = logging.getLogger("aptitude-toolbox")

class DeepSORT(BBoxes2DTracker):

    def __init__(self, tracker_parameters: dict):
        """Initializes a DeepSORT tracker with the given parameters.

        Args:
            tracker_parameters (dict): A dictionary containing the related SORT's parameters
        """
        super().__init__(tracker_parameters)
        self.need_frame = True

        self.max_age = tracker_parameters["DeepSORT"].get("max_age", 30)
        self.min_hits = tracker_parameters["DeepSORT"].get("min_hits", 3)
        self.iou_thresh = tracker_parameters["DeepSORT"].get("iou_thresh", 0.7)

        self.model_path = tracker_parameters["DeepSORT"]["model_path"]
        self.max_cosine_dist = tracker_parameters["DeepSORT"].get("max_cosine_dist", 0.3)
        self.nn_budget = tracker_parameters["DeepSORT"].get("nn_budget", None)

        self.avg_det_conf = tracker_parameters["DeepSORT"].get("avg_det_conf", False)
        self.avg_det_conf_thresh = tracker_parameters["DeepSORT"].get("avg_det_conf_thresh", 0)
        self.most_common_class = tracker_parameters["DeepSORT"].get("most_common_class", False)

        self.encoder = gdet.create_box_encoder(self.model_path, batch_size=1)

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_dist, self.nn_budget)

        log.debug("DeepSORT {} implementation selected.".format(self.pref_implem))
        if self.pref_implem == "Leonlok":
            self.tracker = Tracker(metric, self.iou_thresh, self.max_age, self.min_hits, self.avg_det_conf_thresh)
        else:
            assert False, "[ERROR] Unknown implementation of DeepSORT: {}".format(self.pref_implem)

    def track(self, detection: BBoxes2D, frame=np.ndarray) -> BBoxes2DTrack:
        """Performs an inference on the given frame. 

        Args:
            detection (BBoxes2D): The detection used to infer IDs.
            frame (np.ndarray): The frame where objects have to be tracked

        Returns:
            BBoxes2DTrack: A set of 2DBBoxes detections with the tracking information added.
        """
        if self.pref_implem == "Leonlok":
            start = default_timer()
            features = self.encoder(frame, detection.bboxes)

            detections = [Detection(bbox, conf, cl, feature) for bbox, conf, cl, feature in
                          zip(detection.bboxes, detection.det_confs, detection.class_IDs, features)]

            self.tracker.predict()
            self.tracker.update(detections)
            tracking_time = default_timer() - start

            classes = []
            confidences = []
            bboxes = []
            track_IDs = []

            for track in self.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                bboxes.append(track.to_tlwh())
                track_IDs.append(track.track_id)

                if self.most_common_class:
                    classes.append(track.cls)
                else:
                    classes.append(track.det_cls)

                if self.avg_det_conf:
                    confidences.append(track.adc)
                else:
                    confidences.append(track.detection_confidence)

            return BBoxes2DTrack(detection.detection_time, np.array(bboxes).astype("int"),
                                 np.array(classes), np.array(confidences),
                                 detection.dim_width, detection.dim_height,
                                 tracking_time, np.array(track_IDs))

        else:
            assert False, "[ERROR] Unknown implementation of DeepSORT: {}".format(self.pref_implem)

    def reset_state(self, reset_id: bool = False):
        """Reset the current state of the tracker."""
        self.tracker.reset_state(reset_id)
