"""
Copyright (c) 2021-2022 UCLouvain, ICTEAM
Licensed under GPL-3.0 [see LICENSE for details]
Written by Jonathan Samelson (2021-2022)
"""


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

        # An object that is not tracked for max_age frame is removed from the memory
        self.max_age = tracker_parameters["DeepSORT"].get("max_age", 30)
        
        # Minimum of hits to start tracking the objects
        self.min_hits = tracker_parameters["DeepSORT"].get("min_hits", 3)
        
        # The minimum IOU threshold to keep the association of a previoulsy detected object 
        self.iou_thresh = tracker_parameters["DeepSORT"].get("iou_thresh", 0.7)

        # DeepSORT requires a model weight
        self.model_path = tracker_parameters["DeepSORT"]["model_path"]
        
        # See underlying implementation
        self.max_cosine_dist = tracker_parameters["DeepSORT"].get("max_cosine_dist", 0.3)
        self.nn_budget = tracker_parameters["DeepSORT"].get("nn_budget", None)

        # Whether the average detection confidence should be evaluated to filter out detection
        self.avg_det_conf = tracker_parameters["DeepSORT"].get("avg_det_conf", False)

        # If avg_det_conf is used, the thresholds defines the average confidence 
        # under which a detection will be filtered out
        self.avg_det_conf_thresh = tracker_parameters["DeepSORT"].get("avg_det_conf_thresh", 0)

        # If true, the object class will be the most common class detected over time 
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
            BBoxes2DTrack: A set of 2D bounding boxes identifying detected objects with the tracking information added.
        """
        if self.pref_implem == "Leonlok":
            start = default_timer()

            # Find the features of each detected object
            features = self.encoder(frame, detection.bboxes)

            detections = [Detection(bbox, conf, cl, feature) for bbox, conf, cl, feature in
                          zip(detection.bboxes, detection.det_confs, detection.class_IDs, features)]

            # Associate back using the objects' features
            self.tracker.predict()
            self.tracker.update(detections)
            tracking_time = default_timer() - start

            classes = []
            confidences = []
            bboxes = []
            track_IDs = []

            for track in self.tracker.tracks:
                # If object has left the field of view or is occluded
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                bboxes.append(track.to_tlwh())
                track_IDs.append(track.track_id)

                # Depending on the parameters (see above), choose the oject class
                if self.most_common_class:
                    classes.append(track.cls)
                else:
                    classes.append(track.det_cls)

                # Depending on the parameters (see above), choose the prediction confidence
                if self.avg_det_conf:
                    confidences.append(track.adc)
                else:
                    confidences.append(track.detection_confidence)

            return BBoxes2DTrack(detection.detection_time, np.array(bboxes),
                                 np.array(classes), np.array(confidences),
                                 detection.dim_width, detection.dim_height,
                                 tracking_time, np.array(track_IDs))

        else:
            assert False, "[ERROR] Unknown implementation of DeepSORT: {}".format(self.pref_implem)

    def reset_state(self, reset_id: bool = False):
        """Reset the current state of the tracker."""
        self.tracker.reset_state(reset_id)
