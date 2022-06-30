"""
Simple IOU based tracker with Kalman filter.

Copyright (c) 2017 TU Berlin, Communication Systems Group
Licensed under The MIT License [see LICENSE for details]
Written by Erik Bochinski

See https://github.com/bochinski/iou-tracker for more information

Copyright (c) 2021 UCLouvain, ICTEAM
Licensed under GPL-3.0 [see LICENSE for details]
Adapted for online tracking by Jonathan Samelson (2021)
"""

from pytb.tracking.bboxes.bboxes_2d_tracker.bboxes_2d_tracker import BBoxes2DTracker
from pytb.tracking.bboxes.bboxes_2d_tracker.mbtracker.iou.kiou import KIOU
from pytb.tracking.bboxes.bboxes_2d_tracker.mbtracker.iou.simple_iou import SimpleIOU
from pytb.output.bboxes_2d import BBoxes2D
from pytb.output.bboxes_2d_track import BBoxes2DTrack
from timeit import default_timer
import numpy as np
import logging

log = logging.getLogger("aptitude-toolbox")

class IOU(BBoxes2DTracker):

    def __init__(self, tracker_parameters: dict):
        """Initializes a IOU tracker with the given parameters.

        Args:
            tracker_parameters (dict): A dictionary containing the related SORT's parameters
        """
        super().__init__(tracker_parameters)
        # Minimum of hits to start tracking the objects
        self.min_hits = tracker_parameters["IOU"].get("min_hits", 3)

        # The minimum IOU threshold to keep the association of a previoulsy detected object
        self.iou_thresh = tracker_parameters["IOU"].get("iou_thresh", 0.3)

        log.debug("IOU {} implementation selected.".format(self.pref_implem))
        if self.pref_implem == "SimpleIOU":
            self.tracker = SimpleIOU(self.min_hits, self.iou_thresh)

        elif self.pref_implem == "KIOU":
            # An object that is not tracked for max_age frame is removed from the memory
            self.max_age = tracker_parameters["IOU"].get("max_age", 10)
            self.tracker = KIOU(self.min_hits, self.iou_thresh, self.max_age)

        else:
            assert False, "[ERROR] Unknown implementation of IOU: {}".format(self.pref_implem)

    def track(self, detection: BBoxes2D) -> BBoxes2DTrack:
        """Performs an inference on the given frame.

        Args:
            detection (BBoxes2D): The detection used to infer IDs.

        Returns:
            BBoxes2DTrack: A set of 2D bounding boxes identifying detected objects with the tracking information added.
        """
        # Format the detections before giving them to the trackers
        detection.to_x1_y1_x2_y2()
        dets = []
        for i in range(detection.number_objects):
            bb = detection.bboxes[i]
            if self.pref_implem == "SimpleIOU":
                det = {'bbox': detection.bboxes[i], 'score': detection.det_confs[i], 'class': detection.class_IDs[i]}
                dets.append(det)
            if self.pref_implem == "KIOU":
                det = {'bbox': bb, 'score': detection.det_confs[i], 'class': detection.class_IDs[i],
                       'centroid': [0.5*(bb[0] + bb[2]), 0.5*(bb[1] + bb[3])]}
                dets.append(det)

        start = default_timer()
        res = self.tracker.update(dets)
        tracking_time = default_timer() - start

        # Get results depending on the tracker implementation
        # -1 is for the last step of the tracker
        if self.pref_implem == "SimpleIOU":
            bboxes = [res['bboxes'][-1] for res in res]
            det_confs = [res['scores'][-1] for res in res]
            class_IDs = [res['classes'][-1] for res in res]
            global_IDs = [res['id'] for res in res]

        elif self.pref_implem == "KIOU":
            bboxes = [res[-1]['bbox'] for res in res]
            det_confs = [res[-1]['score'] for res in res]
            class_IDs = [res[-1]['class'] for res in res]
            global_IDs = [res[0]['id'] for res in res]

        else:
            assert False, "[ERROR] Unknown implementation of IOU: {}".format(self.pref_implem)

        # TODO add avg_det_conf and most_common_class options

        output = BBoxes2DTrack(detection.detection_time, np.array(bboxes), np.array(class_IDs), np.array(det_confs),
                               detection.dim_width, detection.dim_height, tracking_time, np.array(global_IDs),
                               bboxes_format="x1_y1_x2_y2")
        output.to_xt_yt_w_h()
        return output

    def reset_state(self, reset_id: bool = False):
        """Reset the current state of the tracker."""
        self.tracker.reset_state(reset_id)

