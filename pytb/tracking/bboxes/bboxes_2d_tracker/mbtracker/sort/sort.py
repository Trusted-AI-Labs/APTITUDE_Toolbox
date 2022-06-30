"""
Copyright (c) 2021-2022 UCLouvain, ICTEAM
Licensed under GPL-3.0 [see LICENSE for details]
Written by Jonathan Samelson (2021-2022)
"""


from pytb.tracking.bboxes.bboxes_2d_tracker.bboxes_2d_tracker import BBoxes2DTracker
from pytb.tracking.bboxes.bboxes_2d_tracker.mbtracker.sort.sort_abewley import Sort as SortAbewley
from pytb.output.bboxes_2d import BBoxes2D
from pytb.output.bboxes_2d_track import BBoxes2DTrack

import numpy as np
from timeit import default_timer
import logging

log = logging.getLogger("aptitude-toolbox")


class SORT(BBoxes2DTracker):

    def __init__(self, tracker_parameters: dict):
        """Initializes a SORT tracker with the given parameters.

        Args:
            tracker_parameters (dict): A dictionary containing the related SORT's parameters
        """
        super().__init__(tracker_parameters)

        # An object that is not tracked for max_age frame is removed from the memory
        self.max_age = tracker_parameters["SORT"].get("max_age", 10)

        # Minimum of hits to start tracking the objects
        self.min_hits = tracker_parameters["SORT"].get("min_hits", 3)

        # The minimum IOU threshold to keep the association of a previoulsy detected object 
        self.iou_thresh = tracker_parameters["SORT"].get("iou_thresh", 0.3)

        # Above 1.0, it enables a fading memory which gives less importance to the older tracks in the memory
        self.memory_fade = tracker_parameters["SORT"].get("memory_fade", 1.0)

        log.debug("SORT {} implementation selected.".format(self.pref_implem))
        if self.pref_implem == "Abewley":
            self.tracker = SortAbewley(self.max_age, self.min_hits, self.iou_thresh, self.memory_fade)
        else:
            assert False, "[ERROR] Unknown implementation of SORT: {}".format(self.pref_implem)

    def track(self, detection: BBoxes2D) -> BBoxes2DTrack:
        """Performs an inference on the given frame. 

        Args:
            detection (BBoxes2D): The detection used to infer IDs.

        Returns:
            BBoxes2DTrack: A set of 2D bounding boxes identifying detected objects with the tracking information added.
        """
        if self.pref_implem == "Abewley":
            if detection.number_objects == 0:
                dets = np.empty((0, 6))
            else:
                detection.to_x1_y1_x2_y2()
                dets = np.column_stack((detection.bboxes, detection.det_confs, detection.class_IDs))

            # Update results based on the detections of current frame
            start = default_timer()
            res = self.tracker.update(dets)
            tracking_time = default_timer() - start

            # Format results
            res_split = np.hsplit(res, np.array([4, 5, 6, 7]))
            bboxes = res_split[0]
            class_IDs = res_split[2].flatten().astype(int)
            det_confs = res_split[1].flatten()
            global_IDs = res_split[3].flatten().astype(int)

            output = BBoxes2DTrack(detection.detection_time, bboxes, class_IDs, det_confs,
                                   detection.dim_width, detection.dim_height, tracking_time, global_IDs,
                                   bboxes_format="x1_y1_x2_y2")
            output.to_xt_yt_w_h()
            return output
        
        else:
            assert False, "[ERROR] Unknown implementation of SORT: {}".format(self.pref_implem)

    def reset_state(self, reset_id: bool = False):
        """Reset the current state of the tracker."""
        self.tracker.reset_state(reset_id)

