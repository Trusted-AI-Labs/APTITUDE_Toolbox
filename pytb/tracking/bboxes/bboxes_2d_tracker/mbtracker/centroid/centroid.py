"""
Copyright (c) 2021-2022 UCLouvain, ICTEAM
Licensed under GPL-3.0 [see LICENSE for details]
Written by Jonathan Samelson (2021-2022)
"""


from pytb.output.bboxes_2d import BBoxes2D
from pytb.output.bboxes_2d_track import BBoxes2DTrack
from pytb.tracking.bboxes.bboxes_2d_tracker.bboxes_2d_tracker import BBoxes2DTracker
from .centroid_rosebrock import CentroidTracker

import numpy as np
from timeit import default_timer
import logging

log = logging.getLogger("aptitude-toolbox")


class Centroid(BBoxes2DTracker):

    def __init__(self, proc_parameters: dict):
        """Initializes a Centroid tracker with the given parameters.

        Args:
            proc_parameters (dict): A dictionary containing the Centroid parameters
        """
        super().__init__(proc_parameters)

        # An object that is not tracked for max_age frame is removed from the memory
        self.max_age = proc_parameters["params"].get("max_age", 10)

        log.debug("Centroid {} implementation selected.".format(self.pref_implem))
        if self.pref_implem == "Rosebrock":
            self.tracker = CentroidTracker(maxDisappeared=self.max_age)
        else:
            assert False, "[ERROR] Unknown implementation of Centroid: {}".format(self.pref_implem)

    def track(self, detection: BBoxes2D) -> BBoxes2DTrack:
        """Performs an inference on the given frame.

        Args:
            detection (BBoxes2D): The detection used to infer IDs.

        Returns:
            BBoxes2DTrack: A set of 2D bounding boxes identifying detected objects with the tracking information added.
        """
        if self.pref_implem == "Rosebrock":
            detection.to_x1_y1_x2_y2()

            # Update the tracker with the last bounding boxes from the detection
            start = default_timer()
            objects = self.tracker.update(detection.bboxes)
            tracking_time = default_timer() - start

            # Initialize an array full of zeros whose the length is the number of objects
            global_IDs = np.zeros(detection.number_objects, dtype=np.int8)

            # Set for associated objects and array for unassociated detections to be remove. 
            obj_ID_taken = set()
            to_remove = []

            # Associate detections and centroids
            for i, bbox in enumerate(detection.bboxes):
                for objectID, centroid in objects.items():
                    if objectID not in obj_ID_taken:
                        cX = int((bbox[0] + bbox[2]) / 2.0)
                        cY = int((bbox[1] + bbox[3]) / 2.0)
                        if centroid[0] == cX and centroid[1] == cY:
                            global_IDs[i] = objectID
                            obj_ID_taken.add(objectID)
                if global_IDs[i] == 0:
                    to_remove.append(i)

            # Remove unassociated detections.
            to_remove = np.array(to_remove)
            if len(to_remove) > 0:
                detection.remove_idx(to_remove)
                global_IDs = np.delete(global_IDs, to_remove)

            output = BBoxes2DTrack(detection.detection_time, detection.bboxes, detection.class_IDs, detection.det_confs,
                                   detection.dim_width, detection.dim_height, tracking_time, global_IDs,
                                   bboxes_format="x1_y1_x2_y2")
            output.to_xt_yt_w_h()
            return output
        
        else:
            assert False, "[ERROR] Unknown implementation of Centroid: {}".format(self.pref_implem)

    def reset_state(self, reset_id: bool = False):
        """Reset the current state of the tracker."""
        self.tracker.reset_state(reset_id)
