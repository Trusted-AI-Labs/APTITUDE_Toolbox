from pytb.output.bboxes_2d import BBoxes2D
from pytb.output.bboxes_2d_track import BBoxes2DTrack
from pytb.tracking.bboxes.bboxes_2d_tracker.bboxes_2d_tracker import BBoxes2DTracker
from .centroid_rosebrock import CentroidTracker

import numpy as np
from timeit import default_timer
import logging

log = logging.getLogger("aptitude-toolbox")


class Centroid(BBoxes2DTracker):

    def __init__(self, tracker_parameters: dict):
        """Initializes a Centroid tracker with the given parameters.

        Args:
            tracker_parameters (dict): A dictionary containing the related Centroid's parameters
        """
        super().__init__(tracker_parameters)

        self.max_age = tracker_parameters["Centroid"].get("max_age", 10)
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
            BBoxes2DTrack: A set of 2DBBoxes detections with the tracking information added.
        """
        if self.pref_implem == "Rosebrock":
            detection.to_x1_y1_x2_y2()

            start = default_timer()
            objects = self.tracker.update(detection.bboxes)
            tracking_time = default_timer() - start

            global_IDs = np.zeros(detection.number_objects, dtype=np.int8)
            obj_ID_taken = set()
            to_remove = []

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
