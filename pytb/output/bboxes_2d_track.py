from pytb.output.bboxes_2d import BBoxes2D

import numpy as np

class BBoxes2DTrack(BBoxes2D):

    def __init__(self, detection_time: float, 
                bboxes: np.ndarray, class_IDs: np.ndarray, det_confs: np.ndarray, dim_width: int, dim_height: int,
                tracking_time:float, global_IDs: np.ndarray, track_confs: np.ndarray = None):
        super().__init__(detection_time, bboxes, class_IDs, det_confs, dim_width, dim_height)

        self.tracking_time = tracking_time

        self.global_IDs = global_IDs
        self.track_confs = track_confs

    def __str__(self):
        s = super().__str__()
        s += "\n---------------------------------"
        s += "\n\tGlobal IDs: " + str(self.global_IDs)
        s += "\n\tTrack confidences: " + str(self.track_confs)
        s += "\n\tTracking time: " + str(self.tracking_time)
        return s

