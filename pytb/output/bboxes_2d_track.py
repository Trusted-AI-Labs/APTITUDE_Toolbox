from pytb.output.bboxes_2d import BBoxes2D

from typing import Optional
import numpy as np


class BBoxes2DTrack(BBoxes2D):

    def __init__(self, detection_time: float,
                 bboxes: np.array, class_IDs: np.array, det_confs: np.array, dim_width: int, dim_height: int,
                 tracking_time: float, global_IDs: np.array, bboxes_format: Optional[str] = None):
        super().__init__(detection_time, bboxes, class_IDs, det_confs, dim_width, dim_height, bboxes_format)

        self.tracking_time = tracking_time
        self.global_IDs = global_IDs

    def __str__(self):
        s = super().__str__()
        s += "\n---------------------------------"
        s += "\n\tGlobal IDs: " + str(self.global_IDs)
        s += "\n\tTracking time: " + str(self.tracking_time)
        return s

    # TODO Override filter methods
