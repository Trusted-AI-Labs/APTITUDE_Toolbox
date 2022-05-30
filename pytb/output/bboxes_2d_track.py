from pytb.output.bboxes_2d import BBoxes2D

import numpy as np


class BBoxes2DTrack(BBoxes2D):

    def __init__(self, detection_time: float,
                 bboxes: np.array, class_IDs: np.array, det_confs: np.array, dim_width: int, dim_height: int,
                 tracking_time: float, global_IDs: np.array, bboxes_format: str = "xt_yt_w_h"):
        """A class that encompasses the result of a Bounding Box 2D Tracker (such as SORT).
        It extends the BBoxes2D class and adds a np.array for the ID of the object and a tracking time.

        Args:
            tracking_time (float): The time elapsed in the tracking
            global_IDs (np.array): A 1-dimensional array that follows the same order as 'bboxes'.
                It contains the IDs of the detected object that should be consistent between successive frames.
        """
        super().__init__(detection_time, bboxes, class_IDs, det_confs, dim_width, dim_height, bboxes_format)

        self.tracking_time = tracking_time
        self.global_IDs = global_IDs

    def remove_idx(self, indices: list):
        """Removes the bounding boxes together with their confidence, class ID and global ID given a list of indices.

        This method does not return the object instance, it modifies directly the instance's attributes.

        Args:
            indices (list): The indices of the detected objects to remove.
        """
        super().remove_idx(indices)
        self.global_IDs = np.delete(self.global_IDs, indices)
        self.number_objects -= len(indices)

    # Private methods
    def _select_indices(self, indices: np.array):
        """Selects the bounding boxes to keep together with their confidence,
        class ID and global ID given a list of indices.

        This method does not return the object instance, it modifies directly the instance's attributes.

        Args:
            indices (np.array): The indices of the detected objects to keep.
        """
        super()._select_indices(indices)
        self.global_IDs = np.take(self.global_IDs, indices)
        self.number_objects = len(self.bboxes)

    def __str__(self):
        s = super().__str__()
        s += "\n---------------------------------"
        s += "\n\tGlobal IDs: " + str(self.global_IDs)
        s += "\n\tTracking time: " + str(self.tracking_time)
        return s
