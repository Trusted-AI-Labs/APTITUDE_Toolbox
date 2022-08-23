"""
Copyright (c) 2021-2022 UCLouvain, ICTEAM
Licensed under GPL-3.0 [see LICENSE for details]
Written by Jonathan Samelson (2021-2022)
"""

import logging
from pytb.tracking.tracker import Tracker
from pytb.tracking.bboxes.bboxes_2d_tracker.bboxes_2d_tracker import BBoxes2DTracker
import pytb.utils.validator as val

log = logging.getLogger("aptitude-toolbox")


class TrackerFactory:

    @staticmethod
    def create_tracker(proc_parameters: dict) -> Tracker:
        """
        Creates a tracker given a dictionary of parameters 
        provided that the defined parameters are valid. 
        Otherwise, an error message is given indicating the invalid parameters.
        Based on a model type and an implementation, it initializes the appropriate tracker with the given parameters.

        Args:
            proc_parameters (dict): A dictionary describing the tracker to initialize.

        Returns:
            Detection: A concrete implementation of a Tracker (e.g YOLO).
        """
        # assert val.validate_proc_parameters(proc_parameters), \
        #     "[ERROR] Invalid Proc (tracker) parameter(s) detected, check the above for details."
        track_type = proc_parameters["output_type"]

        if track_type == "bboxes2D":
            return TrackerFactory._bboxes_2d_tracker(proc_parameters)

        elif track_type == "pose":
            return TrackerFactory._pose_tracker(proc_parameters)

    @staticmethod
    def _bboxes_2d_tracker(proc_parameters: dict) -> BBoxes2DTracker:
        """
        Creates a `BBoxes2DTracker` given a dictionary of parameters.
        It then branches on the model type to initialize a concrete detector implementation (e.g SORT).
        """
        model_type = proc_parameters["model_type"]

        log.info("Model type {} selected.".format(model_type))
        if model_type == "SORT":
            from pytb.tracking.bboxes.bboxes_2d_tracker.mbtracker.sort.sort import SORT
            return SORT(proc_parameters)
        elif model_type == "DeepSORT":
            from pytb.tracking.bboxes.bboxes_2d_tracker.mbtracker.deepsort.deepsort import DeepSORT
            return DeepSORT(proc_parameters)
        elif model_type == "Centroid":
            from pytb.tracking.bboxes.bboxes_2d_tracker.mbtracker.centroid.centroid import Centroid
            return Centroid(proc_parameters)
        elif model_type == "IOU":
            from pytb.tracking.bboxes.bboxes_2d_tracker.mbtracker.iou.iou import IOU
            return IOU(proc_parameters)

    @staticmethod
    def _pose_tracker(proc_parameters: dict) -> None:
        """
        TODO. No implementation of pose tracker yet.
        Creates a PoseTracker given a dictionary of parameters.
        It then branches on the model type to initialize a concrete tracker implementation (e.g XXX).
        """
        pass
