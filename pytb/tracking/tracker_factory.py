import logging
from pytb.tracking.tracker import Tracker
from pytb.tracking.bboxes.bboxes_2d_tracker.bboxes_2d_tracker import BBoxes2DTracker
import pytb.utils.validator as val

log = logging.getLogger("aptitude-toolbox")

class TrackerFactory:

    @staticmethod
    def create_tracker(tracker_parameters: dict) -> Tracker:
        """
        Creates a tracker given a dictionary of parameters 
        provided that the defined parameters are valid. 
        Otherwise, an error message is given indicating the invalid parameters.
        It first branches on the tracker type (e.g. `BBoxes2DTracker`) 
        and then follow the chain to initialize the required tracker.

        Args:
            tracker_parameters (dict): A dictionary describing the tracker to initialize.

        Returns:
            Detection: A concrete implementation of a Tracker (e.g YOLO).
        """
        assert val.validate_tracker_parameters(tracker_parameters), \
            "[ERROR] Invalid Proc (tracker) parameter(s) detected, check the above for details."
        track_type = tracker_parameters["Tracker"]["type"]

        if track_type == "BBoxes2DTracker":
            return TrackerFactory._bboxes_2d_tracker(tracker_parameters)

        elif track_type == "PoseTracker":
            return TrackerFactory._pose_tracker(tracker_parameters)

    @staticmethod
    def _bboxes_2d_tracker(tracker_parameters: dict) -> BBoxes2DTracker:
        """
        Creates a `BBoxes2DTracker` given a dictionary of parameters.
        It then branches on the model type to initialize a concrete detector implementation (e.g SORT).
        """
        model_type = tracker_parameters["BBoxes2DTracker"]["model_type"]

        log.info("Model type {} selected.".format(model_type))
        if model_type == "SORT":
            from pytb.tracking.bboxes.bboxes_2d_tracker.mbtracker.sort.sort import SORT
            return SORT(tracker_parameters)
        elif model_type == "DeepSORT":
            from pytb.tracking.bboxes.bboxes_2d_tracker.mbtracker.deepsort.deepsort import DeepSORT
            return DeepSORT(tracker_parameters)
        elif model_type == "Centroid":
            from pytb.tracking.bboxes.bboxes_2d_tracker.mbtracker.centroid.centroid import Centroid
            return Centroid(tracker_parameters)
        elif model_type == "IOU":
            from pytb.tracking.bboxes.bboxes_2d_tracker.mbtracker.iou.iou import IOU
            return IOU(tracker_parameters)

    @staticmethod
    def _pose_tracker(tracker_parameters: dict) -> None:
        """
        TODO. No implementation of pose tracker yet.
        Creates a PoseTracker given a dictionary of parameters.
        It then branches on the model type to initialize a concrete tracker implementation (e.g XXX).
        """
        pass
