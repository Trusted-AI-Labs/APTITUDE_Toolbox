from pytb.tracking.tracker import Tracker
from pytb.tracking.bboxes.bboxes_2d_tracker.bboxes_2d_tracker import BBoxes2DTracker


class TrackerFactory:

    @staticmethod
    def create_tracker(tracker_parameters: dict) -> Tracker:
        # _validate_parameters(detector_parameters)
        det_type = tracker_parameters["Tracker"]["type"]

        if det_type == "BBoxes2DTracker":
            return TrackerFactory._bboxes_2d_tracker(tracker_parameters)

        elif det_type == "PoseTracker":
            return TrackerFactory._pose_tracker(tracker_parameters)

    @staticmethod
    def _bboxes_2d_tracker(tracker_parameters: dict) -> BBoxes2DTracker:
        model_type = tracker_parameters["BBoxes2DTracker"]["model_type"]

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
        pass

    @staticmethod
    def _validate_parameters(tracker_parameters: dict) -> bool:
        """Check compatibility between provided parameters.

        Args:
            tracker_parameters (dict): the dictionary containing tracker parameters.
        
        Returns:
            bool: whether it is a valid configuration
        """
        # TODO validate parameters from create_detector
        pass
