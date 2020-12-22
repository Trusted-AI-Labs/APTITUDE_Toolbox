from pytb.tracking.tracker import Tracker
from abc import ABC, abstractmethod

from pytb.output.bboxes_2d import BBoxes2D
from pytb.output.bboxes_2d_track import BBoxes2DTrack

class BBoxes2DTracker(Tracker):
    
    def __init__(self, tracker_parameters: dict):
        """Initiliazes the BBoxes2D tracker with the given parameters.

        Args:
            tracker_parameters (dict): A dictionary containing the related tracker's parameters
        """
        super().__init__()
        self.pref_implem = tracker_parameters["BBoxes2DTracker"]["pref_implem"]

    @abstractmethod
    def track(self, detection: BBoxes2D) -> BBoxes2DTrack:
        """Performs an inference on the given frame. 

        Args:
            detection (BBoxes2D): The detection used to infer IDs.

        Returns:
            BBoxes2DTrack: A set of 2DBBoxes detections with the tracking information added.
        """
        pass