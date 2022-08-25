"""
Copyright (c) 2021-2022 UCLouvain, ICTEAM
Licensed under GPL-3.0 [see LICENSE for details]
Written by Jonathan Samelson (2021-2022)
"""


from abc import abstractmethod

from pytb.output.bboxes_2d import BBoxes2D
from pytb.output.bboxes_2d_track import BBoxes2DTrack
from pytb.tracking.tracker import Tracker


class BBoxes2DTracker(Tracker):

    def __init__(self, proc_parameters: dict):
        """
        This class encompasses the attributes that are common to most trackers of 2D bounding boxes.
        Initializes the BBoxes2D tracker with the given parameters.
        
        Args:
            proc_parameters (dict): A dictionary containing the parameters of the desired tracker.
        """
        super().__init__()

        # A tracker can have multiple implementations (e.g. in different frameworks),
        # this parameter allows to choose one (required).
        self.pref_implem = proc_parameters["pref_implem"]

    @abstractmethod
    def track(self, detection: BBoxes2D) -> BBoxes2DTrack:
        """Performs an inference on the given frame. 

        Args:
            detection (BBoxes2D): The detection used to infer IDs.

        Returns:
            BBoxes2DTrack: A set of 2D bounding boxes identifying detected objects with the tracking information added.
        """
        pass

    @abstractmethod
    def reset_state(self):
        """Reset the current state of the tracker."""
        pass
