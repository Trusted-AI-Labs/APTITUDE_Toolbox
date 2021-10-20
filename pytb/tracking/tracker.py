from abc import ABC, abstractmethod
from pytb.output.detection import Detection


class Tracker(ABC):

    @abstractmethod
    def __init__(self):
        """Initializes the tracker with the given parameters.
        """
        super().__init__()
        self.need_frame = False

    @abstractmethod
    def track(self, detection: Detection) -> Detection:
        """Performs a tracking method to match the IDs between frames. 

        Args:
            detection (Detection): The detection used to infer IDs.
        
        Returns:
            Detection: A set of detections with the tracking information added.
        """
        pass

    @abstractmethod
    def reset_state(self):
        """Reset the current state of the tracker."""
        pass
