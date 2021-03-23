from pytb.tracking.tracker import Tracker
from pytb.output.detection import Detection
import pytb.utils.transformation as tfm

from typing import Optional
from timeit import default_timer
import numpy as np
import cv2


class TrackingManager:

    def __init__(self, tracker: Tracker, preprocess_parameters: dict, postprocess_parameters: dict):

        # _validate_preprocess_parameters(preprocess_parameters)
        # _validate_postprocess_parameters(postprocess_parameters)

        self.tracker = tracker
        self.preprocess_parameters = preprocess_parameters
        self.postprocess_parameters = postprocess_parameters

    @staticmethod
    def _validate_preprocess_parameters(preprocess_parameters: dict):
        # TODO
        pass

    @staticmethod
    def _validate_postprocess_parameters(postprocess_parameters: dict):
        # TODO
        pass

    def track(self, detection: Detection, frame: Optional[np.ndarray] = None) -> Detection:
        start = default_timer()
        if frame is not None:
            frame = tfm.pre_process(self.preprocess_parameters, frame)
            resize_params = self.preprocess_parameters["resize"]
            detection.change_dims(resize_params["width"], resize_params["height"])
        preproc_time = default_timer() - start

        # call the concrete method of the tracker
        start = default_timer()
        if self.tracker.need_frame:
            track = self.tracker.track(detection, frame)
        else:
            track = self.tracker.track(detection)
        track.processing_time += default_timer() - start

        track.preprocessing_time = preproc_time

        # Post process
        start = default_timer()
        if track.number_objects != 0:
            track = tfm.post_process(self.postprocess_parameters, track)
        track.postprocessing_time = default_timer() - start
        return track

    def reset_state(self):
        self.tracker.reset_state()
