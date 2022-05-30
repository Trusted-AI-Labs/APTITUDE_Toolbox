from pytb.tracking.tracker import Tracker
from pytb.output.detection import Detection
import pytb.utils.transformation as tfm
import pytb.utils.validator as val

from typing import Optional
from timeit import default_timer
import numpy as np
import logging

log = logging.getLogger("aptitude-toolbox")


class TrackingManager:

    def __init__(self, tracker: Tracker, preprocess_parameters: dict, postprocess_parameters: dict):
        assert val.validate_preprocess_parameters(preprocess_parameters), \
            "[ERROR] Invalid Preproc parameter(s) detected, check any error reported above for details."
        assert val.validate_postprocess_parameters(postprocess_parameters), \
            "[ERROR] Invalid Postproc parameter(s) detected, check any error reported above for details."

        self.tracker = tracker
        self.preprocess_parameters = preprocess_parameters
        self.postprocess_parameters = postprocess_parameters

        self.preproc_roi = None
        self.postproc_roi = None

    def track(self, detection: Detection, frame: Optional[np.ndarray] = None) -> Detection:
        start = default_timer()
        border_px = None
        if frame is not None:
            frame, self.preproc_roi, border_px, detection = tfm.pre_process(self.preprocess_parameters, frame,
                                                                    self.preproc_roi, detection)
            log.debug("Preprocessing done.")
        preproc_time = default_timer() - start

        if border_px is not None:
            self.postprocess_parameters["borders_detection"] = border_px

        # call the concrete method of the tracker
        start = default_timer()
        if self.tracker.need_frame:
            track = self.tracker.track(detection, frame)
        else:
            track = self.tracker.track(detection)
        track.processing_time += default_timer() - start

        track.preprocessing_time = preproc_time
        log.debug("Actual tracking done.")

        # Post process
        start = default_timer()
        if track.number_objects != 0:
            track, self.postproc_roi = tfm.post_process(self.postprocess_parameters, track, self.postproc_roi)
        track.postprocessing_time = default_timer() - start
        log.debug("Postprocessing done.")

        return track

    def reset_state(self):
        self.tracker.reset_state()
        log.debug("Tracker state reset.")
