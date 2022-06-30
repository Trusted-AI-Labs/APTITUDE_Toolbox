"""
Copyright (c) 2021-2022 UCLouvain, ICTEAM
Licensed under GPL-3.0 [see LICENSE for details]
Written by Jonathan Samelson (2021-2022)
"""

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
        """
        This class is used to handle all the steps of the tracking in one single `track` method.
        Specifically, it applies the preprocess parameters, infer the object IDs based on the detected objects 
        and filters out the results using the postprocess parameters.
        This constructor initialize a TrackingManager provided that the preprocess and post process
        parameters are valid. Otherwise, an error message is given indicating the invalid parameter(s).


        Args:
            tracker (Tracker): A concrete implementation of a Tracker initialized using the `tracker_factory`.
            preprocess_parameters (dict): A dictionary containing the image transformation before proceeding to the tracking.
            postprocess_parameters (dict): A dictionary containing the thresholds to filter out the results of the tracking. 

        Returns:
            Detection: An initalized DetectionManager that can be called on each frame using `detect` method.
        """
        assert val.validate_preprocess_parameters(preprocess_parameters), \
            "[ERROR] Invalid Preproc parameter(s) detected, check any error reported above for details."
        assert val.validate_postprocess_parameters(postprocess_parameters), \
            "[ERROR] Invalid Postproc parameter(s) detected, check any error reported above for details."

        self.tracker = tracker
        self.preprocess_parameters = preprocess_parameters
        self.postprocess_parameters = postprocess_parameters

        # Those two attributes are used to keep the value of the ROI to be applied an image sequence
        # to avoid repeated readings.
        self.preproc_roi = None
        self.postproc_roi = None

        # TODO define a method to handle the proper update of the parameters

    def track(self, detection: Detection, frame: Optional[np.ndarray] = None) -> Detection:
        """
        Proceeds to the tracking on the given frame.
        It first transforms the image using the preprocess parameters of this instance of `TrackingManager`.
        Then, it infers the results of the detection (see the concrete implementation of the provided tracker). 
        Finally, it filters out the result using the postprocess parameters of this instance of `TrackingManager`.
        """
        start = default_timer()
        border_px = None
        if frame is not None:
            frame, self.preproc_roi, border_px, detection = tfm.pre_process(self.preprocess_parameters, frame,
                                                                    self.preproc_roi, detection)
            log.debug("Preprocessing done.")
        preproc_time = default_timer() - start

        # if border_px has a value, it means that borders were added
        # and they will need to be removed during postprocess. 
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
        """
        Resets the state of the tracker as if it was just initialized.
        """
        self.tracker.reset_state()
        log.debug("Tracker state reset.")
