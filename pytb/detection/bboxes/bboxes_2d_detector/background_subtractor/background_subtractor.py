from pytb.detection.bboxes.bboxes_2d_detector.bboxes_2d_detector import BBoxes2DDetector
from pytb.output.bboxes_2d import BBoxes2D

from timeit import default_timer
import cv2
import numpy as np
import logging

log = logging.getLogger("aptitude-toolbox")


class BackgroundSubtractor(BBoxes2DDetector):

    def __init__(self, detector_parameters: dict):
        """Initializes the detector with the given parameters.

        Args:
            detector_parameters (dict): A dictionary containing the BackgroundSubstractor detector parameters
        """
        super().__init__(detector_parameters)
        # From cv2.approxPolyDP: Specifies the approximation accuracy. 
        # This is the maximum distance between the original curve and its approximation.
        self.contour_thresh = detector_parameters["BackgroundSubtractor"].get("contour_thresh", 3)

        # The minimum intensity of the pixels in the foreground image.
        self.intensity = detector_parameters["BackgroundSubtractor"].get("intensity", 50)

        log.debug("BackgroundSubtractor {} implementation selected.".format(self.pref_implem))
        if self.pref_implem == "mean" or self.pref_implem == "median":
            # If the pref_implem is "mean" or "median", the results will be based on the mean or median
            # values of the previous images. 
            self.max_last_images = detector_parameters["BackgroundSubtractor"].get("max_last_images", 50)
            self.last_images = []

        elif self.pref_implem == "frame_diff":
            # IF the pref_implem is "frame_diff", the results will be solely based on the previous image.
            self.prev_image = None

        else:
            assert False, "[ERROR] Unknown implementation of BackgroundSubtractor: {}".format(self.pref_implem)

    def detect(self, frame: np.ndarray) -> BBoxes2D:
        """Performs an inference using a background subtraction method on the given frame.

        Args:
            frame (np.ndarray): The frame to infer detections from a background substractor.

        Returns:
            BBoxes2D: A set of 2D bounding boxes identifying  the detected objects.
        """
        img_sub = None

        # Convert the frame to the gray-scale representation
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        start = default_timer()

        if self.pref_implem == "mean" or self.pref_implem == "median":
            self.last_images.append(frame_gray)
            if len(self.last_images) == self.max_last_images + 1:
                self.last_images.pop(0)

            # Obtain mean or median values of the previous images
            if self.pref_implem == "mean":
                background_image = np.mean(np.array(self.last_images), axis=0).astype(np.uint8)
            else:
                background_image = np.median(np.array(self.last_images), axis=0).astype(np.uint8)
            
            # Get the difference between the current image and the previous images (that should describes the background)
            foreground_image = cv2.absdiff(frame_gray, background_image)

            # Keep the most important difference, where the pixels has a higher value than self.intensity,
            # to remove the noise.
            img_sub = np.where(foreground_image > self.intensity, frame_gray, np.array([0], np.uint8))

        elif self.pref_implem == "frame_diff":
            # If no previous image, return an empty result.
            if self.prev_image is None:
                self.prev_image = frame_gray
                return BBoxes2D(0, np.array([]), np.array([]), np.array([]), frame.shape[1], frame.shape[0])
            
            # Otherwise, take the difference with the previous image.
            else:
                foreground_image = cv2.absdiff(self.prev_image, frame_gray)
                img_sub = np.where(foreground_image > self.intensity, frame_gray, np.array([0], np.uint8))
                self.prev_image = frame_gray

        else:
            assert False, "[ERROR] Unknown implementation of BackgroundSubtractor: {}".format(self.pref_implem)

        bboxes = []
        # Find the contours of different objects that can be seen in the foreground image.
        contours = cv2.findContours(img_sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cont in contours[0]:
            poly = cv2.approxPolyDP(cont, self.contour_thresh, True)
            x, y, w, h = cv2.boundingRect(poly)
            bboxes.append([x, y, w, h])
        end = default_timer()

        return BBoxes2D(end-start, np.array(bboxes), np.zeros(len(bboxes)).astype(int),
                        np.ones(len(bboxes)), frame.shape[1], frame.shape[0])
