from pytb.detection.bboxes.bboxes_2d_detector.bboxes_2d_detector import BBoxes2DDetector
from pytb.output.bboxes_2d import BBoxes2D

from timeit import default_timer
import cv2
import numpy as np


class BackgroundSubtractor(BBoxes2DDetector):

    def __init__(self, detector_parameters: dict):
        """Initializes the detectors with the given parameters.

        Args:
            detector_parameters (dict): A dictionary containing the BackgroundSubstractor detector parameters
        """
        super().__init__(detector_parameters)
        self.contour_thresh = detector_parameters["BackgroundSubtractor"].get("contour_thresh", 3)
        self.intensity = detector_parameters["BackgroundSubtractor"].get("intensity", 50)

        if self.pref_implem == "mean" or self.pref_implem == "median":
            self.max_last_images = detector_parameters["BackgroundSubtractor"].get("max_last_images", 50)
            self.last_images = []

        if self.pref_implem == "frame_diff":
            self.prev_image = None

    def detect(self, frame: np.ndarray) -> BBoxes2D:
        """Performs an inference using a background subtraction method on the given frame.

        Args:
            frame (np.ndarray): The frame to infer YOLO detections

        Returns:
            BBoxes2D: A set of 2DBBoxes of the detected objects.
        """
        img_sub = None

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        start = default_timer()

        if self.pref_implem == "mean" or self.pref_implem == "median":
            self.last_images.append(frame_gray)
            if len(self.last_images) == self.max_last_images + 1:
                self.last_images.pop(0)

            if self.pref_implem == "mean":
                background_image = np.mean(np.array(self.last_images), axis=0).astype(np.uint8)
            else:
                background_image = np.median(np.array(self.last_images), axis=0).astype(np.uint8)
            foreground_image = cv2.absdiff(frame_gray, background_image)
            img_sub = np.where(foreground_image > self.intensity, frame_gray, np.array([0], np.uint8))

        if self.pref_implem == "frame_diff":
            if self.prev_image is None:
                self.prev_image = frame_gray
                return BBoxes2D(0, np.array([]), np.array([]), np.array([]), frame.shape[1], frame.shape[0])
            else:
                foreground_image = cv2.absdiff(self.prev_image, frame_gray)
                img_sub = np.where(foreground_image > self.intensity, frame_gray, np.array([0], np.uint8))
                self.prev_image = frame_gray

        before_cont = default_timer()
        bboxes = []
        contours = cv2.findContours(img_sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cont in contours[0]:
            poly = cv2.approxPolyDP(cont, self.contour_thresh, True)
            x, y, w, h = cv2.boundingRect(poly)
            bboxes.append([x, y, w, h])
        end = default_timer()

        # print("-----------------------")
        # print("method", before_cont-start)
        # print("contour", end-before_cont)
        # print("total", end-start)

        return BBoxes2D(end-start, np.array(bboxes), np.zeros(len(bboxes)), np.ones(len(bboxes)),
                        frame.shape[1], frame.shape[0])
