from pytb.output.detection import Detection

import numpy as np
import cv2


class BBoxes2D(Detection):

    def __init__(self, detection_time: float,
                 bboxes: np.array, class_IDs: np.array, det_confs: np.array,
                 dim_width: int, dim_height: int, bboxes_format: str = "xt_yt_w_h"):
        """A class that encompasses the result of a Bounding Box 2D Detector (such as YOLO).
        It also includes utility methods to filter the results or to apply transformation.

        Args:
            detection_time (float): The time elapsed in detection (e.g. forward method of a DNN)
            bboxes (np.array): The bounding boxes, including one np.array for each bounding box
                (i.e. it is a 2-dimensional array).
                It is either formatted as [[x_top, y_top, width, height], [...], [...], ...] (default)
                or [[x_top_left, y_top_left, x_bottom_right, y_bottom_right], [...], [...], ...]
                depending on the value of bboxes_format
            class_IDs (np.array): The ID of the class of each detected object.
                It follows the order of 'bboxes', but it is a 1-dimensional array.
            det_confs (np.array): The confidence of each detected object, typically ranging from 0 to 1.
                It follows the order of 'bboxes', but it is a 1-dimensional array.
            dim_width (int): The image width on which the 'bboxes' have been detected.
            dim_height (int): The image height on which the 'bboxes' have been detected.
            bboxes_format (str): Either "xt_yt_w_h" (default) or "x1_y1_x2_y2"
                depending on the format of 'bboxes'.
        """
        number_objects = 0 if bboxes is None else len(bboxes)
        super().__init__(number_objects=number_objects)

        self.detection_time = detection_time

        self.bboxes = bboxes
        self.bboxes_format = bboxes_format
        self.class_IDs = class_IDs
        self.det_confs = det_confs

        self.dim_width = dim_width
        self.dim_height = dim_height

    def nms_filter(self, pref_implem: str, nms_thresh: float):
        """Apply a Non-Max Suppression algorithm. Based on a threshold, it removes bounding boxes that overlap
        and only keeps one entity based on a selection criteria. This criteria depends on the implementation.

        This method does not return the object instance, it filters out directly on the instance's attributes.

        Args:
            pref_implem (str): At the moment, either "cv2" or "Malisiewicz".
                The former is the implementation of OpenCV, the latter is a custom implementation published in
                pyimagesearch (see more information below in _nms_malisiewicz documentation)
            nms_thresh (float): The threshold to apply the Non-Max Suppression ranging from 0 to 1.

        """
        if self.number_objects > 0:
            if pref_implem == "cv2":
                return self.cv2_filter(nms_thresh, conf_thresh=0, eta=1.0, top_k=0)
            elif pref_implem == "Malisiewicz":
                return self._nms_malisiewicz(nms_thresh)

    def top_k(self, k: int):
        """Keeps only the K most confident prediction. If a choice has to be made between two detections
        that have the same confidence score, this choice is arbitrary.

        This method does not return the object instance, it filters out directly on the instance's attributes.

        Args:
            k (int): The number of detections to keep (i.e. including the Kth most confident detection).
        """
        if k > 0:
            elements_of_interest = np.lexsort([self.det_confs])[::-1][:k]
            self._select_indices(elements_of_interest)

    def confidence_filter(self, threshold: float):
        """Keeps only the detections that have a confidence score above the threshold.

        This method does not return the object instance, it filters out directly on the instance's attributes.

        Args:
            threshold (float): The threshold above which the detection is kept.
        """
        elements_of_interest = np.argwhere([e > threshold for e in self.det_confs]).flatten()
        self._select_indices(elements_of_interest)

    def class_filter(self, classes_of_interest: set):
        """Keeps only the detections that are included in the set of classes of interest.

        This method does not return the object instance, it filters out directly on the instance's attributes.

        Args:
            classes_of_interest (set): The set of classes that will be kept.
        """
        elements_of_interest = np.argwhere([c in classes_of_interest for c in self.class_IDs]).flatten()
        self._select_indices(elements_of_interest)

    def height_filter(self, threshold: float, max_filter: bool):
        """Keeps only the detections whose the height is above/below the percentage of the height of the frame
        on which the detection has been found (represented by dim_height attribute).

        This method does not return the object instance, it filters out directly on the instance's attributes.

        Args:
            threshold (float): The percentage of the image height.
            max_filter (bool): Whether to apply a max or min filter. True means it keeps only detections
                strictly below the threshold, False means it keeps only detections strictly above the threshold.
        """
        if self.bboxes_format == "xt_yt_w_h":
            if max_filter:
                elements_of_interest = np.argwhere(
                    [e[3] < self.dim_height * threshold for e in self.bboxes]).flatten()
            else:
                elements_of_interest = np.argwhere(
                    [e[3] > self.dim_height * threshold for e in self.bboxes]).flatten()
        else:
            if max_filter:
                elements_of_interest = np.argwhere(
                    [(e[3] - e[1]) < self.dim_height * threshold for e in self.bboxes]).flatten()
            else:
                elements_of_interest = np.argwhere(
                    [(e[3] - e[1]) > self.dim_height * threshold for e in self.bboxes]).flatten()
        self._select_indices(elements_of_interest)

    def width_filter(self, threshold: float, max_filter: bool):
        """Keeps only the detections whose the width is above/below the percentage of the width of the frame
        on which the detection has been found (represented by dim_width attribute).

        This method does not return the object instance, it filters out directly on the instance's attributes.

        Args:
            threshold (float): The percentage of the image width.
            max_filter (bool): Whether to apply a max or min filter. True means it keeps only detections
                strictly below the threshold, False means it keeps only detections strictly above the threshold.
        """
        if self.bboxes_format == "xt_yt_w_h":
            if max_filter:
                elements_of_interest = np.argwhere(
                    [e[2] < self.dim_width * threshold for e in self.bboxes]).flatten()
            else:
                elements_of_interest = np.argwhere(
                    [e[2] > self.dim_width * threshold for e in self.bboxes]).flatten()
        else:
            if max_filter:
                elements_of_interest = np.argwhere(
                    [(e[2] - e[0]) < self.dim_width * threshold for e in self.bboxes]).flatten()
            else:
                elements_of_interest = np.argwhere(
                    [(e[2] - e[0]) > self.dim_width * threshold for e in self.bboxes]).flatten()
        self._select_indices(elements_of_interest)

    def min_area_filter(self, threshold: int):
        """Keeps only the detections whose the area is above the minimum area threshold. More specifically,
        if the detection's width multiplied by the detection's height is not (strictly) above the threshold,
        it is filtered out.

        This method does not return the object instance, it filters out directly on the instance's attributes.

        Args:
            threshold (float): The threshold in square pixels (px²). It should be provided in accordance with
                dim_width and dim_height since the width and height of the bounding boxes are scaled according to
                those attributes.
        """
        if self.bboxes_format == "xt_yt_w_h":
            elements_of_interest = np.argwhere(
                [e[2] * e[3] > threshold for e in self.bboxes]).flatten()
        else:
            elements_of_interest = np.argwhere(
                [(e[2] - e[0]) * (e[3] - e[1]) > threshold for e in self.bboxes]).flatten()
        self._select_indices(elements_of_interest)

    def cv2_filter(self, nms_thresh: float, conf_thresh: float, eta: float = 1.0, top_k: int = 0):
        """Filters the detections using OpenCV NMSBoxes. In addition to Non Max Suppression (NMS),
        it can be used to directly remove  the bounding boxes that are below a confidence threshold or
        to keep only K detections.

        This method does not return the object instance, it filters out directly on the instance's attributes.

        Args:
            nms_thresh (float): The threshold for the non-max suppression (NMS).
            conf_thresh (float): The confidence threshold. Only detections above this threshold are kept.
            eta (float): A coefficient in adaptive threshold formula : nms_thresh i+1 = eta . nms_thresh i
            top_k (int): The maximum numbers of bounding boxes to keep.
        """
        elements_of_interest = cv2.dnn.NMSBoxes(self.bboxes.tolist(), self.det_confs.tolist(),
                                                conf_thresh, nms_thresh, eta, top_k)[:, 0]
        self._select_indices(elements_of_interest)

    def roi_filter(self, roi: np.ndarray, max_outside_roi_thresh: float):
        """Removes the bounding boxes that are outside the Region Of Interest (ROI) based on a threshold.

        This method does not return the object instance, it filters out directly the instance's attributes.

        Args:
            roi (np.ndarray): The binary image mask. If the provided image has more than 1 channel,
                it will be automatically converted to a 1-channel image.
                The mask must match dim_width and dim_height attributes.
            max_outside_roi_thresh (float): The threshold of area percentage above which a bounding box
                has to be removed ranging from 0 to 1.
        """
        assert roi.shape[1] == self.dim_width and roi.shape[0] == self.dim_height, \
            "Could not filter detections using ROI because the ROI dimensions (W, H: {}) " \
            "does not match detections dimensions (W, H: {})." \
                .format((roi.shape[1], roi.shape[0]), (self.dim_width, self.dim_height))

        # Convert to 1-channel image if necessary
        if len(roi.shape) == 3 and roi.shape[2] > 1:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        elements_of_interest = []

        for i, bbox in enumerate(self.bboxes):
            # Create a black image
            bbox_rect = np.zeros((self.dim_height, self.dim_width), np.uint8)

            # Draw a white rectangle
            if self.bboxes_format == "xt_yt_w_h":
                cv2.rectangle(bbox_rect, (round(bbox[0]), round(bbox[1])),
                              (round(bbox[0] + bbox[2]), round(bbox[1] + bbox[3])), color=255, thickness=-1)
            else:
                cv2.rectangle(bbox_rect, (round(bbox[0]), round(bbox[1])),
                              (round(bbox[2]), round(bbox[3])), color=255, thickness=-1)
            nb_white_px_before = len(np.where(bbox_rect.flatten() == 255)[0])

            # Bitwise and operation results in a white box that is the area in the ROI
            after = cv2.bitwise_and(roi, bbox_rect)
            nb_white_px_after = len(np.where(after.flatten() == 255)[0])
            percentage_outside_roi = 1 - (nb_white_px_after / nb_white_px_before)

            # Keep only bounding boxes whose the area percentage outside
            # the ROI are not above the defined threshold
            if percentage_outside_roi <= max_outside_roi_thresh:
                elements_of_interest.append(i)

        self._select_indices(np.array(elements_of_interest, np.uint8))

    # Reformatting methods
    def to_x1_y1_x2_y2(self):
        """Changes the format of the 'bboxes' attribute. The bounding boxes array is converted from
        [x_top, y_top, width, height] to [x_left_top, y_left_top, x_right_bottom, y_right_bottom].

        This method does not return the object instance, it modifies directly the instance's attributes.
        """
        if self.bboxes_format == "xt_yt_w_h":
            for bbox in self.bboxes:
                bbox += np.array([0, 0, bbox[0], bbox[1]])
            self.bboxes_format = "x1_y1_x2_y2"

    def to_xt_yt_w_h(self):
        """Changes the format of the 'bboxes' attribute. The bounding boxes array is converted from
        [x_left_top, y_left_top, x_right_bottom, y_right_bottom] to [x_top, y_top, width, height].

        This method does not return the object instance, it modifies directly the instance's attributes.
        """
        if self.bboxes_format == "x1_y1_x2_y2":
            for i, bbox in enumerate(self.bboxes):
                self.bboxes[i] = np.array([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])
            self.bboxes_format = "xt_yt_w_h"

    def change_dims(self, new_width: int, new_height: int):
        """It sets the dim_width and dim_height to the provided values.
        Scales up/down the dimensions of the bounding boxes in accordance to the new values such that the
        dimensions of the bounding boxes are like if they were detected on an image whose the dimension
        is dim_width x dim_height.

        This method does not return the object instance, it modifies directly the instance's attributes.

        Args:
            new_width (int): The new value of dim_width.
            new_height (int): The new value of dim_height.
        """

        ratio_width = (new_width / self.dim_width)
        ratio_height = (new_height / self.dim_height)
        for i, bbox in enumerate(self.bboxes):
            self.bboxes[i] = bbox * np.array([ratio_width, ratio_height, ratio_width, ratio_height])
        self.dim_width = new_width
        self.dim_height = new_height

    def remove_borders(self, borders_px: np.array):
        """Adjusts the bounding boxes sizes to take into account the removal of black borders around the image.

        This method does not return the object instance, it modifies directly the instance's attributes.

        Args:
            borders_px (np.array): The number of pixels that was added on each side of the frame
            in the following order: [right, left, bottom, top].
        """
        if borders_px[1] > 0 or borders_px[3] > 0:
            for i, bbox in enumerate(self.bboxes):
                if self.bboxes_format == "xt_yt_w_h":
                    self.bboxes[i] = bbox - np.array([borders_px[1], borders_px[3], 0, 0])
                else:
                    self.bboxes[i] = bbox - np.array([borders_px[1], borders_px[3], borders_px[1], borders_px[3]])

        self.dim_width -= borders_px[0] + borders_px[1]
        self.dim_height -= borders_px[2] + borders_px[3]

    def add_borders(self, borders_px):
        """Adjusts the bounding boxes sizes to take into account the addition of black borders around the image.

        This method does not return the object instance, it modifies directly the instance's attributes.

        Args:
            borders_px (np.array): The number of pixels that was added on each side of the frame
            in the following order: [right, left, bottom, top].
        """
        self.dim_width += borders_px[0] + borders_px[1]
        self.dim_height += borders_px[2] + borders_px[3]

        if borders_px[1] > 0 or borders_px[3] > 0:
            for i, bbox in enumerate(self.bboxes):
                if self.bboxes_format == "xt_yt_w_h":
                    self.bboxes[i] = bbox + np.array([borders_px[1], borders_px[3], 0, 0])
                else:
                    self.bboxes[i] = bbox + np.array([borders_px[1], borders_px[3], borders_px[1], borders_px[3]])

    def remove_idx(self, indices: np.array):
        """Removes the bounding boxes together with their confidence and class ID given a list of indices.

        This method does not return the object instance, it modifies directly the instance's attributes.

        Args:
            indices (np.array): The indices of the detected objects to remove.
        """
        if len(indices) > 0:
            self.bboxes = np.delete(self.bboxes, indices, axis=0)
            self.class_IDs = np.delete(self.class_IDs, indices)
            self.det_confs = np.delete(self.det_confs, indices)
            self.number_objects -= len(indices)

    # Private methods
    def _select_indices(self, indices: np.array):
        """Selects the bounding boxes to keep together with their confidence and class ID given a list of indices.

        This method does not return the object instance, it modifies directly the instance's attributes.

        Args:
            indices (np.array): The indices of the detected objects to keep.
        """
        if len(indices) > 0:
            self.bboxes = np.take(self.bboxes, indices, axis=0)
            self.det_confs = np.take(self.det_confs, indices)
            self.class_IDs = np.take(self.class_IDs, indices)
            self.number_objects = len(self.bboxes)
        else:
            self.bboxes = np.array([])
            self.det_confs = np.array([])
            self.class_IDs = np.array([])
            self.number_objects = 0

    def _nms_malisiewicz(self, nms_thresh):
        """An implementation of Non-max Suppression (NMS) by Dr. Tomasz Malisiewicz.

        Original code from [1]_  that has been adapted to include confidence score.

        .. [1] http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

        Args:
            nms_thresh (float): The threshold to apply the Non-Max Suppression ranging from 0 to 1.
        """
        self.to_x1_y1_x2_y2()
        boxes = self.bboxes.astype(np.float)
        pick = []

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        if self.det_confs is not None:
            idxs = np.argsort(self.det_confs)
        else:
            idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > nms_thresh)[0])))

        self.to_xt_yt_w_h()
        self._select_indices(np.array(pick))

    def __str__(self):
        s = super().__str__()
        s += "\n---------------------------------"
        s += "\n\tclass IDs: " + str(self.class_IDs)
        s += "\n\tDetection confidences: " + str(self.det_confs)
        s += "\n\tBounding Boxes: " + str(self.bboxes)
        s += "\n\tFormat: " + str(self.bboxes_format)
        s += "\n\tDetection time: " + str(self.detection_time)
        return s
