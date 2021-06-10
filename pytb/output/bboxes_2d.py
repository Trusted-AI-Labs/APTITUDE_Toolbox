from pytb.output.detection import Detection

from typing import Optional
import numpy as np
import cv2


class BBoxes2D(Detection):

    def __init__(self, detection_time: float,
                 bboxes: np.array, class_IDs: np.array, det_confs: np.array,
                 dim_width: int, dim_height: int, bboxes_format: Optional[str] = None):
        number_objects = 0 if bboxes is None else len(bboxes)
        super().__init__(number_objects=number_objects)

        self.detection_time = detection_time

        self.bboxes = bboxes
        if bboxes_format is None:
            self.bboxes_format = "xt_yt_w_h"
        else:
            self.bboxes_format = bboxes_format
        self.class_IDs = class_IDs
        self.det_confs = det_confs

        self.dim_width = dim_width
        self.dim_height = dim_height

    def nms_filter(self, pref_implem: str, nms_thresh: float):
        if pref_implem == "cv2":
            return self.cv2_filter(nms_thresh, conf_thresh=0, eta=1.0, top_k=0)
        elif pref_implem == "Malisiewicz":
            return self._nms_malisiewicz(nms_thresh)

    def top_k(self, k: int):
        if k > 0:
            elements_of_interest = np.lexsort([self.det_confs])[::-1][:k]
            self._select_indices(elements_of_interest)

    def confidence_filter(self, threshold: float):
        elements_of_interest = np.argwhere([e > threshold for e in self.det_confs]).flatten()
        self._select_indices(elements_of_interest)

    def class_filter(self, classes_of_interest: set):
        elements_of_interest = np.argwhere([c in classes_of_interest for c in self.class_IDs]).flatten()
        self._select_indices(elements_of_interest)

    def height_filter(self, threshold: float, max_filter: bool):
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

    def cv2_filter(self, nms_thresh: float, conf_thresh: float, eta=1.0, top_k=0):
        if self.number_objects != 0:
            elements_of_interest = cv2.dnn.NMSBoxes(self.bboxes.tolist(), self.det_confs.tolist(),
                                                    conf_thresh, nms_thresh, eta, top_k)[:, 0]
            self._select_indices(elements_of_interest)

    # Reformatting methods
    def to_x1_y1_x2_y2(self):
        if self.bboxes_format == "xt_yt_w_h":
            for bbox in self.bboxes:
                bbox += np.array([0, 0, bbox[0], bbox[1]])
            self.bboxes_format = "x1_y1_x2_y2"

    def to_xt_yt_w_h(self):
        if self.bboxes_format == "x1_y1_x2_y2":
            for i, bbox in enumerate(self.bboxes):
                self.bboxes[i] = np.array([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])
            self.bboxes_format = "xt_yt_w_h"

    def change_dims(self, new_width, new_height):
        ratio_width = (new_width / self.dim_width)
        ratio_height = (new_height / self.dim_height)
        for i, bbox in enumerate(self.bboxes):
            self.bboxes[i] = bbox * np.array([ratio_width, ratio_height, ratio_width, ratio_height])
        self.dim_width = new_width
        self.dim_height = new_height

    def remove_idx(self, s):
        self.bboxes = np.delete(self.bboxes, s, axis=0)
        self.class_IDs = np.delete(self.class_IDs, s)
        self.det_confs = np.delete(self.det_confs, s)
        self.number_objects -= len(s)

    # Private methods
    def _select_indices(self, indices: np.ndarray):
        self.bboxes = np.take(self.bboxes, indices, axis=0)
        self.det_confs = np.take(self.det_confs, indices)
        self.class_IDs = np.take(self.class_IDs, indices)
        self.number_objects = len(self.bboxes)

    def _nms_malisiewicz(self, nms_thresh):
        """Original code from [1]_ has been adapted to include confidence score.

        .. [1] http://www.pyimagesearch.com/2015/02/16/
           faster-non-maximum-suppression-python/

        Args:
            nms_thresh ([float]): [description]
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
        return self._select_indices(np.array(pick))

    def __str__(self):
        s = super().__str__()
        s += "\n---------------------------------"
        s += "\n\tclass IDs: " + str(self.class_IDs)
        s += "\n\tDetection confidences: " + str(self.det_confs)
        s += "\n\tBounding Boxes: " + str(self.bboxes)
        s += "\n\tFormat: " + str(self.bboxes_format)
        s += "\n\tDetection time: " + str(self.detection_time)
        return s
