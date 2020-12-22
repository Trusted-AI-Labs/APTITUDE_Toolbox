from pytb.output.detection import Detection
import numpy as np
import cv2

class BBoxes2D(Detection):

    def __init__(self, detection_time: float, 
                bboxes: np.ndarray, class_IDs: np.ndarray, det_confs: np.ndarray, dim_width: int, dim_height: int):
        super().__init__(number_objects=len(bboxes))

        self.detection_time = detection_time

        self.bboxes = bboxes # format [xc, yc, w, h]
        self.class_IDs = class_IDs
        self.det_confs = det_confs

        self.prev_track_IDs = None

        self.dim_width = dim_width
        self.dim_height = dim_height

    def nms_filter(self, pref_implem: str, nms_thresh: float):
        if pref_implem == "cv2":
            return self.cv2_filter(nms_thresh, conf_thresh=0, eta=1.0, top_k=0)

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

    def cv2_filter(self, nms_thresh: float, conf_thresh: float, eta=1.0, top_k=0):
        if self.number_objects != 0:
            elements_of_interest = cv2.dnn.NMSBoxes(self.bboxes.tolist(), self.det_confs.tolist(), conf_thresh, nms_thresh, eta, top_k)[:,0]
            self._select_indices(elements_of_interest)

    def _select_indices(self, indices: np.ndarray):
        self.bboxes = np.take(self.bboxes, indices, axis=0)
        self.det_confs = np.take(self.det_confs, indices)
        self.class_IDs = np.take(self.class_IDs, indices)
        self.number_objects = len(self.bboxes)

    # Reformating methods
    def to_x1_y1_x2_y2(self):
        for bbox in self.bboxes:
            bbox += np.array([0, 0, bbox[0], bbox[1]])

    def to_xt_yt_w_h(self):
        for i, bbox in enumerate(self.bboxes):
            self.bboxes[i] = np.array([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])

    def change_dims(self, new_width, new_height):
        ratio_width = (new_width / self.dim_width)
        ratio_height = (new_height / self.dim_height)
        for i, bbox in enumerate(self.bboxes):
            self.bboxes[i] = bbox * np.array([ratio_width, ratio_height, ratio_width, ratio_height])
        self.dim_width = new_width
        self.dim_height = new_height

    def __str__(self):
        s = super().__str__()
        s += "\n---------------------------------"
        s += "\n\tclass IDs: " + str(self.class_IDs)
        s += "\n\tDetection confidences: " + str(self.det_confs)
        s += "\n\tBounding Boxes: " + str(self.bboxes)
        s += "\n\tPrevious track IDs : " + str(self.prev_track_IDs)
        s += "\n\tDetection time: " + str(self.detection_time)
        return s


