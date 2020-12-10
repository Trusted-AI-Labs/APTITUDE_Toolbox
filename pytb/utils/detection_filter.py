import cv2
import numpy as np
from pytb.output.bboxes_2d import BBoxes_2D

class DetectionFilter:

    @staticmethod
    def nms_filter(bboxes_2d, pref_implem, nms_thresh, conf_thresh=0, eta=1.0, top_k=0):
        if pref_implem == "cv2":
            indices = cv2.dnn.NMSBoxes(bboxes_2d.bboxes.tolist(), bboxes_2d.det_confs.tolist(), conf_thresh, nms_thresh, eta, top_k)[:,0]
            bboxes_2d.bboxes = bboxes_2d.bboxes[indices]
            bboxes_2d.det_confs = bboxes_2d.det_confs[indices]
            bboxes_2d.class_IDs = bboxes_2d.class_IDs[indices]
            return bboxes_2d

    @staticmethod
    def top_k(bboxes_2d, k):
        new_order = np.lexsort([bboxes_2d.det_confs])[::-1][:k]
        bboxes_2d.det_confs = bboxes_2d.det_confs[new_order]
        bboxes_2d.class_IDs = bboxes_2d.class_IDs[new_order]
        bboxes_2d.bboxes = bboxes_2d.bboxes[new_order]
        return bboxes_2d

    @staticmethod
    def confidence_filter(bboxes_2d, threshold):
        elements_of_interest = np.argwhere([e > threshold for e in bboxes_2d.det_confs]).flatten()
        return DetectionFilter._select_indices(bboxes_2d, elements_of_interest)

    @staticmethod
    def class_filter(bboxes_2d, classes_of_interest):
        elements_of_interest = np.argwhere([c in classes_of_interest for c in bboxes_2d.class_IDs]).flatten()
        return DetectionFilter._select_indices(bboxes_2d, elements_of_interest)

    @staticmethod
    def _select_indices(bboxes_2d, indices):
        bboxes_2d.bboxes = np.take(bboxes_2d.bboxes, indices, axis=0)
        bboxes_2d.det_confs = np.take(bboxes_2d.det_confs, indices)
        bboxes_2d.class_IDs = np.take(bboxes_2d.class_IDs, indices)
        return bboxes_2d

    @staticmethod
    def cv2_filter(bboxes_2d, conf_thresh, nms_thresh, eta, top_k):
        pass
