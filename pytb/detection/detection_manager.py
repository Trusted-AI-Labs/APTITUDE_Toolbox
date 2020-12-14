import pytb.utils.image_helper as ih

from timeit import default_timer
import cv2

class DetectionManager:

    def __init__(self, detector, preprocess_parameters, postprocess_parameters):
        # TODO check detector object of type "Detector"

        # _validate_preprocess_parameters(preprocess_parameters)
        # _validate_postprocess_parameters(postprocess_parameters)
        
        self.detector = detector
        self.preprocess_parameters = preprocess_parameters
        self.postprocess_parameters = postprocess_parameters

    @staticmethod
    def _validate_preprocess_parameters(preprocess_parameters):
        # TODO
        pass

    @staticmethod
    def _validate_postprocess_parameters(postprocess_parameters):
        #TODO
        pass

    def detect(self, org_frame):
        start = default_timer()
        edit_frame = self._pre_process(org_frame)
        preproc_time = default_timer()-start

        # call the concrete method of the detector
        start = default_timer()
        detections = self.detector.detect(edit_frame)
        detections.processing_time = default_timer()-start

        detections.preprocessing_time = preproc_time

        # Post process
        start = default_timer()
        if detections.number_detections != 0:
            detections = self._post_process(detections)
        detections.postprocessing_time = default_timer()-start

        # Display results
        # ratio = max(edit_frame.shape)
        # for b in detections.bboxes:
        #     (x, y, w, h) = b
        #     x = int(x * (ratio/detections.dim_width))
        #     w = int(w * (ratio/detections.dim_width))
        #     y = int(y * (ratio/detections.dim_height))
        #     h = int(h * (ratio/detections.dim_height))
        #     color = (255,0,0)
        #     cv2.rectangle(edit_frame, (x, y), (x + w, y + h), color, 2)
        # cv2.imshow("res", edit_frame)
        # cv2.waitKey(0)

        return detections

    def _pre_process(self, image):
        if "roi" in self.preprocess_parameters:
            roi_params = self.preprocess_parameters["roi"]
            # Apply a mask via a mask file
            if "path" in roi_params:
                image = ih.apply_roi_file(image, roi_params["path"])
            # Apply a mask via a polyline
            elif "coords":
                image = ih.apply_roi_coords(image, roi_params["coords"])

        if "resize" in self.preprocess_parameters:
            resize_params = self.preprocess_parameters["resize"]
            image = ih.resize(image, resize_params["width"], resize_params["height"])

        if "border" in self.preprocess_parameters:
            border_params = self.preprocess_parameters["border"]
            image = ih.add_borders(image, centered=border_params["centered"])
        return image

    def _post_process(self, detections):
        if "nms" in self.postprocess_parameters:
            nms_params = self.postprocess_parameters["nms"]
            detections.nms_filter(nms_params["pref_implem"], nms_params["nms_thresh"])
        if "coi" in self.postprocess_parameters:
            detections.class_filter(self.postprocess_parameters["coi"])
        if "min_conf" in self.postprocess_parameters:
            detections.confidence_filter(self.postprocess_parameters["min_conf"])
        if "top_k" in self.postprocess_parameters:
            detections.top_k(self.postprocess_parameters["top_k"])
        return detections