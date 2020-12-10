from pytb.utils.detection_filter import DetectionFilter

from time import time

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
        start = time()
        # TODO apply preprocess parameters
        preproc_time = time()-start

        # call the concrete method of the detector
        start = time()
        detections = self.detector.detect(org_frame)
        detections.processing_time = time()-start
        detections.preprocessing_time = preproc_time

        # Post process
        start = time()
        detections = self._post_process(detections)
        detections.postprocessing_time = time()-start

        return detections

    def _post_process(self, detections):
        if "nms" in self.postprocess_parameters:
            nms_params = self.postprocess_parameters["nms"]
            DetectionFilter.nms_filter(detections, nms_params["pref_implem"], nms_params["nms_thresh"], nms_params["conf_thresh"])
        if "coi" in self.postprocess_parameters:
            DetectionFilter.class_filter(detections, self.postprocess_parameters["coi"])
        if "min_conf" in self.postprocess_parameters:
            DetectionFilter.confidence_filter(detections, self.postprocess_parameters["min_conf"])
        if "top_k" in self.postprocess_parameters:
            DetectionFilter.top_k(detections, self.postprocess_parameters["top_k"])
        return detections