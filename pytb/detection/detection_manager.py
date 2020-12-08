

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
        output = self.detector.detect(org_frame)
        output.preprocessing_time = preproc_time
        output.processing_time = time()-start

        start = time()
        # TODO apply postprocess parameters
        output.postprocessing_time = time()-start

        return output