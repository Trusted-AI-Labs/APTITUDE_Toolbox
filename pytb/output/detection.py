from abc import ABC, abstractmethod

class Detection(ABC):

    def __init__(self, inference_time, number_detections):
        super().__init__()

        self.inference_time = inference_time
        self.number_detections = number_detections

        self.image_path = None
        self.processing_time = -1
        self.preprocessing_time = -1
        self.postprocessing_time = -1

    def __str__(self):
        s = super().__str__()
        s += "\n\tImage path: " + str(self.image_path)
        s += "\n\tProcessing time: " + str(self.processing_time)
        s += "\n\t\tInference time: " + str(self.inference_time)
        s += "\n\tPre-processing time: " + str(self.preprocessing_time)
        s += "\n\tPost-processing time: " + str(self.postprocessing_time)
        s += "\n\tNumber of detections: " + str(self.number_detections)
        return s

