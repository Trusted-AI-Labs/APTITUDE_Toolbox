from abc import ABC, abstractmethod

class Detection(ABC):

    def __init__(self, inference_time, completed):
        super().__init__()

        self.inference_time = inference_time
        self.completed = completed

        self.image_path = None
        self.processing_time = -1

    def __str__(self):
        s = super().__str__()
        s += "\n\tImage path: " + str(self.image_path)
        s += "\n\tProcessing time: " + str(self.processing_time)
        s += "\n\tInference time: " + str(self.inference_time)
        s += "\n\tCompleted: " + str(self.completed)
        return s

