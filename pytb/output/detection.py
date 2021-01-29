from abc import ABC, abstractmethod


class Detection(ABC):

    def __init__(self, number_objects: int):
        super().__init__()

        self.number_objects = number_objects

        self.image_path = None
        self.processing_time = 0
        self.preprocessing_time = 0
        self.postprocessing_time = 0

    def __str__(self):
        s = super().__str__()
        s += "\n\tImage path: " + str(self.image_path)
        s += "\n\tProcessing time: " + str(self.processing_time)
        s += "\n\tPre-processing time: " + str(self.preprocessing_time)
        s += "\n\tPost-processing time: " + str(self.postprocessing_time)
        s += "\n\tNumber of objects: " + str(self.number_objects)
        return s
