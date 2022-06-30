"""
Copyright (c) 2021-2022 UCLouvain, ICTEAM
Licensed under GPL-3.0 [see LICENSE for details]
Written by Jonathan Samelson (2021-2022)
"""

from abc import ABC


class Detection(ABC):

    def __init__(self, number_objects: int):
        """An abstract class representing a Detection. It stores the number of detected objects,
        but also the preprocessing, processing and postprocessing time that are filled by the DetectionManager

        Args:
            number_objects (int): The number of detected objects.
        """
        super().__init__()

        self.number_objects = number_objects

        self.processing_time = 0
        self.preprocessing_time = 0
        self.postprocessing_time = 0

    def __str__(self):
        s = super().__str__()
        s += "\n\tProcessing time: " + str(self.processing_time)
        s += "\n\tPre-processing time: " + str(self.preprocessing_time)
        s += "\n\tPost-processing time: " + str(self.postprocessing_time)
        s += "\n\tNumber of objects: " + str(self.number_objects)
        return s
