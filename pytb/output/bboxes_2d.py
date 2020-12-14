from pytb.output.detection import Detection

class BBoxes_2D(Detection):

    def __init__(self, inference_time, completed, 
                bboxes, class_IDs, det_confs, dim_width, dim_height):
        super().__init__(inference_time, completed)

        self.bboxes = bboxes
        self.class_IDs = class_IDs
        self.det_confs = det_confs

        self.dim_width = dim_width
        self.dim_height = dim_height
        
        self.prev_track_IDs = None

    def __str__(self):
        s = super().__str__()
        s += "\n\tclass IDs: " + str(self.class_IDs)
        s += "\n\tDetection confidences: " + str(self.det_confs)
        s += "\n\tBounding Boxes: " + str(self.bboxes)
        s += "\n\tPrevious track IDs : " + str(self.prev_track_IDs)
        return s


