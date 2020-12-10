from pytb.output.bboxes_2d import BBoxes_2D

class BBoxes_2D_Track(BBoxe_2D):

    def __init__(self, inference_time, completed, 
                bboxes, class_IDs, det_conf, 
                global_IDs, track_conf):
        super().__init__(inference_time, completed, bboxes, class_IDs, det_conf)

        self.global_IDs = global_IDs
        self.track_conf = track_conf

    def __str__(self):
        s = super().__str__()
        s += "\n\tGlobal IDs: " + str(self.global_IDs)
        s += "\n\ttrack confidence: " + str(self.track_conf)

