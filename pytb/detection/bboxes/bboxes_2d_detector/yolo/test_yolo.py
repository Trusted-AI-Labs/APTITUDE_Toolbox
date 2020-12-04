from pytb.detection.bboxes.bboxes_2d_detector.yolo.yolo import YOLO

from PIL import Image
import time

if __name__ == "__main__":
    pil_image = Image.open("C:/Users/samelson/Pictures/img00230_keep.jpg")

    config_dict = {}
    config_dict["BBoxes2DDetector"] = {
        "pref_implem": "cv2-DetectionModel",
        "model_path": "E:\APTITUDE\ivy\data\detectors\yolo-v4-mio\yolov4-mio.cfg",
        "config_path": "E:\APTITUDE\ivy\data\detectors\yolo-v4-mio\yolov4-mio_99000.weights",
        "input_width" : 416,
        "input_height" : 416
    }
    config_dict["YOLO"] = {"conf_thresh": 0.25, "nms_thresh": 0.45}

    start = time.time()
    yolo1 = YOLO(config_dict)
    end = time.time()
    print("YOLO 1 init duration = " + str(end-start))

    config_dict["BBoxes2DDetector"]["pref_implem"] = "cv2-ReadNet"

    start = time.time()
    yolo2 = YOLO(config_dict)
    end = time.time()
    print("YOLO 2 init duration = " + str(end-start))
    
    delay = 100 
    for i in range(100):
        start = time.time()
        res = yolo1.detect(pil_image)
        end = time.time()
        delay = min(delay, (end-start))
    print(res)
    print("YOLO 1 detect duration = " + str(delay))
    
    delay = 100
    for i in range(100):
        start = time.time()
        res = yolo2.detect(pil_image)
        end = time.time()
        delay = min(delay, (end-start))
    print(res)
    print("YOLO 2 detect duration = " + str(delay))
    
