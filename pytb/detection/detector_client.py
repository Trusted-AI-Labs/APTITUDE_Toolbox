from PIL import Image
import time

from pytb.detection.detection_manager import DetectionManager
from pytb.detection.detector_factory import DetectorFactory

if __name__ == "__main__":
    
    config_dict = {}
    config_dict["Detector"] = {
        "type": "BBoxes2DDetector"
    }
    config_dict["BBoxes2DDetector"] = {
        "model": "YOLO",
        "pref_implem": "cv2-DetectionModel",
        "model_path": "E:\APTITUDE\ivy\data\detectors\yolo-v4-mio\yolov4-mio.cfg",
        "config_path": "E:\APTITUDE\ivy\data\detectors\yolo-v4-mio\yolov4-mio_99000.weights",
        "input_width" : 416,
        "input_height" : 416
    }
    config_dict["YOLO"] = {"conf_thresh": 0.25, "nms_thresh": 0.45}
    
    preproc_params = {}
    postproc_params = {}

    image_path = "C:/Users/samelson/Pictures/img00230_keep.jpg"
    pil_image = Image.open(image_path)

    # Instantiate first configuration
    start = time.time()
    detection_manager1 = DetectionManager(DetectorFactory.create_detector(config_dict), preproc_params, postproc_params)
    # yolo1 = YOLO(config_dict)
    end = time.time()
    print("YOLO 1 init duration = " + str(end-start))

    # Instantiate second configuration
    config_dict["BBoxes2DDetector"]["pref_implem"] = "cv2-ReadNet"

    start = time.time()
    detection_manager2 = DetectionManager(DetectorFactory.create_detector(config_dict), preproc_params, postproc_params)
    # yolo2 = YOLO(config_dict)
    end = time.time()
    print("YOLO 2 init duration = " + str(end-start))
    
    # Test both configurations
    delay = 100 
    for i in range(100):
        start = time.time()
        res = detection_manager1.detect(pil_image)
        end = time.time()
        delay = min(delay, (end-start))
    res.image_path = image_path
    print(res)
    print("YOLO 1 detect min duration = " + str(delay))
    
    delay = 100
    for i in range(100):
        start = time.time()
        res = detection_manager2.detect(pil_image)
        end = time.time()
        delay = min(delay, (end-start))
    res.image_path = image_path
    print(res)
    print("YOLO 2 detect min duration = " + str(delay))