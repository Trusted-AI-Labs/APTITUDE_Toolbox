from PIL import Image
import cv2
import time
import copy

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
    postproc_params["coi"] = {0, 3}
    postproc_params["min_conf"] = 0.5
    postproc_params["top_k"] = 5

    # image_path = "C:/Users/samelson/Pictures/img00230_keep.jpg"
    image_path = "C:/Users/samelson/Pictures/truck2_cam10.png"
    # pil_image = Image.open(image_path)
    cv2_image = cv2.imread(image_path)

    # Instantiate first configuration
    start = time.time()
    detection_manager1 = DetectionManager(DetectorFactory.create_detector(config_dict), preproc_params, postproc_params)
    # yolo1 = YOLO(config_dict)
    end = time.time()
    print("YOLO 1 init duration = " + str(end-start))

    # Instantiate second configuration
    config_dict2 = copy.deepcopy(config_dict)
    config_dict2["BBoxes2DDetector"]["pref_implem"] = "cv2-ReadNet"
    config_dict2["YOLO"] = {"conf_thresh": 0, "nms_thresh": 0}
    preproc_params2 = {}
    postproc_params2 = {}
    postproc_params2["nms"] = {"pref_implem" : "cv2", "nms_thresh" : 0.45, "conf_thresh" : 0.25}
    postproc_params2["coi"] = {0, 3}
    postproc_params2["min_conf"] = 0.5
    postproc_params2["top_k"] = 5

    start = time.time()
    detection_manager2 = DetectionManager(DetectorFactory.create_detector(config_dict2), preproc_params2, postproc_params2)
    # yolo2 = YOLO(config_dict)
    end = time.time()
    print("YOLO 2 init duration = " + str(end-start))
    
    # Test both configurations
    for i in range(10):
        start = time.time()
        res = detection_manager1.detect(cv2_image)
        end = time.time()
    res.image_path = image_path
    print(res)

    print("------------------------------------------")
    
    for i in range(10):
        start = time.time()
        res = detection_manager2.detect(cv2_image)
        end = time.time()
    res.image_path = image_path
    print(res)