from PIL import Image
from timeit import default_timer
import cv2
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
    preproc_params["border"] = {"centered": True}
    # preproc_params["roi"] = {"path": "C:/Users/samelson/Pictures/roi.jpg"}

    postproc_params = {}
    # postproc_params["coi"] = {0, 3}
    # postproc_params["min_conf"] = 0.5
    # postproc_params["top_k"] = 5

    # image_path = "C:/Users/samelson/Pictures/img00230_stretch.jpg"
    # image_path = "C:/Users/samelson/Pictures/img00230_keep.jpg"
    image_path = "C:/Users/samelson/Pictures/truck2_cam10.png"
    # image_path = "C:/Users/samelson/Pictures/truck_cam4dawn.png"

    cv2_image = cv2.imread(image_path)

    # Instantiate first configuration
    start = default_timer()
    detection_manager1 = DetectionManager(DetectorFactory.create_detector(config_dict), preproc_params, postproc_params)
    end = default_timer()
    print("YOLO 1 init duration = " + str(end-start))

    # Instantiate second configuration
    config_dict2 = copy.deepcopy(config_dict)
    config_dict2["BBoxes2DDetector"]["pref_implem"] = "cv2-ReadNet"
    config_dict2["YOLO"] = {"conf_thresh": 0, "nms_thresh": 0}

    preproc_params2 = {}
    preproc_params2["border"] = {"centered": False}
    # preproc_params2["roi"] = {"coords": [(0,0), (640, 0), (640, 200), (320,480), (0, 480)]}

    postproc_params2 = {}
    postproc_params2["nms"] = {"pref_implem" : "cv2", "nms_thresh" : 0.45, "conf_thresh" : 0.25}
    # postproc_params2["coi"] = {0, 3}
    # postproc_params2["min_conf"] = 0.5
    # postproc_params2["top_k"] = 5

    start = default_timer()
    detection_manager2 = DetectionManager(DetectorFactory.create_detector(config_dict2), preproc_params2, postproc_params2)
    end = default_timer()
    print("YOLO 2 init duration = " + str(end-start))
    
    # Test both configurations
    for i in range(1):
        start = default_timer()
        res = detection_manager1.detect(cv2_image)
        end = default_timer()
    res.image_path = image_path
    print(res)
    print("Total duration for 1 :", end-start)

    print("------------------------------------------")
    
    for i in range(1):
        start = default_timer()
        res = detection_manager2.detect(cv2_image)
        end = default_timer()
    res.image_path = image_path
    print(res)
    print("Total duration for 2 :", end-start)