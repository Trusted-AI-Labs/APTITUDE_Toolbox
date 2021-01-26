from PIL import Image
from timeit import default_timer
import numpy as np
import cv2
import json

from pytb.detection.detection_manager import DetectionManager
from pytb.detection.detector_factory import DetectorFactory
import pytb.utils.image_helper as ih

if __name__ == "__main__":

    with open('configs/detect1.json') as config_file:
        config1 = json.load(config_file)

    config1_proc = config1['Proc']
    config1_preproc = config1['Preproc']
    config1_postproc = config1['Postproc']

    # image_path = "C:/Users/samelson/Pictures/img00230_stretch.jpg"
    # image_path = "C:/Users/samelson/Pictures/img00230_keep.jpg"
    # image_path = "C:/Users/samelson/Pictures/truck2_cam10.png"
    image_path = "C:/Users/samelson/Pictures/truck_cam4dawn.png"

    for i in range(100):
        start = default_timer()
        cv2_image = ih.get_cv2_img_from_str(image_path)
        end = default_timer()
        print("Reading time:", (end-start))

    # Instantiate first configuration
    start = default_timer()
    detection_manager1 = DetectionManager(DetectorFactory.create_detector(config1_proc), config1_preproc, config1_postproc)
    end = default_timer()
    print("YOLO 1 init duration = " + str(end-start))

    with open('configs/detect2.json') as config_file:
        config2 = json.load(config_file)

    config2_proc = config2['Proc']
    config2_preproc = config2['Preproc']
    config2_postproc = config2['Postproc']

    start = default_timer()
    detection_manager2 = DetectionManager(DetectorFactory.create_detector(config2_proc), config2_preproc, config2_postproc)
    end = default_timer()
    print("YOLO 2 init duration = " + str(end-start))
    
    # Test both configurations
    for i in range(10):
        start = default_timer()
        res = detection_manager1.detect(cv2_image)
        end = default_timer()
    res.image_path = image_path
    print(res)
    print("Total duration for 1 :", end-start)

    print("------------------------------------------")
    
    for i in range(10):
        start = default_timer()
        res = detection_manager2.detect(cv2_image)
        end = default_timer()
    res.image_path = image_path
    print(res)
    print("Total duration for 2 :", end-start)