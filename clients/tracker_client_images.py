from timeit import default_timer
import numpy as np
import cv2
import json
import time
import os

from tkinter import Tcl
from pytb.tracking.tracking_manager import TrackingManager
from pytb.tracking.tracking_factory import TrackingFactory
from pytb.output.bboxes_2d import BBoxes2D
import pytb.utils.image_helper as ih
from pytb.detection.detection_manager import DetectionManager
from pytb.detection.detector_factory import DetectorFactory

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(10000, 3), dtype="uint8")
font = cv2.FONT_HERSHEY_DUPLEX
line_type = cv2.LINE_AA
thickness = 2


if __name__ == "__main__":

    with open('configs/detect-DM.json') as config_file:
        detect1 = json.load(config_file)

    detect1_proc = detect1['Proc']
    detect1_preproc = detect1['Preproc']
    detect1_postproc = detect1['Postproc']

    with open('configs/track-deepsort.json') as config_file:
        track1 = json.load(config_file)

    track1_proc = track1['Proc']
    track1_preproc = track1['Preproc']
    track1_postproc = track1['Postproc']

    with open('configs/classes.json') as config_file:
        CLASSES = json.load(config_file)['classes']

    # Instantiate first configuration
    start = default_timer()
    detection_manager = DetectionManager(DetectorFactory.create_detector(detect1_proc), detect1_preproc, detect1_postproc)
    end = default_timer()
    print("Detector init duration = " + str(end-start))

    start = default_timer()
    tracking_manager = TrackingManager(TrackingFactory.create_tracker(track1_proc), track1_preproc, track1_postproc)
    end = default_timer()
    print("Tracker init duration = " + str(end-start))

    folder_path = "E:\samelson\_Dataset\Digital Twin\cam1"

    file_list = os.listdir(folder_path)
    file_list_sorted = Tcl().call('lsort', '-dict', file_list)

    read_time = 0
    start_time = default_timer()
    before_loop = start_time
    counter = 0
    frame_interval = 4
    fps_number_frames = 10
    is_paused = False

    for i, image_name in enumerate(file_list_sorted):
        if not image_name.endswith(".jpg"):
            continue

        k = cv2.waitKey(1) & 0xFF
        if k == ord('p'): # pause/play loop if 'p' key is pressed
            is_paused = not is_paused
        if k == ord('q'): # end video loop if 'q' key is pressed
            break

        if is_paused: 
            time.sleep(0.5)
            continue


        read_time_start = default_timer()
        frame = ih.get_cv2_img_from_str(os.path.join(folder_path, image_name))
        read_time += default_timer() - read_time_start

        counter += 1
        if counter % frame_interval != 0:
            continue

        # frame = ih.resize(frame, 960, 540)
        (H, W, _) = frame.shape

        det = detection_manager.detect(frame)
        res = tracking_manager.track(det, frame)
        
        # Visualize
        res.to_x1_y1_x2_y2()
        res.change_dims(W, H)

        for i in range(res.number_objects):
            id = res.global_IDs[i]
            # id = 1
            color = [int(c) for c in COLORS[id]]
            vehicle_label = 'I: {0}, T: {1} ({2})'.format(id, CLASSES[res.class_IDs[i]], str(res.det_confs[i])[:4])
            cv2.rectangle(frame, (res.bboxes[i][0], res.bboxes[i][1]), (res.bboxes[i][2], res.bboxes[i][3]), color, thickness)
            cv2.putText(frame, vehicle_label, (res.bboxes[i][0], res.bboxes[i][1]- 5), font, 1, color, thickness, line_type)
        
        cv2.imshow("Result", frame)

        if counter % fps_number_frames == 0:
            print("FPS:", fps_number_frames/(default_timer()-start_time))
            start_time = default_timer()

    print("Average FPS:", counter/(default_timer()-before_loop))
    print("Average FPS w/o read time:", counter/(default_timer()-before_loop-read_time))

    cv2.destroyAllWindows()