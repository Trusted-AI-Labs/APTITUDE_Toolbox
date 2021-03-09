import json
import os
import time
from timeit import default_timer
from tkinter import Tcl
import csv

import cv2
import numpy as np

import pytb.utils.image_helper as ih
from pytb.detection.detection_manager import DetectionManager
from pytb.detection.detector_factory import DetectorFactory
from pytb.tracking.tracker_factory import TrackerFactory
from pytb.tracking.tracking_manager import TrackingManager

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(10000, 3), dtype="uint8")
font = cv2.FONT_HERSHEY_DUPLEX
line_type = cv2.LINE_AA
thickness = 2


def main(cfg_detect, cfg_track, cfg_classes, folder_path, frame_interval, show_fps=False):
    with open(cfg_detect) as config_file:
        detect1 = json.load(config_file)

    detect1_proc = detect1['Proc']
    detect1_preproc = detect1['Preproc']
    detect1_postproc = detect1['Postproc']

    with open(cfg_track) as config_file:
        track1 = json.load(config_file)

    track1_proc = track1['Proc']
    track1_preproc = track1['Preproc']
    track1_postproc = track1['Postproc']

    with open(cfg_classes) as config_file:
        CLASSES = json.load(config_file)['classes']

    # Instantiate first configuration
    start = default_timer()
    detection_manager = DetectionManager(DetectorFactory.create_detector(detect1_proc),
                                         detect1_preproc, detect1_postproc)
    end = default_timer()
    print("Detector init duration = " + str(end - start))

    start = default_timer()
    tracking_manager = TrackingManager(TrackerFactory.create_tracker(track1_proc),
                                       track1_preproc, track1_postproc)
    end = default_timer()
    print("Tracker init duration = " + str(end - start))

    file_list = os.listdir(folder_path)
    file_list_sorted = Tcl().call('lsort', '-dict', file_list)

    warmup_time = 0
    read_time = 0
    start_time = default_timer()
    before_loop = start_time
    counter = 0
    fps_number_frames = 10
    is_paused = False

    for image_name in file_list_sorted:
        if not image_name.endswith(".jpg"):
            continue

        k = cv2.waitKey(1) & 0xFF
        if k == ord('p'):  # pause/play loop if 'p' key is pressed
            is_paused = not is_paused
        if k == ord('q'):  # end video loop if 'q' key is pressed
            break

        if is_paused:
            time.sleep(0.5)
            continue

        counter += 1
        if counter % frame_interval != 0:
            continue

        read_time_start = default_timer()
        frame = ih.get_cv2_img_from_str(os.path.join(folder_path, image_name))
        read_time += default_timer() - read_time_start


        (H, W, _) = frame.shape

        warmup_time_sart = default_timer()
        det = detection_manager.detect(frame)
        if tracking_manager.tracker.need_frame:
            res = tracking_manager.track(det, frame)
        else: 
            res = tracking_manager.track(det)
        if counter <= 5:
            warmup_time += default_timer() - warmup_time_sart

        # Add ground-truth if present
        base_name = image_name[:-4]
        csv_file_name = os.path.join(folder_path, base_name + ".csv")
        if os.path.exists(csv_file_name):
            print(csv_file_name)
            with open(csv_file_name, 'r') as read_obj:
                csv_reader = csv.reader(read_obj, delimiter=";")
                for i, row in enumerate(csv_reader):
                    if i == 0 or row[6] == "-1":
                        continue
                    top_x, top_y = W, H
                    bot_x, bot_y = 0, 0
                    for j in range(44, 60, 2):
                        x, y = int(row[j]), int(row[j+1])
                        if x != -1 and x <= top_x and y != -1 and y <= top_y:
                            top_x, top_y = x, y
                        if x != -1 and x >= bot_x and y != -1 and y >= bot_y:
                            bot_x, bot_y = x, y
                    # print((top_x, top_y, bot_x, bot_y))
                    cv2.rectangle(frame, (top_x, top_y), (bot_x, bot_y), (255, 255, 255), thickness)
                    # for j in range(44, 60, 2):
                    #     x, y = int(row[j]), int(row[j+1])
                    #     cv2.line(frame, (x, y), (x, y), (255, 255, 255), thickness)


        # Visualize
        res.to_x1_y1_x2_y2()
        res.change_dims(W, H)

        for i in range(res.number_objects):
            id = res.global_IDs[i]
            # id = 1
            color = [int(c) for c in COLORS[id]]
            vehicle_label = 'I: {0}, T: {1} ({2})'.format(id, CLASSES[res.class_IDs[i]], str(res.det_confs[i])[:4])
            cv2.rectangle(frame, (res.bboxes[i][0], res.bboxes[i][1]), (res.bboxes[i][2], res.bboxes[i][3]), color,
                          thickness)
            cv2.putText(frame, vehicle_label, (res.bboxes[i][0], res.bboxes[i][1] - 5), font, 1, color, thickness,
                        line_type)

        cv2.imshow("Result", frame)

        if show_fps and counter % fps_number_frames == 0:
            print("FPS:", fps_number_frames / (default_timer() - start_time))
            start_time = default_timer()

    print("Average FPS:", counter / (default_timer() - before_loop - warmup_time))
    print("Average FPS w/o read time:", counter / (default_timer() - before_loop - read_time - warmup_time))

    cv2.destroyAllWindows()
