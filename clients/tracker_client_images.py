import json
import os
from timeit import default_timer
from tkinter import Tcl
from tqdm import tqdm
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


def main(cfg_detect, cfg_track, cfg_classes, folder_path, frame_interval, record_path, record_fps, headless,
         show_fps, gt_folder_path):

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

    # Instantiate detector
    start = default_timer()
    detection_manager = DetectionManager(DetectorFactory.create_detector(detect1_proc),
                                         detect1_preproc, detect1_postproc)
    end = default_timer()
    print("Detector init duration = " + str(end - start))

    # Instantiate tracker
    start = default_timer()
    tracking_manager = TrackingManager(TrackerFactory.create_tracker(track1_proc),
                                       track1_preproc, track1_postproc)
    end = default_timer()
    print("Tracker init duration = " + str(end - start))

    # Get sequence, the list of images
    included_extensions = ['jpg', 'jpeg', 'bmp', 'png', 'gif']
    file_list = [f for f in os.listdir(folder_path)
                 if any(f.endswith(ext) for ext in included_extensions)]
    file_list_sorted = Tcl().call('lsort', '-dict', file_list)

    record = record_path is not None
    frame_test = ih.get_cv2_img_from_str(os.path.join(folder_path, file_list[0]))
    H, W, _ = frame_test.shape
    if record:
        output_video = cv2.VideoWriter(record_path, cv2.VideoWriter_fourcc(*'mp4v'), record_fps, (W, H))

    warmup_time = 0
    read_time = 0
    start_time = default_timer()
    before_loop = start_time
    counter = 0
    fps_number_frames = 10
    is_paused = False

    for image_name in tqdm(file_list_sorted):
        if not image_name.endswith(".jpg"):
            continue

        k = cv2.waitKey(1) & 0xFF
        if k == ord('p'):  # pause/play loop if 'p' key is pressed
            is_paused = True
        if k == ord('q'):  # end video loop if 'q' key is pressed
            break

        while is_paused:
            k = cv2.waitKey(1) & 0xFF
            if k == ord('p'):  # pause/play loop if 'p' key is pressed
                is_paused = False

        counter += 1
        if counter % frame_interval != 0:
            continue

        read_time_start = default_timer()
        frame = ih.get_cv2_img_from_str(os.path.join(folder_path, image_name))
        read_time += default_timer() - read_time_start

        (H, W, _) = frame.shape

        warmup_time_start = default_timer()
        det = detection_manager.detect(frame)
        if tracking_manager.tracker.need_frame:
            res = tracking_manager.track(det, frame)
        else: 
            res = tracking_manager.track(det)
        if counter <= 5:
            warmup_time += default_timer() - warmup_time_start


        # Visualize
        res.to_x1_y1_x2_y2()
        res.change_dims(W, H)
        # print(res)

        if gt_folder_path is not None:
            frame = add_ground_truths(frame, image_name, gt_folder_path, W, H)

        for i in range(res.number_objects):
            id = res.global_IDs[i]
            # id = 1
            color = [int(c) for c in COLORS[id]]
            vehicle_label = 'I: {0}, T: {1} ({2})'.format(id, CLASSES[res.class_IDs[i]], str(res.det_confs[i])[:4])
            cv2.rectangle(frame, (res.bboxes[i][0], res.bboxes[i][1]), (res.bboxes[i][2], res.bboxes[i][3]), color,
                          thickness)
            cv2.putText(frame, vehicle_label, (res.bboxes[i][0], res.bboxes[i][1] - 5), font, 1, color, thickness,
                        line_type)
        if not headless:
            cv2.imshow("Result", frame)
        if record:
            output_video.write(frame)

        if show_fps and counter % fps_number_frames == 0:
            print("FPS:", fps_number_frames / (default_timer() - start_time))
            start_time = default_timer()

    print("Average FPS:", counter / (default_timer() - before_loop - warmup_time))
    print("Average FPS w/o read time:", counter / (default_timer() - before_loop - read_time - warmup_time))

    if record:
        output_video.release()
    cv2.destroyAllWindows()


def add_ground_truths(frame, image_name, gt_folder_path, W, H):
    base_name = image_name[:-4]
    csv_file_name = os.path.join(gt_folder_path, base_name + ".csv")
    if os.path.exists(csv_file_name):
        with open(csv_file_name, 'r') as read_obj:
            csv_reader = csv.reader(read_obj, delimiter=";")
            for i, row in enumerate(csv_reader):
                if i == 0 or row[6] == "-1":
                    continue
                max_x, max_y, min_x, min_y = 0, 0, W, H
                for j in range(44, 60, 2):
                    x, y = int(row[j]), int(row[j + 1])
                    if x == -1 or y == -1:
                        continue
                    if x < min_x:
                        min_x = x
                    if y < min_y:
                        min_y = y
                    if x > max_x:
                        max_x = x
                    if y > max_y:
                        max_y = y
                cv2.rectangle(frame, (max_x, max_y), (min_x, min_y), (255, 255, 255), thickness)
                # for j in range(44, 60, 2):
                #     x, y = int(row[j]), int(row[j+1])
                #     cv2.line(frame, (x, y), (x, y), (255, 255, 255), thickness)

    return frame
