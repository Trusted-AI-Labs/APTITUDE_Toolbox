import json
import os
from timeit import default_timer
from tkinter import Tcl
from tqdm import tqdm
import csv

import cv2
import numpy as np
import logging

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


def main(cfg_detect, cfg_track, cfg_classes, folder_path, frame_interval, record_path, record_fps,
         mot_path, headless, gt_path):

    log = logging.getLogger("aptitude-toolbox")

    with open(cfg_detect) as config_file:
        detect1 = json.load(config_file)
        log.debug("Detector config loaded.")

    detect1_proc = detect1['Proc']
    detect1_preproc = detect1['Preproc']
    detect1_postproc = detect1['Postproc']

    with open(cfg_track) as config_file:
        track1 = json.load(config_file)
        log.debug("Tracker config loaded.")

    track1_proc = track1['Proc']
    track1_preproc = track1['Preproc']
    track1_postproc = track1['Postproc']

    with open(cfg_classes) as config_file:
        CLASSES = json.load(config_file)['classes']
        log.debug("Classes config loaded.")

    # Instantiate detector
    start = default_timer()
    detection_manager = DetectionManager(DetectorFactory.create_detector(detect1_proc),
                                         detect1_preproc, detect1_postproc)
    end = default_timer()
    log.info("Detector init duration = {}s".format(str(end - start)))

    # Instantiate tracker
    start = default_timer()
    tracking_manager = TrackingManager(TrackerFactory.create_tracker(track1_proc),
                                       track1_preproc, track1_postproc)
    end = default_timer()
    log.info("Tracker init duration = {}s".format(str(end - start)))

    # Get sequence, the list of images
    included_extensions = ['jpg', 'jpeg', 'bmp', 'png', 'gif']
    file_list = [f for f in os.listdir(folder_path)
                 if any(f.endswith(ext) for ext in included_extensions)]
    file_list_sorted = Tcl().call('lsort', '-dict', file_list)
    if len(file_list_sorted) > 0:
        log.debug("Image sequence sorted.")
    else:
        log.error("Image list is empty, check if the input path is correct.")

    record = record_path is not None
    frame_test = ih.get_cv2_img_from_str(os.path.join(folder_path, file_list[0]))
    H, W, _ = frame_test.shape
    if record:
        output_video = cv2.VideoWriter(record_path, cv2.VideoWriter_fourcc(*'mp4v'), record_fps, (W, H))
        log.debug("VideoWriter opened successfully.")

    if gt_path is not None and not os.path.isdir(gt_path):
        with open(gt_path, 'r') as gt:
            gt_lines = gt.readlines()
            gt_line_number = 0
            gt_frame_num = 1  # 1 for RGB, 2 for residues
            log.debug("Ground truth file open successfully.")

    warmup_time = 0
    read_time = 0
    start_time = default_timer()
    last_update = default_timer()
    before_loop = start_time
    counter = 0
    is_paused = False

    output_lines = []

    for image_name in tqdm(file_list_sorted):

        time_update = default_timer()
        if not headless and time_update - last_update > (1/10):
            k = cv2.waitKey(1) & 0xFF
            if k == ord('p'):  # pause/play loop if 'p' key is pressed
                log.debug("Process paused/resumed.")
                is_paused = not is_paused
            if k == ord('q'):  # end video loop if 'q' key is pressed
                log.info("Process exited.")
                break
            last_update = time_update

        while is_paused:
            k = cv2.waitKey(1) & 0xFF
            if k == ord('p'):  # pause/play loop if 'p' key is pressed
                is_paused = False

        if counter % frame_interval != 0:
            continue

        read_time_start = default_timer()
        frame = ih.get_cv2_img_from_str(os.path.join(folder_path, image_name))
        read_time += default_timer() - read_time_start
        log.debug("Image read successfully.")

        (H, W, _) = frame.shape

        warmup_time_start = default_timer()
        log.debug("Before detection.")
        det = detection_manager.detect(frame)
        log.debug("After detection & before tracking.")
        if tracking_manager.tracker.need_frame:
            res = tracking_manager.track(det, frame)
            log.debug("After tracking, with frame.")
        else:
            res = tracking_manager.track(det)
            log.debug("After tracking, without frames.")
        if counter <= 5:
            warmup_time += default_timer() - warmup_time_start

        # Visualize
        res.change_dims(W, H)
        log.debug("Dimensions of the results changed: (W: {}, H:{}).".format(W, H))
        # print(res)

        # Add to output file
        if mot_path is not None:
            for i in range(res.number_objects):
                # counter + 1 for RGB, +2 for residues
                results = "{0},{1},{2},{3},{4},{5},-1,-1,-1,-1\n".format(counter + 1, res.global_IDs[i],
                                                                         res.bboxes[i][0], res.bboxes[i][1],
                                                                         res.bboxes[i][2], res.bboxes[i][3])
                log.debug(results)
                output_lines.append(results)
                log.debug("Results added to output path.")

        res.to_x1_y1_x2_y2()
        log.debug("Results converted to x1,y1,x2,y2.")

        if gt_path is not None:
            if os.path.isdir(gt_path):  # For CSV files from Digital Twin project
                frame = add_ground_truths(frame, image_name, gt_path, W, H)
            else:
                # counter+1 for RGB, +2 for residues
                while gt_frame_num == counter+1 and gt_line_number < len(gt_lines):
                    line = gt_lines[gt_line_number]
                    new_gt_frame_num, id, left, top, width, height, _, _, _, _ = line.split(",")
                    new_gt_frame_num, id, left, top, width, height = round(new_gt_frame_num), round(id), round(left), round(top), \
                                                                     round(width), round(height)
                    if new_gt_frame_num > gt_frame_num:
                        gt_frame_num = new_gt_frame_num
                        # Don't increment gt_line_number
                        break
                    cv2.rectangle(frame, (left, top), (left + width, top + height), (255, 255, 255), 2)
                    # cv2.putText(frame, str(id), (left, top - 5), font, 1, (255, 255, 255), 2, line_type)
                    gt_line_number += 1
            log.debug("Ground truth bounding boxes added to the image.")

        for i in range(res.number_objects):
            id = res.global_IDs[i]
            color = [int(c) for c in COLORS[id]]
            vehicle_label = 'I: {0}, T: {1} ({2})'.format(id, CLASSES[res.class_IDs[i]], str(res.det_confs[i])[:4])
            cv2.rectangle(frame, (round(res.bboxes[i][0]), round(res.bboxes[i][1])),
                            (round(res.bboxes[i][2]), round(res.bboxes[i][3])), color, thickness)
            cv2.putText(frame, vehicle_label, (round(res.bboxes[i][0]), round(res.bboxes[i][1] - 5)),
                        font, 1, color, thickness, line_type)
        log.debug("Results bounding boxes added to the image.")

        if not headless:
            cv2.imshow("Result", frame)
            log.debug("Frame displayed.")
        if record:
            output_video.write(frame)
            log.debug("Frame written to VideoWriter.")

        counter += 1


    if mot_path is not None:
        with open(mot_path, "w") as out:
            out.writelines(output_lines)

    log.info("Average FPS: {}".format(str(counter / (default_timer() - before_loop - warmup_time))))
    log.info("Average FPS w/o read time: {}".format(str(counter / (default_timer() - before_loop - read_time - warmup_time))))

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
