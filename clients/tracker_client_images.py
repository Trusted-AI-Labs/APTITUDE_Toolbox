"""
Copyright (c) 2021-2022 UCLouvain, ICTEAM
Licensed under GPL-3.0 [see LICENSE for details]
Written by Jonathan Samelson (2021-2022)
"""

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


def main(cfg_detect, cfg_track, cfg_classes, folder_path, frame_interval, record_path, record_fps, record_size,
         mot_path, headless, display_size, gt_path):

    log = logging.getLogger("aptitude-toolbox")

    # Get parameters of the different stages of the detection process
    with open(cfg_detect) as config_file:
        detect1 = json.load(config_file)
        log.debug("Detector config loaded.")

    detect1_proc = detect1['proc']
    detect1_preproc = detect1['preproc']
    detect1_postproc = detect1['postproc']

    # Get parameters of the different stages of the tracking process
    with open(cfg_track) as config_file:
        track1 = json.load(config_file)
        log.debug("Tracker config loaded.")

    track1_proc = track1['proc']
    track1_preproc = track1['preproc']
    track1_postproc = track1['postproc']

    # Get the classes of the object to be detected
    with open(cfg_classes) as config_file:
        CLASSES = json.load(config_file)['classes']
        log.debug("Classes config loaded.")

    # Instantiate the detector
    start = default_timer()
    detection_manager = DetectionManager(DetectorFactory.create_detector(detect1_proc),
                                         detect1_preproc, detect1_postproc)
    end = default_timer()
    log.info("Detector init duration = {}s".format(str(end - start)))

    # Instantiate the tracker
    start = default_timer()
    tracking_manager = TrackingManager(TrackerFactory.create_tracker(track1_proc),
                                       track1_preproc, track1_postproc)
    end = default_timer()
    log.info("Tracker init duration = {}s".format(str(end - start)))

    # Get the sequence, the list of images in the correct order
    included_extensions = ['jpg', 'jpeg', 'bmp', 'png', 'gif']
    file_list = [f for f in os.listdir(folder_path)
                 if any(f.endswith(ext) for ext in included_extensions)]
    file_list_sorted = Tcl().call('lsort', '-dict', file_list)
    if len(file_list_sorted) > 0:
        log.debug("Image sequence sorted.")
    else:
        log.error("Image list is empty, check if the input path is correct.")

    # Get the first image to initialize the video dimension if not specified
    record = record_path is not None
    frame_test = ih.get_cv2_img_from_str(os.path.join(folder_path, file_list[0]))
    H, W, _ = frame_test.shape
    if record:
        if record_size is None:
            record_size = (H, W)
        output_video = cv2.VideoWriter(record_path, cv2.VideoWriter_fourcc(*'mp4v'), record_fps, record_size)
        log.debug("VideoWriter opened successfully.")

    # Read and initialize values to display the ground truth
    if gt_path is not None and not os.path.isdir(gt_path):
        with open(gt_path, 'r') as gt:
            gt_lines = gt.readlines()
            gt_line_number = 0
            gt_frame_num = 1
            log.debug("Ground truth file open successfully.")

    warmup_time = 0
    read_time = 0
    start_time = default_timer()
    last_update = default_timer()
    before_loop = start_time
    counter = 0
    tot_det_time = 0
    tot_track_time = 0
    is_paused = False

    output_lines = []

    for image_name in tqdm(file_list_sorted):

        # Check if a key was pressed but with some delay, as it resource consuming
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

        # Frame interval can be used to skip frames.
        # Skip the frame is the counter is not divisible by frame_interval
        if counter % frame_interval != 0:
            continue

        # Read the frame and measure elapsed time
        read_time_start = default_timer()
        frame = ih.get_cv2_img_from_str(os.path.join(folder_path, image_name))
        read_time += default_timer() - read_time_start
        log.debug("Image read successfully.")

        (H, W, _) = frame.shape

        # Proceed to the detection & tracking
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
        # Don't count the time of the first 5 iteration if we consider there is a 'warmup time'
        if counter <= 5:
            warmup_time += default_timer() - warmup_time_start

        tot_det_time += det.detection_time
        tot_track_time += res.tracking_time

        # Change dimensions of the result to match to the initial dimension of the frame
        res.change_dims(W, H)
        log.debug("Dimensions of the results changed: (W: {}, H:{}).".format(W, H))
        # print(res)

        # Add the result to the output file in MOT format
        if mot_path is not None:
            for i in range(res.number_objects):
                results = "{0},{1},{2},{3},{4},{5},-1,-1,-1,-1\n".format(counter + 1, res.global_IDs[i],
                                                                         res.bboxes[i][0], res.bboxes[i][1],
                                                                         res.bboxes[i][2], res.bboxes[i][3])
                log.debug(results)
                output_lines.append(results)
                log.debug("Results added to output path.")

        res.to_x1_y1_x2_y2()
        log.debug("Results converted to x1,y1,x2,y2.")

        # Add GT bboxes from GT file (MOT format) to the frame
        if gt_path is not None:
            # If the path is a directory, read the format of CSV files from Digital Twin project
            if os.path.isdir(gt_path):
                frame = add_ground_truths_DT(frame, image_name, gt_path, W, H)
            else:
                while gt_frame_num == counter+1 and gt_line_number < len(gt_lines):
                    line = gt_lines[gt_line_number]
                    new_gt_frame_num, id, left, top, width, height, _, _, _, _ = line.split(",")
                    new_gt_frame_num, id, left, top, width, height = round(new_gt_frame_num), round(id), round(left), round(top), \
                                                                     round(width), round(height)
                    if new_gt_frame_num > gt_frame_num:
                        gt_frame_num = new_gt_frame_num
                        # Don't increment gt_line_number
                        break

                    # Draw a white rectangle for each bbox
                    cv2.rectangle(frame, (left, top), (left + width, top + height), (255, 255, 255), 2)
                    # Write a text with the ground truth ID
                    # cv2.putText(frame, str(id), (left, top - 5), font, 1, (255, 255, 255), 2, line_type)
                    gt_line_number += 1
            log.debug("Ground truth bounding boxes added to the image.")

        # Add the bboxes from the process to the frame
        for i in range(res.number_objects):
            id = res.global_IDs[i]
            color = [int(c) for c in COLORS[id]]
            vehicle_label = 'I: {0}, T: {1} ({2})'.format(id, CLASSES[res.class_IDs[i]], str(res.det_confs[i])[:4])

            # Draw a rectangle (with a random color) for each bbox
            cv2.rectangle(frame, (round(res.bboxes[i][0]), round(res.bboxes[i][1])),
                            (round(res.bboxes[i][2]), round(res.bboxes[i][3])), color, thickness)

            # Write a text with the vehicle label, the confidence score and the ID
            cv2.putText(frame, vehicle_label, (round(res.bboxes[i][0]), round(res.bboxes[i][1] - 5)),
                        font, 1, color, thickness, line_type)
        log.debug("Results bounding boxes added to the image.")

        # If headless flag is absent, show the results of the process in a dedicated window
        if not headless:
            frame_display = frame
            if display_size is not None:
                frame_display = cv2.resize(frame_display, display_size)
                log.debug("Frame resized for display")
            cv2.imshow("Result", frame_display)
            log.debug("Frame displayed.")

        # If record flag is present, record the resulting frame on the disk
        if record:
            frame_record = frame
            if record_size is not None:
                frame_record = cv2.resize(frame_record, record_size)
                log.debug("Frame resized for record")
            output_video.write(frame_record)
            log.debug("Frame written to VideoWriter.")

        counter += 1

    # If MOT path flag is present, write all the lines at once to save on the disk
    if mot_path is not None:
        with open(mot_path, "w") as out:
            out.writelines(output_lines)

    # Display elapsed time (without taking a warm-up time into account)
    log.info("Average FPS: {}".format(str(counter / (default_timer() - before_loop))))
    log.info("Average FPS w/o read time: {}".format(str(counter / (default_timer() - before_loop - read_time))))

    log.info("Total detection time: {}".format(tot_det_time))
    log.info("Total tracking time: {}".format(tot_track_time))

    log.info("Average detection time: {}".format(tot_det_time / counter))
    log.info("Average tracking time: {}".format(tot_track_time / counter))

    if record:
        output_video.release()
    cv2.destroyAllWindows()

# This function allows to read the GT bboxes from the CSV of the Digital Twin project.
def add_ground_truths_DT(frame, image_name, gt_folder_path, W, H):
    base_name = image_name[:-4]
    csv_file_name = os.path.join(gt_folder_path, base_name + ".csv")
    if os.path.exists(csv_file_name):
        with open(csv_file_name, 'r') as read_obj:
            csv_reader = csv.reader(read_obj, delimiter=";")
            next(csv_reader)  # Ignore header
            for i, row in enumerate(csv_reader):
                max_x, max_y, min_x, min_y = 0, 0, W, H
                # Step is 2 because we read two cells at once
                for j in range(44, 60, 2):
                    x, y = int(row[j]), int(row[j + 1])
                    # Take the largest possible rectangle
                    if x < min_x:
                        min_x = x
                    if y < min_y:
                        min_y = y
                    if x > max_x:
                        max_x = x
                    if y > max_y:
                        max_y = y
                cv2.rectangle(frame, (max_x, max_y), (min_x, min_y), (255, 255, 255), thickness)

    return frame
