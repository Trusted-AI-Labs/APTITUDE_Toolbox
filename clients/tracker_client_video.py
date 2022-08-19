"""
Copyright (c) 2021-2022 UCLouvain, ICTEAM
Licensed under GPL-3.0 [see LICENSE for details]
Written by Jonathan Samelson (2021-2022)
"""

import json
import time
from timeit import default_timer
from tqdm import tqdm

import cv2
import ffmpeg
import numpy as np
import logging

from pytb.detection.detection_manager import DetectionManager
from pytb.detection.detector_factory import DetectorFactory
from pytb.tracking.tracker_factory import TrackerFactory
from pytb.tracking.tracking_manager import TrackingManager
from pytb.utils.video_capture_async import VideoCaptureAsync

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(10000, 3), dtype="uint8")
font = cv2.FONT_HERSHEY_DUPLEX
line_type = cv2.LINE_AA
thickness = 2


def main(cfg_detect, cfg_track, cfg_classes, video_path, frame_interval, record_path, record_fps, record_size,
         mot_path, headless, display_size, async_flag, gt_path):

    log = logging.getLogger("aptitude-toolbox")

    # Get parameters of the different stages of the detection process
    with open(cfg_detect) as config_file:
        detect1 = json.load(config_file)
        log.debug("Detector config loaded.")

    detect1_proc = detect1['Proc']
    detect1_preproc = detect1['Preproc']
    detect1_postproc = detect1['Postproc']

    # Get parameters of the different stages of the tracking process
    with open(cfg_track) as config_file:
        track1 = json.load(config_file)
        log.debug("Tracker config loaded.")

    track1_proc = track1['Proc']
    track1_preproc = track1['Preproc']
    track1_postproc = track1['Postproc']

    # Get the classes of the object to be detected
    with open(cfg_classes) as config_file:
        CLASSES = json.load(config_file)['classes']
        log.debug("Classes config loaded.")

    # Instantiate the detector
    start = default_timer()
    detection_manager = DetectionManager(DetectorFactory.create_detector(detect1_proc), detect1_preproc,
                                         detect1_postproc)
    end = default_timer()
    log.info("Detector init duration = {}s".format(str(end - start)))

    # Instantiate the tracker
    start = default_timer()
    tracking_manager = TrackingManager(TrackerFactory.create_tracker(track1_proc), track1_preproc, track1_postproc)
    end = default_timer()
    log.info("Tracker init duration = {}s".format(str(end - start)))

    # If async flag is present, use the VideoCaptureAsync class
    # It allows to load the images in parallel of the detection/tracking process
    if async_flag:
        cap = VideoCaptureAsync(video_path)
    else:
        cap = cv2.VideoCapture(video_path)

    # Measure elapsed time to read the image
    read_time_start = default_timer()
    is_reading, frame = cap.read()
    read_time = default_timer() - read_time_start
    if is_reading:
        log.debug("Video file opened successfully.")
    else:
        log.error("Error while opening video file, check if the input path is correct.")

    # Read GT file in MOT format
    if gt_path is not None:
        with open(gt_path, 'r') as gt:
            gt_lines = gt.readlines()
            gt_line_number = 0
            gt_frame_num = 1
            log.debug("Ground truth file open successfully.")

    is_paused = False

    # If record flag is present, initializes the VideoWriter to save the results of the process
    record = record_path is not None
    H, W, _ = frame.shape
    if record:
        if record_size is None:
            record_size = (H, W)
        output_video = cv2.VideoWriter(record_path, cv2.VideoWriter_fourcc(*'mp4v'), record_fps, record_size)
        log.debug("VideoWriter opened successfully.")

    start_time = default_timer()
    last_update = default_timer()
    before_loop = start_time
    counter = 0
    tot_det_time = 0
    tot_track_time = 0

    output_lines = []

    # Get the number of frames of the video
    probe = ffmpeg.probe(video_path)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    nb_frames = int(video_info['nb_frames'])
    pbar = tqdm(total=nb_frames)

    while is_reading:

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

        if is_paused:
            time.sleep(0.5)
            continue

        # Frame interval can be used to skip frames.
        # Do the process if the counter is divisible by frame_interval
        if counter % frame_interval == 0:
            (H, W, _) = frame.shape
            log.debug("Before detection.")
            det = detection_manager.detect(frame)
            log.debug("After detection & before tracking.")
            if tracking_manager.tracker.need_frame:
                res = tracking_manager.track(det, frame)
                log.debug("After tracking, with frame.")
            else:
                res = tracking_manager.track(det)
                log.debug("After tracking, without frames.")

            tot_det_time += det.detection_time
            tot_track_time += res.tracking_time

            # Change dimensions of the result to match to the initial dimension of the frame
            res.change_dims(W, H)
            log.debug("Dimensions of the results changed: (W: {}, H:{}).".format(W, H))

            # Add the result to the output file in MOT format
            if mot_path is not None:
                for i in range(res.number_objects):
                    results = "{0},{1},{2},{3},{4},{5},-1,-1,-1,-1\n".format(counter + 1, res.global_IDs[i],
                                                                             res.bboxes[i][0], res.bboxes[i][1],
                                                                             res.bboxes[i][2], res.bboxes[i][3])
                    log.debug(i, results)
                    output_lines.append(results)

            res.to_x1_y1_x2_y2()
            log.debug("Results converted to x1,y1,x2,y2.")
            # print(res)

            # Add GT bboxes from GT file (MOT format) to the frame
            if gt_path is not None:
                while gt_frame_num == counter+1 and gt_line_number < len(gt_lines):
                    line = gt_lines[gt_line_number]
                    new_gt_frame_num, id, left, top, width, height, _, _, _, _ = line.split(",")
                    new_gt_frame_num, id, left, top, width, height = int(new_gt_frame_num), int(id), int(left), \
                                                                     int(top), int(width), int(height)
                    if new_gt_frame_num > gt_frame_num:
                        gt_frame_num = new_gt_frame_num
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
        pbar.update(1)
        counter += 1

        # Read the new frame before starting a new iteration
        read_time_start = default_timer()
        is_reading, frame = cap.read()
        read_time += default_timer() - read_time_start

    pbar.close()

    log.info("Average FPS: {}".format(str(counter / (default_timer() - before_loop))))
    log.info("Average FPS without read time: {}".format(str(counter / (default_timer() - before_loop - read_time))))

    log.info("Total detection time: {}".format(tot_det_time))
    log.info("Total tracking time: {}".format(tot_track_time))

    log.info("Average detection time: {}".format(tot_det_time / counter))
    log.info("Average tracking time: {}".format(tot_track_time / counter))

    # Write all the lines at once in a file
    if mot_path is not None:
        with open(mot_path, "w") as out:
            out.writelines(output_lines)
            log.debug("Lines written to output path.")

    # Stop method of VideoCaptureAsync must be called before release
    if async_flag:
        cap.stop()
    cap.release()

    # If record flag is enabled, release the video so it can be written effectively on the disk
    if record:
        output_video.release()

    cv2.destroyAllWindows()
