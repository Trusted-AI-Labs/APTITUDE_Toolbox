import json
import time
from timeit import default_timer

import cv2
import numpy as np

import pytb.utils.image_helper as ih
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


def main(cfg_detect, cfg_track, cfg_classes, video_path, roi_path, frame_interval, record_path, record_fps,
         mot_path, headless, show_fps, async_flag, gt_path):

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
    detection_manager = DetectionManager(DetectorFactory.create_detector(detect1_proc), detect1_preproc,
                                         detect1_postproc)
    end = default_timer()
    print("Detector init duration = " + str(end - start))

    # Instantiate tracker
    start = default_timer()
    tracking_manager = TrackingManager(TrackerFactory.create_tracker(track1_proc), track1_preproc, track1_postproc)
    end = default_timer()
    print("Tracker init duration = " + str(end - start))

    if async_flag:
        cap = VideoCaptureAsync(video_path)
    else:
        cap = cv2.VideoCapture(video_path)

    read_time_start = default_timer()
    is_reading, frame = cap.read()
    read_time = default_timer() - read_time_start

    # Apply ROI if any
    if roi_path is not None:
        roi_mask = ih.get_cv2_img_from_str(roi_path)
        frame = ih.get_roi_frame_from_mask(frame, roi_mask)

    # Read GT file in MOT format
    if gt_path is not None:
        with open(gt_path, 'r') as gt:
            gt_lines = gt.readlines()
            gt_line_number = 1
            gt_frame_num = 1

    is_paused = False

    record = record_path is not None
    H, W, _ = frame.shape
    if record:
        output_video = cv2.VideoWriter(record_path, cv2.VideoWriter_fourcc(*'mp4v'), record_fps, (W, H))

    start_time = default_timer()
    before_loop = start_time
    counter = 0
    fps_number_frames = 10

    output_lines = []

    while is_reading:

        k = cv2.waitKey(1) & 0xFF
        if k == ord('p'):  # pause/play loop if 'p' key is pressed
            is_paused = not is_paused
        if k == ord('q'):  # end video loop if 'q' key is pressed
            break

        if is_paused:
            time.sleep(0.5)
            continue

        if counter % frame_interval == 0:
            (H, W, _) = frame.shape
            det = detection_manager.detect(frame)
            if tracking_manager.tracker.need_frame:
                res = tracking_manager.track(det, frame)
            else: 
                res = tracking_manager.track(det)

            res.change_dims(W, H)

            # Add to output file
            if mot_path is not None:
                for i in range(res.number_objects):
                    output_lines.append("{0},{1},{2},{3},{4},{5},{6},-1,-1,-1\n".format(counter+1, res.global_IDs[i],
                                        res.bboxes[i][0], res.bboxes[i][1], res.bboxes[i][2], res.bboxes[i][3],
                                        res.det_confs[i]))

            # Visualize
            res.to_x1_y1_x2_y2()
            # print(res)

            # Apply GT bboxes
            if gt_path is not None:
                while gt_frame_num == counter+1:
                    line = gt_lines[gt_line_number]
                    gt_frame_num, id, left, top, width, height, _, _, _, _ = line.split(",")
                    gt_frame_num, id, left, top, width, height = int(gt_frame_num), int(id), int(left), int(top), \
                                                              int(width), int(height)
                    cv2.rectangle(frame, (left, top), (left + width, top + height), (255, 255, 255), 2)
                    # cv2.putText(frame, str(id), (left, top - 5), font, 1, (255, 255, 255), 2, line_type)
                    gt_line_number += 1

            for i in range(res.number_objects):
                id = res.global_IDs[i]
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

        counter += 1
        if show_fps and counter % fps_number_frames == 0:
            print("FPS:", fps_number_frames / (default_timer() - start_time))
            start_time = default_timer()

        read_time_start = default_timer()
        is_reading, frame = cap.read()

        # Apply ROI if any
        if is_reading and roi_path is not None:
            frame = ih.get_roi_frame_from_mask(frame, roi_mask)

        read_time += default_timer() - read_time_start

    if mot_path is not None:
        with open(mot_path, "w") as out:
            out.writelines(output_lines)

    print("Counter: ", counter)
    print("Average FPS:", counter / (default_timer() - before_loop))
    print("Average FPS w/o read time:", counter / (default_timer() - before_loop - read_time))

    if async_flag:
        cap.stop()
    else:
        cap.release()

    if record:
        output_video.release()

    cv2.destroyAllWindows()