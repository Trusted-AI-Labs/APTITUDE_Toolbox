import json
import time
from timeit import default_timer

import cv2
import numpy as np

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


def main(cfg_detect, cfg_track, cfg_classes, video_path, frame_interval, record_path, record_fps, headless,
         show_fps=False, async_flag=False):

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
    detection_manager = DetectionManager(DetectorFactory.create_detector(detect1_proc), detect1_preproc,
                                         detect1_postproc)
    end = default_timer()
    print("Detector init duration = " + str(end - start))

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
    is_paused = False

    record = record_path is not None
    H, W, _ = frame.shape
    if record:
        output_video = cv2.VideoWriter(record_path, cv2.VideoWriter_fourcc(*'mp4v'), record_fps, (W, H))

    start_time = default_timer()
    before_loop = start_time
    counter = 0
    fps_number_frames = 10

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

            # Visualize
            res.to_x1_y1_x2_y2()
            res.change_dims(W, H)
            print(res)

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
        read_time += default_timer() - read_time_start

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
