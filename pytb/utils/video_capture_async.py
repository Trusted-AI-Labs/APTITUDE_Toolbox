# file: video_capture_async.py
# from https://github.com/LeonLok/Deep-SORT-YOLOv4/blob/master/tensorflow2.0/deep-sort-yolov4/videocaptureasync.py

import threading
import cv2
from time import sleep
import copy


class VideoCaptureAsync:
    def __init__(self, file_path: str):
        """
        This class allows to read a video file asynchronously, meaning that the next frame is fetched
        between the call to the read() function.
        It is used the same way as cv2.VideoCapture, except that it is created using cap = VideoCaptureAsync(video_path)
        and that cap.stop() should be called before calling cap.release()

        Args:
            file_path (str): The path to the video file to read
        """

        # , width=2688, height=1520):
        self.src = file_path
        self.cap = cv2.VideoCapture(self.src)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.started = False
        self.ready = False
        self.read_lock = threading.Lock()

        self.start()

    def set(self, var1, var2):
        self.cap.set(var1, var2)

    def start(self):
        if self.started:
            print('[!] Asynchronous video capturing has already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            while self.ready and self.started:
                sleep(0)
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
                self.ready = True

    def read(self):
        while not self.ready:
            sleep(0)
        with self.read_lock:
            grabbed = self.grabbed
            self.ready = False
            if grabbed:
                frame = self.frame.copy()
                return grabbed, frame
            return grabbed, None

    def is_opened(self):
        return self.cap.isOpened()

    def stop(self):
        self.started = False
        self.thread.join()

    def release(self):
        self.cap.release()

    def get(self, x):
        return self.cap.get(x)

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()
