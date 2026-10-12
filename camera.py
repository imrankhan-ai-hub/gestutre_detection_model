import cv2
import numpy as np


class webcam:
    def __init__(self):
        self.cap=cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error opening video stream or file")
            exit()

    def read(self):
        ret, frame=self.cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            return None
        return cv2.flip(frame, 1)
    def release(self):
        self.cap.release()