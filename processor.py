import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

MODEL_PATH = Path("hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


class HandProcessor:
    def __init__(self):
        if not MODEL_PATH.exists():
            print("Downloading hand model (one time)...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.last_ms = -1

    def process(self, frame):
        """Draws the hand on the frame. Returns (frame, hand) where hand is
        the 21 landmarks, or None if no hand was found."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts = max(int(time.monotonic() * 1000), self.last_ms + 1)
        self.last_ms = ts
        result = self.landmarker.detect_for_video(mp_image, ts)

        if not result.hand_landmarks:
            return frame, None

        hand = result.hand_landmarks[0]
        h, w = frame.shape[:2]
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
        for a, b in CONNECTIONS:
            cv2.line(frame, points[a], points[b], (0, 255, 0), 2)
        for x, y in points:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        return frame, hand

    def close(self):
        self.landmarker.close()