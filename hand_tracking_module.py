"""
Hand Tracking Module supporting both legacy MediaPipe Solutions API (Python <= 3.12)
and modern MediaPipe Tasks API / HandLandmarker (Python 3.13+).
"""

import cv2
import os
import urllib.request
import math
import numpy as np
import mediapipe as mp

# Hand landmark connections for drawing skeleton manually if needed
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),    # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (0, 17)                                # Palm base
]


class HandDetector:
    """
    Encapsulates MediaPipe Hand Tracking with support for both legacy mp.solutions
    and Python 3.13+ Tasks API (HandLandmarker).
    """
    def __init__(self, mode=False, max_hands=1, detection_con=0.7, track_con=0.7, model_path="hand_landmarker.task"):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = float(detection_con)
        self.track_con = float(track_con)
        self.model_path = model_path
        self.tip_ids = [4, 8, 12, 16, 20]
        self.lm_list = []
        self.results = None

        # Check if legacy mp.solutions.hands is available
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self.use_tasks_api = False
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=self.mode,
                max_num_hands=self.max_hands,
                min_detection_confidence=self.detection_con,
                min_tracking_confidence=self.track_con
            )
            self.mp_draw = mp.solutions.drawing_utils
            print("[INFO] Initialized HandDetector using MediaPipe Solutions API.")
        else:
            self.use_tasks_api = True
            self._ensure_model_exists()
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            base_options = mp_python.BaseOptions(model_asset_path=self.model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=self.max_hands,
                min_hand_detection_confidence=self.detection_con,
                min_hand_presence_confidence=self.detection_con,
                min_tracking_confidence=self.track_con,
                running_mode=vision.RunningMode.IMAGE
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
            print("[INFO] Initialized HandDetector using MediaPipe Tasks API (HandLandmarker).")

    def _ensure_model_exists(self):
        """Auto-downloads hand_landmarker.task if missing."""
        if not os.path.exists(self.model_path):
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            print(f"[INFO] Downloading MediaPipe Hand Landmarker model from Google CDN...")
            try:
                urllib.request.urlretrieve(url, self.model_path)
                print(f"[INFO] Model downloaded successfully to {self.model_path}")
            except Exception as e:
                print(f"[ERROR] Failed to download hand landmarker model: {e}")
                raise e

    def find_hands(self, img, draw=True):
        """
        Detects hands in a BGR image frame and draws tracking nodes/connections.
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if not self.use_tasks_api:
            self.results = self.hands.process(img_rgb)
            if self.results.multi_hand_landmarks and draw:
                for hand_lms in self.results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        img, 
                        hand_lms, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                        self.mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2)
                    )
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            self.results = self.detector.detect(mp_image)
            if self.results.hand_landmarks and draw:
                h, w, c = img.shape
                for hand_lms in self.results.hand_landmarks:
                    # Draw connections
                    for p1_id, p2_id in HAND_CONNECTIONS:
                        x1, y1 = int(hand_lms[p1_id].x * w), int(hand_lms[p1_id].y * h)
                        x2, y2 = int(hand_lms[p2_id].x * w), int(hand_lms[p2_id].y * h)
                        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    # Draw joint landmarks
                    for lm in hand_lms:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(img, (cx, cy), 4, (0, 255, 0), cv2.FILLED)

        return img

    def find_positions(self, img, hand_no=0, draw=True):
        """
        Extracts pixel coordinates for all 21 hand landmarks.
        Returns:
            lm_list: List of [id, cx, cy]
            bbox: Bounding box [xmin, ymin, xmax, ymax]
        """
        self.lm_list = []
        bbox = []
        x_list = []
        y_list = []

        h, w, c = img.shape

        if not self.use_tasks_api:
            if self.results and self.results.multi_hand_landmarks:
                if hand_no < len(self.results.multi_hand_landmarks):
                    my_hand = self.results.multi_hand_landmarks[hand_no]
                    for id, lm in enumerate(my_hand.landmark):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        x_list.append(cx)
                        y_list.append(cy)
                        self.lm_list.append([id, cx, cy])
                        if draw:
                            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        else:
            if self.results and self.results.hand_landmarks:
                if hand_no < len(self.results.hand_landmarks):
                    my_hand = self.results.hand_landmarks[hand_no]
                    for id, lm in enumerate(my_hand):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        x_list.append(cx)
                        y_list.append(cy)
                        self.lm_list.append([id, cx, cy])
                        if draw:
                            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        if x_list and y_list:
            xmin, xmax = min(x_list), max(x_list)
            ymin, ymax = min(y_list), max(y_list)
            bbox = [xmin, ymin, xmax, ymax]
            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)

        return self.lm_list, bbox

    def fingers_up(self):
        """
        Determines which fingers are raised up.
        Returns binary list of length 5: [Thumb, Index, Middle, Ring, Pinky]
        1 = Raised / Open, 0 = Closed / Folded.
        """
        fingers = []
        if len(self.lm_list) < 21:
            return [0, 0, 0, 0, 0]

        # Thumb: compare tip (id 4) with IP joint (id 3/2)
        if abs(self.lm_list[self.tip_ids[0]][1] - self.lm_list[self.tip_ids[0] - 2][1]) > 20:
            if self.lm_list[self.tip_ids[0]][1] > self.lm_list[self.tip_ids[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            if self.lm_list[self.tip_ids[0]][2] < self.lm_list[self.tip_ids[0] - 1][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # 4 Fingers: Index, Middle, Ring, Pinky
        for id in range(1, 5):
            if self.lm_list[self.tip_ids[id]][2] < self.lm_list[self.tip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def find_distance(self, p1, p2, img=None, draw=True, r=10, t=3):
        """
        Calculates euclidean distance between 2 landmarks p1 and p2.
        """
        if len(self.lm_list) <= max(p1, p2):
            return 0, img, [0, 0, 0, 0, 0, 0]

        x1, y1 = self.lm_list[p1][1], self.lm_list[p1][2]
        x2, y2 = self.lm_list[p2][1], self.lm_list[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        length = math.hypot(x2 - x1, y2 - y1)

        if img is not None and draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)

        return length, img, [x1, y1, x2, y2, cx, cy]
