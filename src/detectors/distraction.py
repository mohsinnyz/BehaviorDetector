import cv2
import numpy as np
from collections import deque
import src.config as config

class DistractionDetector:
    def __init__(self):
        self.frame_counter = 0
        self.alarm_on = False
        self.pitch_queue = deque(maxlen=5)
        self.yaw_queue = deque(maxlen=5)
        self.roll_queue = deque(maxlen=5)

    def get_head_pose(self, landmarks, frame_w, frame_h):
        """
        Estimates Pitch, Yaw, and Roll.
        """
        # 1. 3D Model Points
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])

        # 2. 2D Image Points
        idx_list = [1, 152, 33, 263, 61, 291]
        image_points = np.array([
            (int(landmarks.landmark[idx].x * frame_w),
             int(landmarks.landmark[idx].y * frame_h)) for idx in idx_list
        ], dtype="double")

        # 3. Camera Matrix
        focal_length = frame_w
        center = (frame_w / 2, frame_h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4, 1))

        # 4. Solve PnP
        success, rotation_vector, _ = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        # 5. Rotation Vector -> Euler Angles
        rmat, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        pitch = angles[0]
        yaw   = angles[1]
        roll  = angles[2]

        # --- FIX START: Normalize Pitch ---
        # If pitch is near 180 or -180, it means the axis is flipped.
        # We shift it to be near 0.
        if pitch > 100:
            pitch -= 180
        elif pitch < -100:
            pitch += 180
            
        # Optional: Scale up slightly if movement feels too small
        pitch = pitch * 1.0 
        yaw   = yaw * 1.0
        # --- FIX END ---

        return pitch, yaw, roll

    def analyze(self, landmarks, frame_w, frame_h):
        if landmarks is None:
            self.pitch_queue.clear()
            self.yaw_queue.clear()
            self.roll_queue.clear()
            self.frame_counter = 0
            self.alarm_on = False
            return False, (0, 0, 0)

        pitch, yaw, roll = self.get_head_pose(landmarks, frame_w, frame_h)

        self.pitch_queue.append(pitch)
        self.yaw_queue.append(yaw)
        self.roll_queue.append(roll)

        if len(self.pitch_queue) > 0:
            avg_pitch = sum(self.pitch_queue) / len(self.pitch_queue)
            avg_yaw   = sum(self.yaw_queue) / len(self.yaw_queue)
            avg_roll  = sum(self.roll_queue) / len(self.roll_queue)
        else:
            avg_pitch, avg_yaw, avg_roll = pitch, yaw, roll

        # Check thresholds
        if abs(avg_pitch) > config.PITCH_THRESHOLD or abs(avg_yaw) > config.YAW_THRESHOLD:
            self.frame_counter += 1
        else:
            self.frame_counter = 0
            self.alarm_on = False

        if self.frame_counter >= config.DISTRACTION_FRAMES:
            self.alarm_on = True

        return self.alarm_on, (avg_pitch, avg_yaw, avg_roll)