import cv2
import numpy as np
import src.config as config

class DistractionDetector:
    def __init__(self):
        self.frame_counter = 0
        self.alarm_on = False

    def get_head_pose(self, landmarks, frame_w, frame_h):
        """
        Estimates Pitch, Yaw, and Roll using SolvePnP.
        """
        # 1. 3D Model Points (Standard Generic Face Model)
        # Nose tip, Chin, Left Eye corner, Right Eye corner, Left Mouth, Right Mouth
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])

        # 2. 2D Image Points (From Mediapipe)
        # Map Mediapipe indices to the 3D model points
        # Nose=1, Chin=152, L_Eye=33, R_Eye=263, L_Mouth=61, R_Mouth=291
        idx_list = [1, 152, 33, 263, 61, 291]
        
        image_points = []
        for idx in idx_list:
            lm = landmarks.landmark[idx]
            x, y = int(lm.x * frame_w), int(lm.y * frame_h)
            image_points.append((x, y))

        image_points = np.array(image_points, dtype="double")

        # 3. Camera Matrix (Approximation)
        focal_length = frame_w
        center = (frame_w / 2, frame_h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

        # 4. Solve PnP to find rotation vector
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        # 5. Convert Rotation Vector to Euler Angles
        # Jacobian not used, just unpacking
        rmat, jac = cv2.Rodrigues(rotation_vector)
        angles, mtxR, mtxQ, Q, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        # Angles are usually: pitch (x), yaw (y), roll (z)
        pitch = angles[0] * 360
        yaw   = angles[1] * 360
        roll  = angles[2] * 360

        return pitch, yaw, roll

    def analyze(self, landmarks, frame_w, frame_h):
        """
        Returns: (is_distracted, (pitch, yaw, roll))
        """
        if landmarks is None:
            return False, (0, 0, 0)

        pitch, yaw, roll = self.get_head_pose(landmarks, frame_w, frame_h)

        # Check Thresholds defined in config
        # Pitch: Looking Up/Down
        # Yaw: Looking Left/Right
        if abs(pitch) > config.PITCH_THRESHOLD or abs(yaw) > config.YAW_THRESHOLD:
            self.frame_counter += 1
        else:
            self.frame_counter = 0
            self.alarm_on = False

        if self.frame_counter >= config.DISTRACTION_FRAMES:
            self.alarm_on = True

        return self.alarm_on, (pitch, yaw, roll)