import math
import src.config as config

class DrowsinessDetector:
    def __init__(self):
        # Mediapipe Landmark Indices for the eyes (Standard 6-point definition)
        # Order: [left_corner, top1, top2, right_corner, bottom2, bottom1]
        self.LEFT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
        
        self.frame_counter = 0
        self.alarm_on = False

    def _euclidean_dist(self, point1, point2):
        """Calculates distance between two 2D points"""
        return math.hypot(point1[0] - point2[0], point1[1] - point2[1])

    def calculate_ear(self, eye_points):
        """
        Calculates Eye Aspect Ratio (EAR)
        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        """
        # Vertical distances
        A = self._euclidean_dist(eye_points[1], eye_points[5])
        B = self._euclidean_dist(eye_points[2], eye_points[4])

        # Horizontal distance
        C = self._euclidean_dist(eye_points[0], eye_points[3])

        if C == 0:
            return 0.0

        ear = (A + B) / (2.0 * C)
        return ear

    def analyze(self, landmarks, frame_w, frame_h):
        """
        Returns: (is_drowsy, ear_score)
        """
        if landmarks is None:
            self.frame_counter = 0
            return False, 0.0

        # Helper to convert normalized landmark to pixel coordinates (x, y)
        def get_coords(indices):
            coords = []
            for idx in indices:
                lm = landmarks.landmark[idx]
                coords.append((int(lm.x * frame_w), int(lm.y * frame_h)))
            return coords

        # Get coordinates for both eyes
        left_eye = get_coords(self.LEFT_EYE_IDXS)
        right_eye = get_coords(self.RIGHT_EYE_IDXS)

        # Calculate EAR for both eyes
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)

        # Average EAR
        avg_ear = (left_ear + right_ear) / 2.0

        # Check threshold
        if avg_ear < config.EAR_THRESHOLD:
            self.frame_counter += 1
        else:
            self.frame_counter = 0
            self.alarm_on = False

        # Trigger alarm if eyes closed for enough frames
        if self.frame_counter >= config.EAR_CONSEC_FRAMES:
            self.alarm_on = True

        return self.alarm_on, avg_ear