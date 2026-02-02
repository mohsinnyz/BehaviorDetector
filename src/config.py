#/Users/mohsinniaz/BehaviorDetector/src/config.py
import os

# ==========================================
# 1. PROJECT PATHS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

# Pointing to your trained Custom Model
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, 'best.pt') 

# ==========================================
# 2. CAMERA SETTINGS
# ==========================================
CAMERA_INDEX = 0      # 0 = Default Webcam
FRAME_WIDTH = 1280    # HD Resolution (Better for distance detection)
FRAME_HEIGHT = 720
FPS = 30

# ==========================================
# 3. OBJECT DETECTION (YOLOv8)
# ==========================================
CONFIDENCE_THRESHOLD = 0.5  # Ignore detections below 50%

# PERFORMANCE OPTIMIZATION: Only run YOLO every N frames
# 30 frames = approx once per second. Lower this (e.g., 10) if you need faster reaction.
DETECTION_INTERVAL = 30  

# Must match the training order: 0=phone, 1=food, 2=drink
CLASS_NAMES = {
    0: 'phone',
    1: 'food',
    2: 'drink'
}

# Visualization Colors (B, G, R)
COLORS = {
    0: (0, 0, 255),    # Red   (Phone - Danger!)
    1: (0, 255, 0),    # Green (Food)
    2: (255, 0, 0),    # Blue  (Drink)
    'default': (255, 255, 255)
}

# ==========================================
# 4. DROWSINESS (Mediapipe)
# ==========================================
EAR_THRESHOLD = 0.25      # Eye Aspect Ratio (below 0.25 = closed)
EAR_CONSEC_FRAMES = 15    # Must close eyes for ~0.5s to trigger

# ==========================================
# 5. DISTRACTION (Head Pose)
# ==========================================
PITCH_THRESHOLD = 15      # Looking Up/Down
YAW_THRESHOLD = 20        # Looking Left/Right
DISTRACTION_FRAMES = 10   # Consecutive frames

# ==========================================
# 6. ALERTS
# ==========================================
ENABLE_AUDIO = True