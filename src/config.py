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
# UPDATED: Lowered to 0.30 to catch Food/Drink easier
CONFIDENCE_THRESHOLD = 0.30  

# UPDATED: Run every 10 frames (approx 0.3s) to catch quick sips/bites
DETECTION_INTERVAL = 10  

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
# UPDATED: Increased to 25 to allow looking down at laptop screen
PITCH_THRESHOLD = 25      
# UPDATED: Increased to 30 for more natural movement
YAW_THRESHOLD = 30        
DISTRACTION_FRAMES = 10   # Consecutive frames

# ==========================================
# 6. ALERTS
# ==========================================
ENABLE_AUDIO = True