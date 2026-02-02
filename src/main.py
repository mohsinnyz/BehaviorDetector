#/Users/mohsinniaz/BehaviorDetector/src/main.py
import cv2
import time
import src.config as config

# Import Custom Modules
from src.detectors.face_mesh import FaceMeshDetector
from src.detectors.drowsiness import DrowsinessDetector
from src.detectors.distraction import DistractionDetector
from src.detectors.object_det import ObjectDetector
from src.utils.alerts import AudioAlert
from src.utils.visualizer import Visualizer

def main():
    # 1. Initialize System
    print("🚀 Initializing Behavior Detector...")
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    # 2. Load Modules
    face_mesh = FaceMeshDetector()
    drowsy_det = DrowsinessDetector()
    distract_det = DistractionDetector()
    object_det = ObjectDetector()
    alerter = AudioAlert()
    viz = Visualizer()

    print("\n✅ SYSTEM READY. Monitoring Started...")
    print(f"ℹ️  Object Detection running every {config.DETECTION_INTERVAL} frames.")

    # 3. Runtime Variables
    prev_time = 0
    frame_count = 0
    current_objects = [] # Store objects to persist them between YOLO scans

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Camera Error")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        frame_count += 1
        active_alerts = []

        # ==========================================
        # PHASE 1: FAST CHECKS (Face - Run Every Frame)
        # ==========================================
        # Face detection is lightweight, so we run it every frame for real-time eyes/head tracking.
        landmarks = face_mesh.get_landmarks(frame)
        
        is_drowsy = False
        is_distracted = False
        ear_score = 0
        pose_data = (0,0,0)

        if landmarks:
            # Drowsiness
            is_drowsy, ear_score = drowsy_det.analyze(landmarks, w, h)
            # Distraction
            is_distracted, pose_data = distract_det.analyze(landmarks, w, h)

        # ==========================================
        # PHASE 2: SLOW CHECKS (YOLO - Run Every N Frames)
        # ==========================================
        # We only run YOLO if the frame count matches the interval (e.g., every 30th frame).
        if frame_count % config.DETECTION_INTERVAL == 0:
            current_objects = object_det.detect(frame)
        
        # ==========================================
        # PHASE 3: ALERTS & LOGIC
        # ==========================================
        # Check Objects (using the persisted 'current_objects' list)
        for obj in current_objects:
            if obj['label'] == 'phone':
                active_alerts.append("!!! PHONE DETECTED !!!")
                alerter.trigger("danger")

        # Check Face
        if is_drowsy:
            active_alerts.append("!!! WAKE UP !!!")
            alerter.trigger("danger")
        
        if is_distracted:
            alerter.trigger("warning")

        # ==========================================
        # PHASE 4: VISUALIZATION
        # ==========================================
        # A. Draw Objects (Persisted)
        viz.draw_objects(frame, current_objects)

        # B. Draw Face Stats
        drowsy_color = (0, 0, 255) if is_drowsy else (0, 255, 0)
        viz.draw_status(frame, "Status", "DROWSY" if is_drowsy else "Awake", (20, 40), drowsy_color)
        viz.draw_status(frame, "EAR", f"{ear_score:.2f}", (20, 70), drowsy_color)

        distract_color = (0, 0, 255) if is_distracted else (0, 255, 0)
        viz.draw_status(frame, "Focus", "DISTRACTED" if is_distracted else "Good", (20, 110), distract_color)

        # C. Draw Big Alerts
        if active_alerts:
            viz.draw_alerts(frame, active_alerts)

        # D. FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        viz.draw_fps(frame, fps)

        cv2.imshow("Behavior Detector AI", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()