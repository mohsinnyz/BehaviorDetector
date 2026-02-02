import cv2
import time
import src.config as config

# Import our custom modules
from src.detectors.face_mesh import FaceMeshDetector
from src.detectors.drowsiness import DrowsinessDetector
from src.detectors.distraction import DistractionDetector
from src.detectors.object_det import ObjectDetector
from src.utils.alerts import AudioAlert

def draw_status(frame, text, value, position, color=(0, 255, 0)):
    """Helper to draw clean status bars on screen"""
    cv2.putText(frame, f"{text}: {value}", position, 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def main():
    # 1. Initialize Camera
    print("🚀 Initializing Camera...")
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    # 2. Initialize All Detectors
    face_mesh = FaceMeshDetector()
    drowsy_det = DrowsinessDetector()
    distract_det = DistractionDetector()
    object_det = ObjectDetector()
    alerter = AudioAlert()

    print("\n✅ SYSTEM READY. Monitoring Started...")
    print("   Press 'q' to Quit.")

    # FPS Calculation variables
    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Camera Error")
            break

        # Flip frame for mirror effect (easier for user)
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # ==========================================
        # PHASE 1: FACE ANALYSIS (Mediapipe)
        # ==========================================
        landmarks = face_mesh.get_landmarks(frame)
        
        # A. Drowsiness Detection
        is_drowsy, ear_score = drowsy_det.analyze(landmarks, w, h)
        
        # B. Distraction Detection
        is_distracted, (pitch, yaw, roll) = distract_det.analyze(landmarks, w, h)

        # ==========================================
        # PHASE 2: OBJECT DETECTION (YOLO)
        # ==========================================
        # Detects: Phone, Food, Drink
        objects = object_det.detect(frame)

        # ==========================================
        # PHASE 3: VISUALIZATION & ALERTS
        # ==========================================
        
        # --- 1. Draw Object Boxes ---
        for obj in objects:
            # Draw Box
            x1, y1, x2, y2 = obj['box']
            color = config.COLORS.get(obj['class_id'], (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw Label
            label = f"{obj['label']} {int(obj['conf']*100)}%"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Trigger Alert for Phone
            if obj['label'] == 'phone':
                alerter.trigger("danger")
                cv2.putText(frame, "!!! PHONE DETECTED !!!", (w//2 - 150, h//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # --- 2. Draw Face Status ---
        if landmarks:
            # Drowsiness Status
            drowsy_color = (0, 0, 255) if is_drowsy else (0, 255, 0)
            status_text = "DROWSY!" if is_drowsy else "Awake"
            draw_status(frame, "Status", status_text, (20, 40), drowsy_color)
            draw_status(frame, "EAR", f"{ear_score:.2f}", (20, 70), drowsy_color)

            # Distraction Status
            distract_color = (0, 0, 255) if is_distracted else (0, 255, 0)
            pose_text = "DISTRACTED!" if is_distracted else "Focused"
            draw_status(frame, "Attention", pose_text, (20, 110), distract_color)
            
            # Trigger Alerts
            if is_drowsy:
                alerter.trigger("danger")
                cv2.putText(frame, "!!! WAKE UP !!!", (w//2 - 100, h//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            
            if is_distracted:
                alerter.trigger("warning")

        # --- 3. FPS Counter ---
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (w - 120, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Show the frame
        cv2.imshow("Behavior Detector AI", frame)

        # Quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()