#/Users/mohsinniaz/BehaviorDetector/src/detectors/object_det.py
import cv2
import math
from ultralytics import YOLO
import src.config as config

class ObjectDetector:
    def __init__(self):
        print(f"🔄 Loading YOLO model from: {config.YOLO_MODEL_PATH}...")
        try:
            # Load the model you trained (best.pt)
            self.model = YOLO(config.YOLO_MODEL_PATH)
            print("✅ YOLO Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            self.model = None

    def detect(self, frame):
        """
        Runs YOLO inference on the frame.
        Returns a list of detections: [{'label': 'phone', 'conf': 0.95, 'box': [x1, y1, x2, y2]}]
        """
        results = []
        
        if self.model is None:
            return results

        # Run inference (stream=True is faster for video)
        predictions = self.model(frame, stream=True, verbose=False, conf=config.CONFIDENCE_THRESHOLD)

        for p in predictions:
            boxes = p.boxes
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Get class info
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Get class name from config
                label = config.CLASS_NAMES.get(cls_id, "Unknown")
                
                results.append({
                    "label": label,
                    "conf": conf,
                    "box": [x1, y1, x2, y2],
                    "class_id": cls_id
                })

        return results