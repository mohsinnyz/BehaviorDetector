#/Users/mohsinniaz/BehaviorDetector/src/utils/visualizer.py

import cv2
import src.config as config

class Visualizer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_status(self, frame, key, value, position, color=(0, 255, 0)):
        """Draws a simple status line: 'Key: Value'"""
        text = f"{key}: {value}"
        cv2.putText(frame, text, position, self.font, 0.6, color, 2)

    def draw_objects(self, frame, objects):
        """Draws bounding boxes and labels for YOLO detections"""
        if not objects:
            return

        for obj in objects:
            x1, y1, x2, y2 = obj['box']
            color = config.COLORS.get(obj['class_id'], config.COLORS['default'])
            
            # Draw Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw Label background for better visibility
            label = f"{obj['label']} {int(obj['conf']*100)}%"
            (w, h), _ = cv2.getTextSize(label, self.font, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
            
            # Draw Text
            cv2.putText(frame, label, (x1, y1 - 5), self.font, 0.6, (255, 255, 255), 2)

    def draw_alerts(self, frame, alerts):
        """
        Draws big alert text in center of screen.
        alerts = ["PHONE DETECTED", "WAKE UP"]
        """
        if not alerts:
            return

        h, w, _ = frame.shape
        center_x, center_y = w // 2, h // 2
        
        offset = 0
        for msg in alerts:
            (text_w, text_h), _ = cv2.getTextSize(msg, self.font, 1.2, 3)
            # Draw red text with black outline for high visibility
            cv2.putText(frame, msg, (center_x - text_w // 2, center_y + offset), 
                        self.font, 1.2, (0, 0, 0), 6) # Outline
            cv2.putText(frame, msg, (center_x - text_w // 2, center_y + offset), 
                        self.font, 1.2, (0, 0, 255), 3) # Text
            offset += 50

    def draw_fps(self, frame, fps):
        h, w, _ = frame.shape
        cv2.putText(frame, f"FPS: {int(fps)}", (w - 120, 40), 
                    self.font, 0.7, (255, 255, 0), 2)