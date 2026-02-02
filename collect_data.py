import cv2
import os
import time

# Settings
output_folder = "data/raw/webcam_captures"

# Counters
counts = {'phone': 0, 'food': 0, 'drink': 0}

# Create folder if not exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

cap = cv2.VideoCapture(0)

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("--- Data Collection Tool ---")
print("Press '0' -> Save PHONE")
print("Press '1' -> Save FOOD")
print("Press '2' -> Save DRINK")
print("Press 'q' -> QUIT")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Camera not found.")
        break
    
    # Copy frame for drawing text (so we don't save the text on the image)
    display_frame = frame.copy()
    
    # Display Status
    status_text = f"Phone: {counts['phone']} | Food: {counts['food']} | Drink: {counts['drink']}"
    cv2.putText(display_frame, status_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Data Collector", display_frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('0'):
        filename = f"{output_folder}/phone_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        counts['phone'] += 1
        print(f"Saved PHONE ({counts['phone']})")
        time.sleep(0.2) # Small delay to prevent accidental double-clicks
        
    elif key == ord('1'):
        filename = f"{output_folder}/food_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        counts['food'] += 1
        print(f"Saved FOOD ({counts['food']})")
        time.sleep(0.2)
        
    elif key == ord('2'):
        filename = f"{output_folder}/drink_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        counts['drink'] += 1
        print(f"Saved DRINK ({counts['drink']})")
        time.sleep(0.2)

cap.release()
cv2.destroyAllWindows()