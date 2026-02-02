#/Users/mohsinniaz/BehaviorDetector/src/detectors/face_mesh.py
import cv2
import mediapipe as mp

class FaceMeshDetector:
    def __init__(self):
        print("🔄 Initializing Mediapipe Face Mesh...")
        self.mp_face_mesh = mp.solutions.face_mesh
        # refine_landmarks=True gives us detailed eye/iris points
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("✅ Face Mesh Ready!")

    def get_landmarks(self, frame):
        """
        Accepts a BGR frame, converts to RGB, and returns 
        the landmarks for the first detected face.
        Returns None if no face is found.
        """
        # Mediapipe requires RGB images
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            # Return landmarks for the first face only
            return results.multi_face_landmarks[0]
            
        return None