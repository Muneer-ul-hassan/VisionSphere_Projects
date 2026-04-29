import cv2
import numpy as np
import os
from typing import List, Dict, Optional, Tuple
from app.config import settings

class FaceRecognizer:
    """Lite face recognition using OpenCV's LBPH or DNN module"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.known_names = []
        self.is_trained = False
        self.face_data_path = settings.face_data_path
        
        # Ensure directories exist
        os.makedirs(self.face_data_path, exist_ok=True)
        self.load_and_train()

    def load_and_train(self):
        """Load saved face images and train the recognizer"""
        faces = []
        labels = []
        self.known_names = []
        
        label_id = 0
        for name in os.listdir(self.face_data_path):
            person_dir = os.path.join(self.face_data_path, name)
            if not os.path.isdir(person_dir):
                continue
                
            self.known_names.append(name)
            for filename in os.listdir(person_dir):
                img_path = os.path.join(person_dir, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    faces.append(img)
                    labels.append(label_id)
            label_id += 1
            
        if faces:
            self.recognizer.train(faces, np.array(labels))
            self.is_trained = True
            print(f"👤 Face recognizer trained with {len(self.known_names)} people")
        else:
            print("👤 No face data found - recognizer ready for enrollment")

    def recognize(self, frame: np.ndarray) -> List[Dict]:
        """Detect and recognize faces in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        results = []
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            name = "Unknown"
            confidence = 0.0
            
            if self.is_trained:
                label_id, conf = self.recognizer.predict(face_roi)
                # LBPH confidence: lower is better (distance)
                # We normalize it for the UI (0 to 1)
                norm_conf = max(0, 100 - conf) / 100.0
                
                if conf < 100: # Threshold for a match
                    name = self.known_names[label_id]
                    confidence = norm_conf
            
            results.append({
                "label": name,
                "confidence": round(float(confidence), 2),
                "bbox": [int(x), int(y), int(x+w), int(y+h)],
                "center": [int(x + w//2), int(y + h//2)],
                "type": "face"
            })
            
        return results

    def enroll_face(self, frame: np.ndarray, name: str, bbox: List[int]):
        """Save a new face to the database and retrain"""
        x1, y1, x2, y2 = bbox
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_roi = gray[max(0, y1):y2, max(0, x1):x2]
        
        if face_roi.size == 0:
            return False
            
        person_dir = os.path.join(self.face_data_path, name)
        os.makedirs(person_dir, exist_ok=True)
        
        # Save image with timestamp
        import time
        filename = f"face_{int(time.time())}.jpg"
        cv2.imwrite(os.path.join(person_dir, filename), face_roi)
        
        # Retrain
        self.load_and_train()
        return True

# Global instance
face_recognizer = FaceRecognizer()
