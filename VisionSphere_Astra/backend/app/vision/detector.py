import cv2
import numpy as np
import torch

# ── PyTorch 2.6 fix: monkey-patch torch.load to force weights_only=False ──
# YOLO models use custom classes blocked by PyTorch 2.6's new security default.
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load
# ──────────────────────────────────────────────────────────────────────────────

from ultralytics import YOLO
from typing import Dict, List, Optional
from collections import Counter
from app.config import settings
from app.vision.face_recognizer import face_recognizer


class VisionDetector:
    """Real-time object detection and tracking using YOLO and FaceRec"""

    def __init__(self):
        self.model = None
        self.confidence_threshold = settings.confidence_threshold
        self.camera = None
        self.is_running = False
        self.latest_frame = None
        self.latest_detections = []
        self.frame_count = 0
        
        # Simple tracking state
        self.objects = {} # id -> {label, center, last_seen}
        self.next_id = 0
        self.max_lost_frames = settings.tracking_max_age

    def load_model(self):
        """Lazy load YOLO model to avoid blocking app startup"""
        if self.model is None:
            print(f"🧠 Loading YOLO model ({settings.yolo_model})...")
            self.model = YOLO(settings.yolo_model)
            print("✅ YOLO model loaded")
        return self.model

    def start_camera(self, camera_index: int = None) -> bool:
        """Start webcam capture"""
        index = camera_index if camera_index is not None else settings.camera_index
        self.camera = cv2.VideoCapture(index)

        # Fallback for Windows MSMF bug
        if not self.camera.isOpened():
            print(f"⚠️ Primary camera failed, trying DirectShow for index {index}")
            self.camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        if not self.camera.isOpened():
            print(f"❌ Camera totally failed to open at index {index}")
            return False

        self.is_running = True
        return True

    def stop_camera(self):
        """Stop webcam capture"""
        self.is_running = False
        if self.camera:
            self.camera.release()
            self.camera = None

    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame from camera"""
        if not self.camera or not self.is_running:
            return None

        ret, frame = self.camera.read()
        if ret:
            self.latest_frame = frame
            return frame
        return None

    def detect_objects(self, frame: Optional[np.ndarray] = None) -> Dict:
        """Run object detection and face recognition on a frame"""
        if frame is None:
            frame = self.capture_frame()

        if frame is None:
            return {"objects": [], "counts": {}, "image": None}

        self.frame_count += 1
        detected_objects = []

        # Ensure model is loaded
        self.load_model()

        # 1. Run YOLO detection (every frame for smoothness)
        # Added conf and iou thresholds to prevent duplicate overlapping boxes
        results = self.model(frame, verbose=False, conf=self.confidence_threshold, iou=0.45)
        result = results[0]
        names = result.names
        boxes = result.boxes

        for box in boxes:
            conf = float(box.conf.item())
            cls_id = int(box.cls.item())
            label = names[cls_id]

            if conf >= self.confidence_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detected_objects.append({
                    "id": -1, # Will be assigned by tracker
                    "label": label,
                    "confidence": round(conf, 3),
                    "bbox": [x1, y1, x2, y2],
                    "center": [(x1 + x2) // 2, (y1 + y2) // 2],
                    "type": "object"
                })

        # 2. Run Face Recognition (every N-th frame for speed)
        faces = []
        if self.frame_count % settings.face_detection_skip_frames == 0:
            faces = face_recognizer.recognize(frame)

        # 3. Deduplication: Merge faces into persons
        final_detections = []
        # First, add all objects that aren't persons
        for obj in detected_objects:
            if obj["label"] != "person":
                final_detections.append(obj)
        
        # Then handle persons and faces
        person_objects = [obj for obj in detected_objects if obj["label"] == "person"]
        
        for person in person_objects:
            px1, py1, px2, py2 = person["bbox"]
            # Find faces inside this person box
            for face in faces[:]: # Copy to allow removal
                fx1, fy1, fx2, fy2 = face["bbox"]
                # Check if face center is inside person box
                fcx, fcy = face["center"]
                if px1 <= fcx <= px2 and py1 <= fcy <= py2:
                    # Update person with identity if face is known
                    if face["label"] != "Unknown":
                        person["label"] = f"Person ({face['label']})"
                    person["face_id"] = face["label"]
                    faces.remove(face) # Face is accounted for
            final_detections.append(person)
            
        # Add remaining faces that weren't inside a person box (safety)
        final_detections.extend(faces)

        # 4. Simple Tracking / ID Assignment
        self._update_tracker(final_detections)

        # Count objects
        counts = Counter([obj["label"] for obj in final_detections])

        # Get annotated frame
        annotated_frame = self._annotate_frame(frame, final_detections)

        return {
            "objects": final_detections,
            "counts": dict(counts),
            "image": annotated_frame,
            "frame_count": len(final_detections)
        }

    def _update_tracker(self, new_detections: List[Dict]):
        """Simple centroid-based tracking for persistence"""
        # (Simplified tracker logic for CPU efficiency)
        new_centers = [obj["center"] for obj in new_detections]
        
        # Match with existing objects
        for obj in new_detections:
            best_id = -1
            min_dist = 50 # Max pixels to consider same object
            
            center = obj["center"]
            for obj_id, data in self.objects.items():
                dist = np.sqrt((center[0]-data["center"][0])**2 + (center[1]-data["center"][1])**2)
                if dist < min_dist:
                    best_id = obj_id
                    min_dist = dist
            
            if best_id != -1:
                obj["id"] = best_id
                self.objects[best_id].update({"center": center, "last_seen": self.frame_count})
            else:
                obj["id"] = self.next_id
                self.objects[self.next_id] = {"label": obj["label"], "center": center, "last_seen": self.frame_count}
                self.next_id += 1

        # Cleanup old objects
        expired = [obj_id for obj_id, data in self.objects.items() 
                   if self.frame_count - data["last_seen"] > self.max_lost_frames]
        for obj_id in expired:
            del self.objects[obj_id]

    def _annotate_frame(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw detections and labels on the frame"""
        img = frame.copy()
        for obj in detections:
            x1, y1, x2, y2 = obj["bbox"]
            label = obj["label"]
            obj_id = obj["id"]
            color = (0, 255, 0) if obj["type"] == "object" else (255, 0, 0)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"#{obj_id} {label}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return img

    def get_frame_base64(self, frame: np.ndarray) -> str:
        """Convert frame to base64 for web transmission"""
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        import base64
        return base64.b64encode(buffer).decode('utf-8')

    def describe_scene_simple(self) -> str:
        """Get a simple text description of the scene"""
        detection = self.detect_objects()
        counts = detection["counts"]

        if not counts:
            return "I don't see anything notable."

        phrases = []
        for label, count in counts.items():
            if count == 1:
                phrases.append(f"a {label}")
            else:
                phrases.append(f"{count} {label}s")

        if len(phrases) == 1:
            return f"I see {phrases[0]}."
        elif len(phrases) == 2:
            return f"I see {phrases[0]} and {phrases[1]}."
        else:
            return "I see " + ", ".join(phrases[:-1]) + f", and {phrases[-1]}."


# Global instance
detector = VisionDetector()
