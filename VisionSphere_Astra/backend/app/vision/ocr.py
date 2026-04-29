"""
OCR (Optical Character Recognition) using EasyOCR
Reads text from camera frames or screen captures
"""
import easyocr
import numpy as np
from typing import List, Dict, Optional, Tuple
import cv2
from app.config import settings


class OCRReader:
    """Read text from images using EasyOCR"""

    def __init__(self):
        # EasyOCR with CPU mode
        # languages: list of language codes
        self.reader = None
        self.is_loaded = False

    def load_model(self):
        """Load OCR model (lazy loading)"""
        if not self.is_loaded:
            print("Loading EasyOCR model...")
            # gpu=False for CPU-only
            # download_enabled=True to download models on first run
            self.reader = easyocr.Reader(
                ['en'],
                gpu=False,
                download_enabled=True,
                verbose=False
            )
            self.is_loaded = True
            print("EasyOCR model loaded")

    def read_text(self, image: np.ndarray) -> Dict:
        """
        Extract text from image

        Returns:
        {
            "text": "Full text content",
            "regions": [
                {
                    "text": "line of text",
                    "confidence": 0.95,
                    "bbox": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                }
            ]
        }
        """
        if not self.is_loaded:
            self.load_model()

        # Run OCR
        results = self.reader.readtext(
            image,
            min_size=10,  # Minimum text size to detect
            text_threshold=0.5,  # Confidence threshold
            link_threshold=0.4,
            canvas_size=1280,  # Max image size (resize if larger)
            mag_ratio=1.0  # No magnification
        )

        # Process results
        regions = []
        full_text = []

        for (bbox, text, confidence) in results:
            if confidence >= 0.5:  # Only include confident detections
                regions.append({
                    "text": text,
                    "confidence": round(confidence, 3),
                    "bbox": [point.tolist() for point in bbox]
                })
                full_text.append(text)

        return {
            "text": " ".join(full_text),
            "regions": regions,
            "line_count": len(regions)
        }

    def read_text_from_base64(self, image_b64: str) -> Dict:
        """Read text from base64 encoded image"""
        import base64
        img_bytes = base64.b64decode(image_b64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return self.read_text(img)

    def find_specific_text(self, image: np.ndarray, search_term: str) -> List[Dict]:
        """Find specific text in image"""
        result = self.read_text(image)
        search_lower = search_term.lower()

        matches = []
        for region in result["regions"]:
            if search_lower in region["text"].lower():
                matches.append(region)

        return matches


# Global instance (lazy loaded)
ocr_reader: Optional[OCRReader] = None


def get_ocr() -> OCRReader:
    """Get or create OCR engine"""
    global ocr_reader
    if ocr_reader is None:
        ocr_reader = OCRReader()
    return ocr_reader
