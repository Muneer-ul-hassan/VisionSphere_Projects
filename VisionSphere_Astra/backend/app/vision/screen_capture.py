"""
Screen Capture for Astra
Allows Astra to see your computer screen
"""
import numpy as np
import cv2
from PIL import Image
from typing import Optional, Tuple
import mss
import mss.tools


class ScreenCapture:
    """Capture screen for analysis"""

    def __init__(self):
        self.sct = mss.mss()
        self.is_initialized = False

    def capture_full_screen(self) -> np.ndarray:
        """Capture entire screen as numpy array (BGR format)"""
        with mss.mss() as sct:
            # Get primary monitor
            monitor = sct.monitors[0]  # All monitors combined

            # Capture screenshot
            screenshot = sct.grab(monitor)

            # Convert to numpy array
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            return img

    def capture_region(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        """Capture specific screen region"""
        with mss.mss() as sct:
            monitor = {
                "left": x,
                "top": y,
                "width": width,
                "height": height
            }

            screenshot = sct.grab(monitor)
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            return img

    def get_screen_resolution(self) -> Tuple[int, int]:
        """Get primary screen resolution"""
        with mss.mss() as sct:
            monitor = sct.monitors[1]  # Primary monitor
            return (monitor["width"], monitor["height"])

    def capture_to_base64(self, region: Optional[dict] = None) -> str:
        """Capture screen and return as base64"""
        if region:
            img = self.capture_region(
                region["x"],
                region["y"],
                region["width"],
                region["height"]
            )
        else:
            img = self.capture_full_screen()

        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        import base64
        return base64.b64encode(buffer).decode('utf-8')


# Global instance
screen_capture = ScreenCapture()
