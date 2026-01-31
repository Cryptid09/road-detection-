import cv2
from src.config import CAMERA_INDEX

class Camera:
    def __init__(self, camera_index=CAMERA_INDEX):
        self.camera_index = camera_index
        self.cap = None
        self.is_opened = False

    def open(self):
        """Open the camera stream."""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera with index {self.camera_index}")
        self.is_opened = True

    def read_frame(self):
        """Read a single frame from the camera."""
        if not self.is_opened:
            raise RuntimeError("Camera is not opened")
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from camera")
        return frame

    def close(self):
        """Close the camera stream."""
        if self.cap is not None:
            self.cap.release()
        self.is_opened = False