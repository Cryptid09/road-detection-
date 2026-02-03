"""
Camera Module
Supports both USB webcam (OpenCV) and Raspberry Pi Camera (picamera2)
"""

import cv2
import numpy as np
from typing import Optional, Tuple

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    Picamera2 = None


class Camera:
    """
    Unified camera interface for USB webcam and Pi Camera
    """
    
    def __init__(self, 
                 camera_index: int = 0,
                 use_pi_camera: bool = False,
                 width: int = 640,
                 height: int = 480,
                 fps: int = 30):
        """
        Initialize camera
        
        Args:
            camera_index: USB webcam index (0, 1, 2, etc.)
            use_pi_camera: If True, use Pi Camera (ribbon), else USB webcam
            width: Camera resolution width
            height: Camera resolution height
            fps: Target FPS
        """
        self.camera_index = camera_index
        self.use_pi_camera = use_pi_camera
        self.width = width
        self.height = height
        self.fps = fps
        
        self.cap = None
        self.picam2 = None
        self._initialize_camera()
    
    def _initialize_camera(self):
        """Initialize the appropriate camera based on configuration"""
        if self.use_pi_camera:
            if not PICAMERA2_AVAILABLE:
                raise RuntimeError(
                    "picamera2 not available. Install with: pip install picamera2"
                )
            self._init_pi_camera()
        else:
            self._init_usb_camera()
    
    def _init_usb_camera(self):
        """Initialize USB webcam using OpenCV"""
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera at index {self.camera_index}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Verify actual resolution
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"USB Camera initialized: {actual_width}x{actual_height}")
    
    def _init_pi_camera(self):
        """Initialize Raspberry Pi Camera using picamera2"""
        self.picam2 = Picamera2()
        
        # Configure camera
        config = self.picam2.create_video_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},
            controls={"FrameRate": self.fps}
        )
        self.picam2.configure(config)
        self.picam2.start()
        
        print(f"Pi Camera initialized: {self.width}x{self.height}")
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read frame from camera
        
        Returns:
            Tuple of (success, frame)
            - success: True if frame was read successfully
            - frame: BGR image array (H, W, 3) or None if failed
        """
        if self.use_pi_camera:
            return self._read_pi_camera()
        else:
            return self._read_usb_camera()
    
    def _read_usb_camera(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame from USB webcam"""
        ret, frame = self.cap.read()
        if ret:
            # OpenCV returns BGR, which is what we need
            return True, frame
        return False, None
    
    def _read_pi_camera(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame from Pi Camera"""
        try:
            # picamera2 returns RGB, convert to BGR for OpenCV compatibility
            frame_rgb = self.picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            return True, frame_bgr
        except Exception as e:
            print(f"Error reading from Pi Camera: {e}")
            return False, None
    
    def release(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
        if self.picam2 is not None:
            self.picam2.stop()
            self.picam2.close()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()


def list_available_cameras(max_test: int = 5) -> list:
    """
    List all available USB cameras
    
    Args:
        max_test: Maximum camera index to test
        
    Returns:
        List of available camera indices
    """
    available = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Try to read a frame to verify it works
            ret, _ = cap.read()
            if ret:
                available.append(i)
            cap.release()
    return available

