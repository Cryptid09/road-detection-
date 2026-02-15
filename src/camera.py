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
        # Try different backends in order of preference
        backends = [
            cv2.CAP_V4L2,      # V4L2 (Linux)
            cv2.CAP_ANY,       # Auto-detect
        ]
        
        for backend in backends:
            self.cap = cv2.VideoCapture(self.camera_index, backend)
            if self.cap.isOpened():
                # Try to read a test frame
                ret, test_frame = self.cap.read()
                if ret:
                    backend_name = {cv2.CAP_V4L2: "V4L2", cv2.CAP_ANY: "AUTO"}
                    print(f"Using backend: {backend_name.get(backend, 'Unknown')}")
                    break
                self.cap.release()
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera at index {self.camera_index}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Set buffer size to 1 to get latest frame
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Verify actual resolution
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"USB Camera initialized: {actual_width}x{actual_height}")
    
    def _init_pi_camera(self):
        """Initialize Raspberry Pi Camera using picamera2"""
        self.picam2 = Picamera2()
        
        # Configure camera for video streaming
        config = self.picam2.create_video_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},
            buffer_count=1  # Use 1 buffer to reduce latency
        )
        self.picam2.configure(config)
        
        # Set controls for better inference performance
        self.picam2.set_controls({
            "FrameRate": self.fps,
            "ExposureTime": 20000,  # Fixed exposure for consistent color
            "AnalogueGain": 1.0,     # Fixed gain for consistent color
            "AwbEnable": True        # Enable auto white balance for correct colors
        })
        
        self.picam2.start()
        
        # Warmup: Let camera stabilize (important for auto-exposure/gain)
        import time
        time.sleep(2)
        # Capture and discard a few frames
        for _ in range(5):
            try:
                self.picam2.capture_array()
            except:
                pass
        
        print(f"Pi Camera initialized: {self.width}x{self.height} @ {self.fps}fps")
    
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
            # picamera2 returns RGB directly - keep it as RGB to avoid double conversion
            # The preprocessing module will handle it correctly
            frame_rgb = self.picam2.capture_array()
            # Return RGB directly (not BGR) since preprocessing expects input and converts appropriately
            return True, frame_rgb
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

