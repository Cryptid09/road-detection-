# Camera Module Documentation

## Overview

The Camera module (`src/camera.py`) is the first component in the road anomaly detection pipeline. It provides a clean, object-oriented interface for capturing video frames from a camera device using OpenCV. This module handles camera initialization, frame capture, and resource cleanup, ensuring reliable video input for the preprocessing stage.

### Key Features
- Simple camera device management
- Error handling for camera availability
- Resource-safe operations (proper cleanup)
- Configurable camera index
- Compatible with USB webcams and Pi Camera

### Dependencies
- `cv2` (OpenCV) - for video capture
- `src.config` - for camera index configuration

---

## Class: Camera

The `Camera` class encapsulates all camera-related operations.

### Constructor: `__init__(self, camera_index=CAMERA_INDEX)`

```python
class Camera:
    def __init__(self, camera_index=CAMERA_INDEX):
        self.camera_index = camera_index
        self.cap = None
        self.is_opened = False
```

**Line-by-line explanation:**
- `class Camera:` - Defines the Camera class
- `def __init__(self, camera_index=CAMERA_INDEX):` - Constructor method, takes optional camera_index parameter (defaults to config value)
- `self.camera_index = camera_index` - Stores the camera device index (0 for default camera, 1 for second camera, etc.)
- `self.cap = None` - Initializes the OpenCV VideoCapture object as None (will be created in open())
- `self.is_opened = False` - Boolean flag to track camera state

### Method: `open(self)`

```python
def open(self):
    """Open the camera stream."""
    self.cap = cv2.VideoCapture(self.camera_index)
    if not self.cap.isOpened():
        raise RuntimeError(f"Failed to open camera with index {self.camera_index}")
    self.is_opened = True
```

**Line-by-line explanation:**
- `def open(self):` - Method to initialize and open the camera
- `"""Open the camera stream."""` - Docstring describing the method
- `self.cap = cv2.VideoCapture(self.camera_index)` - Creates OpenCV VideoCapture object with specified camera index
- `if not self.cap.isOpened():` - Checks if camera opened successfully
- `raise RuntimeError(f"Failed to open camera with index {self.camera_index}")` - Raises error if camera fails to open
- `self.is_opened = True` - Sets state flag to True if successful

### Method: `read_frame(self)`

```python
def read_frame(self):
    """Read a single frame from the camera."""
    if not self.is_opened:
        raise RuntimeError("Camera is not opened")
    ret, frame = self.cap.read()
    if not ret:
        raise RuntimeError("Failed to read frame from camera")
    return frame
```

**Line-by-line explanation:**
- `def read_frame(self):` - Method to capture a single frame
- `"""Read a single frame from the camera."""` - Docstring
- `if not self.is_opened:` - Checks if camera is opened before reading
- `raise RuntimeError("Camera is not opened")` - Error if camera not ready
- `ret, frame = self.cap.read()` - Reads frame from camera; ret is success boolean, frame is numpy array
- `if not ret:` - Checks if frame read was successful
- `raise RuntimeError("Failed to read frame from camera")` - Error if frame read fails
- `return frame` - Returns the captured frame (BGR format, numpy array)

### Method: `close(self)`

```python
def close(self):
    """Close the camera stream."""
    if self.cap is not None:
        self.cap.release()
    self.is_opened = False
```

**Line-by-line explanation:**
- `def close(self):` - Method to close camera and free resources
- `"""Close the camera stream."""` - Docstring
- `if self.cap is not None:` - Safety check to avoid errors if cap is None
- `self.cap.release()` - Releases the OpenCV VideoCapture resources
- `self.is_opened = False` - Resets state flag

---

## Usage Pattern

The Camera class follows a standard resource management pattern:

```python
from src.camera import Camera

# Create camera instance
camera = Camera()  # or Camera(camera_index=1)

# Open camera
camera.open()

# Capture frames in loop
try:
    while True:
        frame = camera.read_frame()
        # Process frame...
        if some_condition:
            break
finally:
    # Always close camera
    camera.close()
```

---

## Integration with Pipeline

The Camera module feeds directly into the Preprocessing module:

```
Camera.read_frame() → Preprocessor.preprocess() → Inference Engine
```

### Data Flow:
1. `camera.read_frame()` returns numpy array (H, W, 3) in BGR format
2. `preprocessor.preprocess()` resizes to (640, 640, 3) and normalizes to [0,1]
3. Result goes to inference engine

### Error Handling:
- Camera not available → RuntimeError in `open()`
- Camera not opened → RuntimeError in `read_frame()`
- Frame read failure → RuntimeError in `read_frame()`

### Performance Considerations:
- Camera opening is done once at startup
- Frame reading is blocking but fast (< 10ms typically)
- Resource cleanup prevents camera lockup

---

## Configuration

Camera index is configured in `src/config.py`:

```python
CAMERA_INDEX = 0  # Default camera
```

For Raspberry Pi with Pi Camera:
- May need `CAMERA_INDEX = 0` or specific device path
- Ensure camera is enabled in raspi-config

---

## Testing

The camera module is tested in `system_validation.py`:

```python
def test_camera():
    camera = Camera()
    # Tests object creation
    return camera
```

For full testing with actual camera:
```python
python test_models.py  # Tests camera opening and frame capture
```

---

## Troubleshooting

### Common Issues:
1. **"Failed to open camera with index X"**
   - Check camera is connected
   - Try different camera index
   - Check camera permissions

2. **"Camera is not opened"**
   - Call `camera.open()` before `read_frame()`

3. **"Failed to read frame from camera"**
   - Camera may be in use by another application
   - Check camera cable/connection
   - Try restarting camera service

### Raspberry Pi Specific:
- Enable camera: `sudo raspi-config`
- Install OpenCV: `sudo apt-get install python3-opencv`
- Camera permissions: Add user to video group

---

## Code Quality Notes

- **Exception Safety**: All methods raise clear RuntimeError messages
- **Resource Management**: Proper cleanup in `close()` method
- **State Tracking**: `is_opened` flag prevents invalid operations
- **Documentation**: Comprehensive docstrings for all methods
- **Type Hints**: Not used (Python 3.7+ compatible), but clear from context
- **Thread Safety**: Not thread-safe; use one camera per thread