"""
Configuration file for Road Anomaly Detection System
All parameters are configurable here for easy tuning
"""

import os

# Model Configuration
MODEL_TYPE = "tflite"  # Options: "onnx" or "tflite"
MODEL_PATH_ONNX = os.path.join("model", "best.onnx")
MODEL_PATH_TFLITE = os.path.join("model", "best_int8.tflite")  # Upload new INT8 quantized model from Kaggle

# Detection Parameters
# For Pi 3: Use 0.7 to reduce false positives and CPU load
# For Pi 4/5: Can use 0.5-0.6 for better detection
CONFIDENCE_THRESHOLD = 0.7  # Pi 3 optimized: higher threshold reduces false positives

# Desired model input size.
# IMPORTANT: Must match your exported TFLite model's fixed input shape!
# - For Pi 3: Use 320x320 or 256x256 (export model with imgsz=320 or imgsz=256)
# - For Pi 4/5: Can use 416x416 or 320x320
# - Current model (best_int8.tflite) may be 640x640 - check model export
# - To get FPS benefit, you MUST export a matching fixed-size model (see PERFORMANCE_OPTIMIZATION.md)
INPUT_SIZE = 320  # Will be auto-detected from model if mismatch (with warning)
NMS_THRESHOLD = 0.45  # Non-Maximum Suppression threshold

# Camera Configuration
CAMERA_INDEX = 0  # USB webcam index (try 0, 1, 2, etc.)
USE_PI_CAMERA = True  # Set to True on Raspberry Pi with ribbon camera
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 15  # Reduced from 30 to reduce processing overhead

# Frame processing configuration
# Process every Nth frame to reduce load on CPU.
# - Pi 3: Use 2 (process every 2nd frame) - MANDATORY for acceptable FPS
# - Pi 4: Use 1-2 depending on input size
# - Pi 5: Can use 1 (process every frame)
FRAME_SKIP = 2  # Pi 3 optimized: process every 2nd frame

# TFLite runtime performance
# Tune this per device:
# - Pi 3: try 2
# - Pi 4: try 3-4
# - Pi 5: try 4
TFLITE_NUM_THREADS = 2

# OpenCV threading (sometimes helps, sometimes hurts on small Pis; keep low)
OPENCV_NUM_THREADS = 1

# Performance Targets
FPS_TARGET = 5  # Minimum FPS target
FPS_DISPLAY_INTERVAL = 30  # Update FPS display every N frames

# Class Labels (from classes.txt)
CLASS_NAMES = {
    0: "pothole",
    1: "crack"
}

# Logging Configuration
LOG_DIR = "logs"
LOG_FILENAME = "detections.csv"
# For Pi 3: Disable image saving to reduce I/O overhead (slow SD cards)
# For Pi 4/5: Can enable if using fast storage (USB SSD)
LOG_INCLUDE_IMAGE = False  # Pi 3 optimized: disable to reduce I/O overhead
LOG_IMAGES_DIR = os.path.join(LOG_DIR, "images")  # Directory for saved images

# Display Configuration
# For Pi 3: ALWAYS disable display for best performance
# For Pi 4/5: Can enable if needed, but headless is faster
SHOW_DISPLAY = False  # Pi 3 optimized: disable display for better FPS
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

