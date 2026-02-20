"""
Configuration file for Road Anomaly Detection System
All parameters are configurable here for easy tuning
"""

import os

# Model Configuration
MODEL_TYPE = "tflite"  # Options: "onnx" or "tflite"
MODEL_PATH_ONNX = os.path.join("model", "best.onnx")
MODEL_PATH_TFLITE = os.path.join("model", "best_int8.tflite")  # Upload new INT8 quantized model from Kaggle


CONFIDENCE_THRESHOLD = 0.7 

INPUT_SIZE = 320  # Will be auto-detected from model if mismatch (with warning)
NMS_THRESHOLD = 0.45  # Non-Maximum Suppression threshold

# Camera Configuration
CAMERA_INDEX = 0  # USB webcam index (try 0, 1, 2, etc.)
USE_PI_CAMERA = True  # Set to True on Raspberry Pi with ribbon camera
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 15  # Reduced from 30 to reduce processing overhead

FRAME_SKIP = 2

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

LOG_INCLUDE_IMAGE = False  
LOG_IMAGES_DIR = os.path.join(LOG_DIR, "images")  # Directory for saved images


SHOW_DISPLAY = False  # Pi 3 optimized: disable display for better FPS
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

