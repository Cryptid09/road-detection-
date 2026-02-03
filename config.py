"""
Configuration file for Road Anomaly Detection System
All parameters are configurable here for easy tuning
"""

import os

# Model Configuration
MODEL_TYPE = "tflite"  # Options: "onnx" or "tflite"
MODEL_PATH_ONNX = os.path.join("model", "best.onnx")
MODEL_PATH_TFLITE = os.path.join("model", "best_int8.tflite")

# Detection Parameters
CONFIDENCE_THRESHOLD = 0.6
INPUT_SIZE = 640  # Model input size (tested at 640, can use 320/416 for Pi)
NMS_THRESHOLD = 0.45  # Non-Maximum Suppression threshold

# Camera Configuration
CAMERA_INDEX = 0  # USB webcam index (try 0, 1, 2, etc.)
USE_PI_CAMERA = False  # Set to True on Raspberry Pi with ribbon camera
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

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
LOG_INCLUDE_IMAGE = False  # Set to True to save images with detections

# Display Configuration
SHOW_DISPLAY = True  # Set to False for headless operation
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

