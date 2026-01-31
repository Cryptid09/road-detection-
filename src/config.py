# Configuration parameters for the road anomaly detection system

# Model configuration
MODEL_TYPE = 'tflite'  # Options: 'onnx' or 'tflite' (tflite recommended for compatibility)
MODEL_PATHS = {
    'onnx': 'models/best.onnx',
    'tflite': 'models/best_int8.tflite'
}

# Inference parameters
INPUT_SIZE = 640
CONFIDENCE_THRESHOLD = 0.6

# Camera configuration
CAMERA_INDEX = 0  # 0 for default camera, or specific index

# Performance targets
FPS_TARGET = 5

# Logging configuration
LOG_FILE = 'logs/anomalies.csv'
LOG_HEADERS = ['timestamp', 'class_name', 'confidence', 'x1', 'y1', 'x2', 'y2']

# Class names (corresponding to model outputs)
CLASS_NAMES = {
    0: 'pothole',
    1: 'crack'
}