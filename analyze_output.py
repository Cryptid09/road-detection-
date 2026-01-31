#!/usr/bin/env python3
"""
Analyze TFLite model output format
"""

import numpy as np
from src.inference import get_inference_engine
from src.config import MODEL_PATHS

def analyze_output():
    print("Analyzing TFLite Model Output Format")
    print("=" * 40)

    # Load model
    engine = get_inference_engine('tflite')
    engine.load_model(MODEL_PATHS['tflite'])

    # Create dummy input
    dummy_input = np.random.rand(1, 640, 640, 3).astype(np.float32)

    # Run inference
    output = engine.run_inference(dummy_input)

    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")

    # For shape (1, 6, 8400), let's examine the structure
    batch_size, num_values, num_detections = output.shape
    print(f"Batch size: {batch_size}")
    print(f"Values per detection: {num_values}")
    print(f"Number of detections: {num_detections}")

    # Look at first few detections
    print("\nFirst 5 detections (first 6 values each):")
    for i in range(min(5, num_detections)):
        detection = output[0, :, i]
        print(f"Detection {i}: {detection}")

    # Analyze value ranges
    print("\nValue ranges:")
    for i in range(num_values):
        values = output[0, i, :]
        print(f"Value {i}: min={values.min():.3f}, max={values.max():.3f}, mean={values.mean():.3f}")

    # Check if values look like YOLO format
    print("\nChecking if values look like YOLO format:")
    sample_detection = output[0, :, 0]  # First detection
    print(f"Sample detection: {sample_detection}")

    # YOLO format typically has:
    # x, y, w, h (normalized 0-1), conf (0-1), class_scores (0-1)
    x, y, w, h, conf, class_id = sample_detection

    print("\nInterpreting as YOLO format:")
    print(f"x: {x:.3f}")
    print(f"y: {y:.3f}")
    print(f"w: {w:.3f}")
    print(f"h: {h:.3f}")
    print(f"conf: {conf:.3f}")
    print(f"class_id: {class_id:.3f}")

if __name__ == "__main__":
    analyze_output()