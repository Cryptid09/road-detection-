#!/usr/bin/env python3
"""
Test script for road anomaly detection models
Uses laptop camera to test ONNX and TFLite inference
"""

import cv2
import time
from src.config import MODEL_TYPE, MODEL_PATHS, CONFIDENCE_THRESHOLD, CLASS_NAMES
from src.camera import Camera
from src.preprocessing import Preprocessor
from src.inference import get_inference_engine
from src.postprocessing import Postprocessor
from src.logging import Logger

def test_model(model_type, max_frames=50):
    """Test a specific model type for a limited number of frames"""
    print(f"\n=== Testing {model_type.upper()} Model ===")

    # Initialize components
    camera = Camera()
    preprocessor = Preprocessor()
    inference_engine = get_inference_engine(model_type)
    postprocessor = Postprocessor()
    logger = Logger()

    # Load model
    model_path = MODEL_PATHS[model_type]
    try:
        inference_engine.load_model(model_path)
    except Exception as e:
        print(f"Failed to load {model_type} model: {e}")
        return False

    # Open camera
    try:
        camera.open()
        print("Camera opened successfully")
    except Exception as e:
        print(f"Failed to open camera: {e}")
        return False

    frame_count = 0
    detection_count = 0

    # Add a test detection to verify logging works
    print("Adding test detection to verify logging...")
    test_detection = {
        'class_name': 'pothole',
        'confidence': 0.85,
        'bbox': (100, 150, 200, 250)
    }
    logger.log_detection(test_detection)
    print("✓ Test detection logged")

    try:
        while frame_count < max_frames:
            # Capture frame
            try:
                frame = camera.read_frame()
            except Exception as e:
                print(f"Failed to read frame: {e}")
                break

            # Preprocess
            preprocessed = preprocessor.preprocess(frame)

            # Run inference
            start_time = time.time()
            outputs = inference_engine.run_inference(preprocessed)
            inference_time = time.time() - start_time

            # Postprocess
            detections = postprocessor.postprocess(outputs, frame)

            # Count detections
            if detections:
                detection_count += len(detections)
                print(f"Frame {frame_count + 1}: Found {len(detections)} detection(s)")

                # Log detections
                for detection in detections:
                    logger.log_detection(detection)
                    print(f"  - {detection['class_name']} (conf: {detection['confidence']:.2f})")

            # Draw detections
            annotated_frame = postprocessor.draw_detections(frame.copy(), detections)

            # Add model info and timing
            cv2.putText(annotated_frame, f"Model: {model_type.upper()}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Inference: {inference_time:.3f}s", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Frame: {frame_count + 1}/{max_frames}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show frame
            cv2.imshow(f'Road Anomaly Detection - {model_type.upper()} Test', annotated_frame)

            frame_count += 1

            # Check for quit
            if cv2.waitKey(100) & 0xFF == ord('q'):  # 100ms delay
                break

    except KeyboardInterrupt:
        print("Test interrupted by user")
    except Exception as e:
        print(f"Error during testing: {e}")
    finally:
        camera.close()
        cv2.destroyAllWindows()

    print(f"Test completed: {frame_count} frames processed, {detection_count} detections logged")
    return True

def main():
    print("Road Anomaly Detection - Model Testing Script")
    print("=" * 50)

    # Test both models
    models_to_test = ['onnx', 'tflite']

    for model_type in models_to_test:
        success = test_model(model_type)
        if not success:
            print(f"Skipping {model_type} model due to errors")

    print("\nTesting complete!")
    print("Check logs/anomalies.csv for detection records")
    print("Press any key to exit...")
    cv2.waitKey(0)

if __name__ == "__main__":
    main()