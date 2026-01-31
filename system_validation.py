#!/usr/bin/env python3
"""
Complete System Validation - Step by Step Analysis
"""

import cv2
import numpy as np
import time
from src.config import MODEL_TYPE, MODEL_PATHS, CONFIDENCE_THRESHOLD, INPUT_SIZE, CLASS_NAMES
from src.camera import Camera
from src.preprocessing import Preprocessor
from src.inference import get_inference_engine
from src.postprocessing import Postprocessor
from src.logging import Logger
from src.fps_counter import FPSCounter

def validate_step(step_name, test_func):
    """Helper to validate each step"""
    print(f"\n{'='*20} {step_name} {'='*20}")
    try:
        result = test_func()
        print(f"✅ {step_name}: PASSED")
        return True, result
    except Exception as e:
        print(f"❌ {step_name}: FAILED - {e}")
        return False, None

def test_config():
    """Test configuration loading"""
    print(f"MODEL_TYPE: {MODEL_TYPE}")
    print(f"MODEL_PATHS: {MODEL_PATHS}")
    print(f"INPUT_SIZE: {INPUT_SIZE}")
    print(f"CONFIDENCE_THRESHOLD: {CONFIDENCE_THRESHOLD}")
    print(f"CLASS_NAMES: {CLASS_NAMES}")
    return True

def test_camera():
    """Test camera initialization (without opening)"""
    camera = Camera()
    print(f"Camera index: {camera.camera_index}")
    print("Camera object created successfully")
    return camera

def test_preprocessing():
    """Test preprocessing pipeline"""
    preprocessor = Preprocessor()

    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    print(f"Input image shape: {test_image.shape}")
    print(f"Input image dtype: {test_image.dtype}")
    print(f"Input value range: {test_image.min()} - {test_image.max()}")

    # Preprocess
    preprocessed = preprocessor.preprocess(test_image)
    print(f"Preprocessed shape: {preprocessed.shape}")
    print(f"Preprocessed dtype: {preprocessed.dtype}")
    print(f"Preprocessed value range: {preprocessed.min():.3f} - {preprocessed.max():.3f}")

    # Validate preprocessing
    assert preprocessed.shape == (1, INPUT_SIZE, INPUT_SIZE, 3), f"Wrong shape: {preprocessed.shape}"
    assert preprocessed.dtype == np.float32, f"Wrong dtype: {preprocessed.dtype}"
    assert 0 <= preprocessed.min() <= preprocessed.max() <= 1, f"Wrong value range: {preprocessed.min()} - {preprocessed.max()}"

    return preprocessor

def test_inference():
    """Test inference engine"""
    engine = get_inference_engine(MODEL_TYPE)
    engine.load_model(MODEL_PATHS[MODEL_TYPE])
    print(f"Model loaded: {MODEL_PATHS[MODEL_TYPE]}")

    # Create dummy input
    dummy_input = np.random.rand(1, INPUT_SIZE, INPUT_SIZE, 3).astype(np.float32)

    # Run inference
    start_time = time.time()
    output = engine.run_inference(dummy_input)
    inference_time = time.time() - start_time

    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"Inference time: {inference_time:.3f}s")

    # Validate output
    assert output.shape[0] == 1, f"Wrong batch size: {output.shape[0]}"
    assert output.dtype == np.float32, f"Wrong dtype: {output.dtype}"

    return engine, output

def test_postprocessing():
    """Test postprocessing"""
    postprocessor = Postprocessor()

    # Get sample output from inference test
    engine = get_inference_engine(MODEL_TYPE)
    engine.load_model(MODEL_PATHS[MODEL_TYPE])
    dummy_input = np.random.rand(1, INPUT_SIZE, INPUT_SIZE, 3).astype(np.float32)
    output = engine.run_inference(dummy_input)

    # Create dummy original image
    original_image = np.zeros((480, 640, 3), dtype=np.uint8)

    # Test postprocessing with very low threshold
    detections = postprocessor.postprocess(output, original_image)

    print(f"Detections found: {len(detections)}")
    if detections:
        print("Sample detection:", detections[0])

    # Test with normal threshold
    postprocessor_normal = Postprocessor()
    detections_normal = postprocessor_normal.postprocess(output, original_image)
    print(f"Detections above threshold ({CONFIDENCE_THRESHOLD}): {len(detections_normal)}")

    return postprocessor

def test_visualization():
    """Test visualization/drawing"""
    postprocessor = Postprocessor()

    # Create test image and detections
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_detections = [
        {
            'class_name': 'pothole',
            'confidence': 0.85,
            'bbox': (100, 100, 200, 200)
        },
        {
            'class_name': 'crack',
            'confidence': 0.72,
            'bbox': (300, 150, 450, 250)
        }
    ]

    # Draw detections
    annotated_image = postprocessor.draw_detections(test_image.copy(), test_detections)

    print(f"Original image shape: {test_image.shape}")
    print(f"Annotated image shape: {annotated_image.shape}")
    print(f"Test detections: {len(test_detections)}")

    # Save test image for verification
    cv2.imwrite('test_visualization.jpg', annotated_image)
    print("Test visualization saved as 'test_visualization.jpg'")

    return True

def test_logging():
    """Test logging system"""
    logger = Logger()

    # Test logging
    test_detection = {
        'class_name': 'pothole',
        'confidence': 0.85,
        'bbox': (100, 150, 200, 250)
    }

    logger.log_detection(test_detection)
    print("Test detection logged")

    # Check if file was created/updated
    import os
    if os.path.exists(logger.log_file):
        print(f"Log file exists: {logger.log_file}")
        with open(logger.log_file, 'r') as f:
            lines = f.readlines()
            print(f"Log file has {len(lines)} lines")
            if len(lines) > 1:  # Header + at least one detection
                print("Last line:", lines[-1].strip())
    else:
        raise FileNotFoundError("Log file was not created")

    return logger

def test_fps_counter():
    """Test FPS counter"""
    fps_counter = FPSCounter()

    # Simulate some frames
    for i in range(10):
        time.sleep(0.1)  # 100ms delay
        fps_counter.update()
        if i % 3 == 0:
            print(f"Frame {i+1}: FPS = {fps_counter.get_fps():.2f}")

    final_fps = fps_counter.get_fps()
    print(f"Final FPS: {final_fps:.2f}")

    # Should be around 10 FPS (1/0.1 = 10)
    expected_fps = 8.0  # Allow some tolerance
    assert expected_fps <= final_fps <= 12.0, f"FPS out of range: {final_fps}"

    return fps_counter

def main():
    """Run complete system validation"""
    print("ROAD ANOMALY DETECTION SYSTEM - COMPLETE VALIDATION")
    print("=" * 60)

    validation_results = []

    # Test each component
    validation_results.append(validate_step("Configuration", test_config))
    validation_results.append(validate_step("Camera Class", test_camera))
    validation_results.append(validate_step("Preprocessing", test_preprocessing))
    validation_results.append(validate_step("Inference Engine", lambda: test_inference()[0]))
    validation_results.append(validate_step("Postprocessing", test_postprocessing))
    validation_results.append(validate_step("Visualization", test_visualization))
    validation_results.append(validate_step("Logging", test_logging))
    validation_results.append(validate_step("FPS Counter", test_fps_counter))

    # Summary
    print(f"\n{'='*20} VALIDATION SUMMARY {'='*20}")
    passed = sum(1 for result, _ in validation_results if result)
    total = len(validation_results)
    print(f"Tests Passed: {passed}/{total}")

    if passed == total:
        print("🎉 ALL TESTS PASSED! System is ready for deployment.")
    else:
        print("⚠️  Some tests failed. Please review the errors above.")

    print("\nNext steps:")
    print("1. Test with real camera input: python test_models.py")
    print("2. Run performance analysis: python performance_analysis.py")
    print("3. Deploy main system: python main.py")

if __name__ == "__main__":
    main()