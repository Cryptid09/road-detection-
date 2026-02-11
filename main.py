#!/usr/bin/env python3
"""
Main Inference Script
Real-time road anomaly detection using YOLOv8 on Raspberry Pi
"""

import cv2
import numpy as np
import sys
import os
import argparse

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.camera import Camera
from src.preprocessing import preprocess_for_inference, denormalize_coordinates
from src.postprocessing import parse_yolo_output, convert_to_pixel_coords
from src.fps_counter import FPSCounter
from src.event_logger import EventLogger
from src.visualization import draw_detections, draw_fps

# Import TFLite inference engine (primary)
from src.inference_tflite import TFLiteInferenceEngine

# Import ONNX inference engine (optional - may not be available on Pi)
try:
    from src.inference_onnx import ONNXInferenceEngine
    ONNX_AVAILABLE = True
except ImportError:
    ONNXInferenceEngine = None
    ONNX_AVAILABLE = False
    print("Warning: ONNX Runtime not available. Using TFLite only.")

# Validate MODEL_TYPE
if config.MODEL_TYPE not in ["onnx", "tflite"]:
    raise ValueError(f"Unsupported MODEL_TYPE: {config.MODEL_TYPE}")

inference_engine = None


def initialize_model():
    """Initialize the inference engine based on config"""
    global inference_engine
    
    if config.MODEL_TYPE == "onnx":
        if not ONNX_AVAILABLE:
            print("ERROR: ONNX Runtime not available on this system")
            print("Falling back to TFLite model...")
            if os.path.exists(config.MODEL_PATH_TFLITE):
                inference_engine = TFLiteInferenceEngine(config.MODEL_PATH_TFLITE)
                print("Model initialized: TFLite (fallback)")
                return
            else:
                raise RuntimeError("ONNX not available and TFLite model not found")
        
        model_path = config.MODEL_PATH_ONNX
        if not os.path.exists(model_path):
            print(f"Warning: ONNX model not found at {model_path}")
            print("Falling back to TFLite model...")
            if os.path.exists(config.MODEL_PATH_TFLITE):
                inference_engine = TFLiteInferenceEngine(config.MODEL_PATH_TFLITE)
                print("Model initialized: TFLite (fallback)")
                return
            else:
                raise FileNotFoundError(f"Model not found: {model_path}")
        inference_engine = ONNXInferenceEngine(model_path)
    
    elif config.MODEL_TYPE == "tflite":
        model_path = config.MODEL_PATH_TFLITE
        if not os.path.exists(model_path):
            print(f"Warning: TFLite model not found at {model_path}")
            if ONNX_AVAILABLE and os.path.exists(config.MODEL_PATH_ONNX):
                print("Falling back to ONNX model...")
                inference_engine = ONNXInferenceEngine(config.MODEL_PATH_ONNX)
                print("Model initialized: ONNX (fallback)")
                return
            else:
                raise FileNotFoundError(f"Model not found: {model_path}")
        inference_engine = TFLiteInferenceEngine(model_path)
    
    print(f"Model initialized: {config.MODEL_TYPE}")


def process_frame(frame, original_shape):
    """
    Process a single frame through the inference pipeline
    
    Args:
        frame: Input BGR image
        original_shape: (height, width) of original frame
    
    Returns:
        List of detections with pixel coordinates
    """
    # Preprocess - use CHW for ONNX, HWC for TFLite
    use_chw = config.MODEL_TYPE == "onnx"
    preprocessed, scale, pad = preprocess_for_inference(frame, config.INPUT_SIZE, use_chw=use_chw)
    
    # Run inference
    raw_output = inference_engine.predict(preprocessed)
    
    # Parse output
    detections = parse_yolo_output(
        raw_output,
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
        nms_threshold=config.NMS_THRESHOLD,
        num_classes=len(config.CLASS_NAMES)
    )
    
    # Convert to pixel coordinates
    pixel_detections = convert_to_pixel_coords(
        detections,
        scale,
        pad,
        original_shape,
        config.INPUT_SIZE
    )
    
    return pixel_detections


def main():
    """Main inference loop"""
    parser = argparse.ArgumentParser(description="Road Anomaly Detection System")
    parser.add_argument("--camera", type=int, default=None,
                       help="Camera index (overrides config)")
    parser.add_argument("--no-display", action="store_true",
                       help="Run in headless mode (no display)")
    parser.add_argument("--model-type", type=str, choices=["onnx", "tflite"],
                       default=None, help="Model type (overrides config)")
    
    args = parser.parse_args()
    
    # Override config if arguments provided
    if args.camera is not None:
        config.CAMERA_INDEX = args.camera
    if args.no_display:
        config.SHOW_DISPLAY = False
    if args.model_type is not None:
        config.MODEL_TYPE = args.model_type
    
    print("=" * 60)
    print("Road Anomaly Detection System")
    print("=" * 60)
    print(f"Model Type: {config.MODEL_TYPE}")
    print(f"Input Size: {config.INPUT_SIZE}x{config.INPUT_SIZE}")
    print(f"Confidence Threshold: {config.CONFIDENCE_THRESHOLD}")
    print(f"Camera: {'Pi Camera' if config.USE_PI_CAMERA else f'USB {config.CAMERA_INDEX}'}")
    print(f"Display: {'Enabled' if config.SHOW_DISPLAY else 'Disabled'}")
    print("=" * 60)
    
    # Initialize model
    try:
        initialize_model()
    except Exception as e:
        print(f"Error initializing model: {e}")
        sys.exit(1)
    
    # Initialize camera
    try:
        camera = Camera(
            camera_index=config.CAMERA_INDEX,
            use_pi_camera=config.USE_PI_CAMERA,
            width=config.CAMERA_WIDTH,
            height=config.CAMERA_HEIGHT,
            fps=config.CAMERA_FPS
        )
    except Exception as e:
        print(f"Error initializing camera: {e}")
        print("\nTip: Use 'python detect_camera.py' to find available cameras")
        sys.exit(1)
    
    # Initialize FPS counter
    fps_counter = FPSCounter(window_size=config.FPS_DISPLAY_INTERVAL)
    
    # Initialize event logger
    event_logger = EventLogger(
        log_dir=config.LOG_DIR,
        filename=config.LOG_FILENAME,
        save_images=config.LOG_INCLUDE_IMAGE,
        images_dir=config.LOG_IMAGES_DIR
    )
    print(f"Logging to: {event_logger.get_log_path()}")
    if config.LOG_INCLUDE_IMAGE:
        print(f"Images will be saved to: {config.LOG_IMAGES_DIR}")
    
    # Main loop
    frame_count = 0
    consecutive_failures = 0
    max_failures = 5
    print("\nStarting inference loop... (Press 'q' to quit)")
    
    try:
        while True:
            # Read frame
            ret, frame = camera.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    print(f"Failed to read frame from camera ({consecutive_failures} consecutive failures)")
                    break
                continue
            
            # Reset failure counter on successful read
            consecutive_failures = 0
            original_shape = frame.shape[:2]  # (height, width)
            
            # Process frame
            detections = process_frame(frame, original_shape)
            
            # Update FPS
            fps = fps_counter.update()
            
            # Log detections (with image if enabled)
            if detections:
                # Save annotated frame if image logging is enabled
                frame_to_log = frame.copy() if config.LOG_INCLUDE_IMAGE else None
                event_logger.log_detections(detections, config.CLASS_NAMES, frame_to_log)
                print(f"Frame {frame_count}: Detected {len(detections)} anomalies")
                if config.LOG_INCLUDE_IMAGE:
                    print(f"  → Image saved to logs/images/")
            
            # Draw visualizations
            if config.SHOW_DISPLAY:
                # Draw detections
                vis_frame = draw_detections(frame, detections, config.CLASS_NAMES)
                
                # Draw FPS
                vis_frame = draw_fps(vis_frame, fps)
                
                # Resize for display if needed
                if vis_frame.shape[1] > config.DISPLAY_WIDTH or vis_frame.shape[0] > config.DISPLAY_HEIGHT:
                    scale = min(
                        config.DISPLAY_WIDTH / vis_frame.shape[1],
                        config.DISPLAY_HEIGHT / vis_frame.shape[0]
                    )
                    new_w = int(vis_frame.shape[1] * scale)
                    new_h = int(vis_frame.shape[0] * scale)
                    vis_frame = cv2.resize(vis_frame, (new_w, new_h))
                
                # Display
                cv2.imshow("Road Anomaly Detection", vis_frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            
            # Print FPS periodically
            if frame_count % config.FPS_DISPLAY_INTERVAL == 0:
                print(f"FPS: {fps:.2f} | Frames processed: {frame_count}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        camera.release()
        if config.SHOW_DISPLAY:
            cv2.destroyAllWindows()
        print(f"\nProcessed {frame_count} frames")
        print(f"Average FPS: {fps_counter.get_fps():.2f}")
        print(f"Logs saved to: {event_logger.get_log_path()}")


if __name__ == "__main__":
    main()

