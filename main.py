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
import time

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
                inference_engine = TFLiteInferenceEngine(
                    config.MODEL_PATH_TFLITE,
                    num_threads=config.TFLITE_NUM_THREADS
                )
                print("Model initialized: TFLite (fallback)")
                return
            else:
                raise RuntimeError("ONNX not available and TFLite model not found")
        
        model_path = config.MODEL_PATH_ONNX
        if not os.path.exists(model_path):
            print(f"Warning: ONNX model not found at {model_path}")
            print("Falling back to TFLite model...")
            if os.path.exists(config.MODEL_PATH_TFLITE):
                inference_engine = TFLiteInferenceEngine(
                    config.MODEL_PATH_TFLITE,
                    num_threads=config.TFLITE_NUM_THREADS
                )
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
        inference_engine = TFLiteInferenceEngine(
            model_path,
            num_threads=config.TFLITE_NUM_THREADS
        )
        
        # Auto-detect model input size and update config if mismatch
        # TFLite models often have fixed input shapes (e.g., [1, 320, 320, 3] or [1, 640, 640, 3])
        if hasattr(inference_engine, 'input_shape') and len(inference_engine.input_shape) >= 2:
            model_input_size = inference_engine.input_shape[1]  # Usually H or W (square)
            if model_input_size != config.INPUT_SIZE:
                print(f"\n⚠️  WARNING: Model expects {model_input_size}x{model_input_size} input, but config.INPUT_SIZE = {config.INPUT_SIZE}")
                print(f"   Auto-adjusting INPUT_SIZE to {model_input_size} to match model")
                config.INPUT_SIZE = model_input_size
    
    print(f"Model initialized: {config.MODEL_TYPE}")
    if config.MODEL_TYPE == "tflite":
        print(f"TFLite threads: {config.TFLITE_NUM_THREADS}")
        print(f"Using input size: {config.INPUT_SIZE}x{config.INPUT_SIZE}")


def process_frame(frame, original_shape, is_rgb=False, profile=False):
    """
    Process a single frame through the inference pipeline
    
    Args:
        frame: Input image (BGR from USB camera, RGB from Pi camera)
        original_shape: (height, width) of original frame
        is_rgb: True if frame is RGB (Pi Camera), False if BGR (USB camera)
        profile: If True, return timing information
    
    Returns:
        List of detections with pixel coordinates (or tuple with timings if profile=True)
    """
    timings = {}
    
    # Preprocess - use CHW for ONNX, HWC for TFLite
    use_chw = config.MODEL_TYPE == "onnx"
    # Check if model is actually INT8 quantized
    is_quantized = (config.MODEL_TYPE == "tflite" and 
                   hasattr(inference_engine, 'is_quantized') and 
                   inference_engine.is_quantized)
    
    t0 = time.time()
    preprocessed, scale, pad = preprocess_for_inference(
        frame, 
        config.INPUT_SIZE, 
        use_chw=use_chw,
        quantized=is_quantized,
        is_bgr=not is_rgb
    )
    timings['preprocess'] = time.time() - t0
    
    # Run inference
    t0 = time.time()
    raw_output = inference_engine.predict(preprocessed)
    timings['inference'] = time.time() - t0
    
    # Parse output
    t0 = time.time()
    detections = parse_yolo_output(
        raw_output,
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
        nms_threshold=config.NMS_THRESHOLD,
        num_classes=len(config.CLASS_NAMES)
    )
    timings['postprocess'] = time.time() - t0
    
    # Convert to pixel coordinates
    t0 = time.time()
    pixel_detections = convert_to_pixel_coords(
        detections,
        scale,
        pad,
        original_shape,
        config.INPUT_SIZE
    )
    timings['coords'] = time.time() - t0
    timings['total'] = sum(timings.values())
    
    if profile:
        return pixel_detections, timings
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
    parser.add_argument("--profile", action="store_true",
                       help="Enable performance profiling (shows timing breakdown)")
    
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
    frame_count = 0           # Number of processed frames
    frame_id = 0              # Number of captured frames (including skipped)
    consecutive_failures = 0
    max_failures = 5
    profile_enabled = args.profile
    timing_stats = {'preprocess': [], 'inference': [], 'postprocess': [], 'coords': [], 'total': []}
    
    print("\nStarting inference loop... (Press 'q' to quit)")
    if profile_enabled:
        print("Performance profiling enabled - timing breakdown will be shown")
    
    try:
        while True:
            # Read frame
            t_read_start = time.time()
            ret, frame = camera.read()
            t_read = time.time() - t_read_start
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    print(f"Failed to read frame from camera ({consecutive_failures} consecutive failures)")
                    break
                continue
            
            # Reset failure counter on successful read
            consecutive_failures = 0
            original_shape = frame.shape[:2]  # (height, width)

            # Frame skipping: process only every Nth frame to reduce CPU load
            frame_id += 1
            if config.FRAME_SKIP > 1 and (frame_id % config.FRAME_SKIP) != 0:
                continue
            
            # Process frame (Pi Camera outputs RGB, USB camera outputs BGR)
            is_rgb = config.USE_PI_CAMERA
            if profile_enabled:
                detections, timings = process_frame(frame, original_shape, is_rgb=is_rgb, profile=True)
                for key in timing_stats:
                    timing_stats[key].append(timings.get(key, 0))
            else:
                detections = process_frame(frame, original_shape, is_rgb=is_rgb)
            
            # Update FPS
            fps = fps_counter.update()
            
            # Log detections (with image if enabled in config)
            if detections:
                # Save annotated frame if image logging is enabled
                # Convert RGB to BGR for logging (event_logger expects BGR)
                frame_to_log = None
                if config.LOG_INCLUDE_IMAGE:
                    frame_to_log = frame.copy()
                    if is_rgb:
                        frame_to_log = cv2.cvtColor(frame_to_log, cv2.COLOR_RGB2BGR)
                event_logger.log_detections(detections, config.CLASS_NAMES, frame_to_log)
                print(f"Frame {frame_count}: Detected {len(detections)} anomalies")
                if config.LOG_INCLUDE_IMAGE:
                    print(f"  → Image saved to {config.LOG_IMAGES_DIR}/")
            
            # Draw visualizations
            if config.SHOW_DISPLAY:
                # Convert RGB to BGR for display (Pi Camera outputs RGB, USB outputs BGR)
                display_frame = frame.copy()
                if is_rgb:
                    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
                
                # Draw detections
                vis_frame = draw_detections(display_frame, detections, config.CLASS_NAMES)
                
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
                if profile_enabled and frame_count > 0:
                    # Calculate average timings
                    avg_timings = {key: np.mean(timing_stats[key]) for key in timing_stats}
                    print(f"  Timing breakdown (avg over {config.FPS_DISPLAY_INTERVAL} frames):")
                    print(f"    Preprocess: {avg_timings['preprocess']*1000:.1f}ms")
                    print(f"    Inference:  {avg_timings['inference']*1000:.1f}ms ({avg_timings['inference']/avg_timings['total']*100:.1f}%)")
                    print(f"    Postprocess: {avg_timings['postprocess']*1000:.1f}ms")
                    print(f"    Coords:      {avg_timings['coords']*1000:.1f}ms")
                    print(f"    Total:       {avg_timings['total']*1000:.1f}ms")
                    print(f"    Camera read: {t_read*1000:.1f}ms")
                    # Reset stats for next interval
                    timing_stats = {key: [] for key in timing_stats}
    
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

