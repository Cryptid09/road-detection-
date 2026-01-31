#!/usr/bin/env python3
"""
Real-time Road Anomaly Detection System
Main script for edge AI inference on Raspberry Pi
"""

import cv2
import time
from src.config import MODEL_TYPE, MODEL_PATHS, FPS_TARGET
from src.camera import Camera
from src.preprocessing import Preprocessor
from src.inference import get_inference_engine
from src.postprocessing import Postprocessor
from src.logging import Logger
from src.fps_counter import FPSCounter

def main():
    print("Starting Road Anomaly Detection System...")

    # Initialize components
    camera = Camera()
    preprocessor = Preprocessor()
    inference_engine = get_inference_engine()
    postprocessor = Postprocessor()
    logger = Logger()
    fps_counter = FPSCounter()

    # Load model
    model_path = MODEL_PATHS[MODEL_TYPE]
    inference_engine.load_model(model_path)

    # Open camera
    camera.open()

    print("System initialized. Press 'q' to quit.")

    try:
        while True:
            # Capture frame
            frame = camera.read_frame()

            # Start timing for FPS
            start_time = time.time()

            # Preprocess
            preprocessed = preprocessor.preprocess(frame)

            # Run inference
            outputs = inference_engine.run_inference(preprocessed)

            # Postprocess
            detections = postprocessor.postprocess(outputs, frame)

            # Log detections
            for detection in detections:
                logger.log_detection(detection)

            # Draw detections on frame
            annotated_frame = postprocessor.draw_detections(frame.copy(), detections)

            # Update FPS
            fps_counter.update()
            fps = fps_counter.get_fps()

            # Display FPS on frame
            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show frame
            cv2.imshow('Road Anomaly Detection', annotated_frame)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Check if we need to skip frames to maintain target FPS
            target_frame_time = 1.0 / FPS_TARGET
            if processing_time < target_frame_time:
                time.sleep(target_frame_time - processing_time)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        camera.close()
        cv2.destroyAllWindows()
        print("System shutdown complete.")

if __name__ == "__main__":
    main()