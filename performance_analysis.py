#!/usr/bin/env python3
"""
Performance Analysis System for Road Anomaly Detection
"""

import time
import cv2
import numpy as np
from collections import defaultdict
from src.config import MODEL_TYPE, MODEL_PATHS, CONFIDENCE_THRESHOLD
from src.camera import Camera
from src.preprocessing import Preprocessor
from src.inference import get_inference_engine
from src.postprocessing import Postprocessor
from src.logging import Logger
from src.fps_counter import FPSCounter

class PerformanceAnalyzer:
    def __init__(self):
        self.reset_metrics()

    def reset_metrics(self):
        self.metrics = {
            'total_frames': 0,
            'total_detections': 0,
            'inference_times': [],
            'preprocessing_times': [],
            'postprocessing_times': [],
            'detections_per_frame': [],
            'class_distribution': defaultdict(int),
            'confidence_scores': [],
            'fps_values': []
        }

    def update_metrics(self, inference_time, preprocessing_time, postprocessing_time,
                      detections, fps):
        self.metrics['total_frames'] += 1
        self.metrics['total_detections'] += len(detections)
        self.metrics['inference_times'].append(inference_time)
        self.metrics['preprocessing_times'].append(preprocessing_time)
        self.metrics['postprocessing_times'].append(postprocessing_time)
        self.metrics['detections_per_frame'].append(len(detections))
        self.metrics['fps_values'].append(fps)

        for detection in detections:
            self.metrics['class_distribution'][detection['class_name']] += 1
            self.metrics['confidence_scores'].append(detection['confidence'])

    def get_summary(self):
        if self.metrics['total_frames'] == 0:
            return "No frames processed yet"

        confidence_info = ""
        if self.metrics['confidence_scores']:
            confidence_info = f"- Confidence range: {np.min(self.metrics['confidence_scores']):.3f} - {np.max(self.metrics['confidence_scores']):.3f}\n- Average confidence: {np.mean(self.metrics['confidence_scores']):.3f}"
        else:
            confidence_info = "- No detections made"

        summary = f"""
Performance Analysis Summary
{'='*30}
Total Frames Processed: {self.metrics['total_frames']}
Total Detections: {self.metrics['total_detections']}
Average Detections per Frame: {self.metrics['total_detections']/self.metrics['total_frames']:.2f}

Timing Analysis (ms):
- Preprocessing:  {np.mean(self.metrics['preprocessing_times'])*1000:.2f} avg, {np.max(self.metrics['preprocessing_times'])*1000:.2f} max
- Inference:      {np.mean(self.metrics['inference_times'])*1000:.2f} avg, {np.max(self.metrics['inference_times'])*1000:.2f} max
- Postprocessing: {np.mean(self.metrics['postprocessing_times'])*1000:.2f} avg, {np.max(self.metrics['postprocessing_times'])*1000:.2f} max

FPS Analysis:
- Average FPS: {np.mean(self.metrics['fps_values']):.2f}
- Min FPS: {np.min(self.metrics['fps_values']):.2f}
- Max FPS: {np.max(self.metrics['fps_values']):.2f}

Detection Analysis:
- Classes detected: {dict(self.metrics['class_distribution'])}
{confidence_info}
"""
        return summary

def analyze_performance(model_type='tflite', duration_seconds=30, use_camera=True):
    """
    Comprehensive performance analysis
    """
    print(f"Starting Performance Analysis for {model_type.upper()} model")
    print(f"Duration: {duration_seconds} seconds, Camera: {use_camera}")
    print("=" * 60)

    # Initialize components
    analyzer = PerformanceAnalyzer()
    camera = Camera() if use_camera else None
    preprocessor = Preprocessor()
    inference_engine = get_inference_engine(model_type)
    postprocessor = Postprocessor()
    logger = Logger()
    fps_counter = FPSCounter()

    # Load model
    inference_engine.load_model(MODEL_PATHS[model_type])

    # Open camera if needed
    if camera:
        camera.open()
        print("Camera opened successfully")
    else:
        print("Running without camera (using dummy frames)")

    start_time = time.time()
    frame_count = 0

    try:
        while (time.time() - start_time) < duration_seconds:
            # Generate or capture frame
            if camera:
                frame = camera.read_frame()
            else:
                # Create dummy frame for testing
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

            frame_count += 1

            # Preprocessing timing
            prep_start = time.time()
            preprocessed = preprocessor.preprocess(frame)
            prep_time = time.time() - prep_start

            # Inference timing
            infer_start = time.time()
            outputs = inference_engine.run_inference(preprocessed)
            infer_time = time.time() - infer_start

            # Postprocessing timing
            post_start = time.time()
            detections = postprocessor.postprocess(outputs, frame)
            post_time = time.time() - post_start

            # Update FPS
            fps_counter.update()
            current_fps = fps_counter.get_fps()

            # Update analyzer
            analyzer.update_metrics(infer_time, prep_time, post_time, detections, current_fps)

            # Log detections
            for detection in detections:
                logger.log_detection(detection)

            # Progress update
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                print(".1f")

    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
    except Exception as e:
        print(f"Error during analysis: {e}")
    finally:
        if camera:
            camera.close()

    # Print final summary
    print("\n" + analyzer.get_summary())

    return analyzer.metrics

if __name__ == "__main__":
    # Run performance analysis
    metrics = analyze_performance(
        model_type='tflite',
        duration_seconds=10,  # Short test
        use_camera=False     # Use dummy frames for testing
    )