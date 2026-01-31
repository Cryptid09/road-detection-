#!/usr/bin/env python3
"""
Simple inference test without camera
Creates dummy input and tests model loading/inference
"""

import numpy as np
import time
from src.config import MODEL_PATHS
from src.inference import get_inference_engine

def test_inference_only():
    """Test inference engines without camera"""
    print("Testing Inference Engines (No Camera Required)")
    print("=" * 50)

    # Create dummy input (batch of 1, 640x640x3 RGB image)
    dummy_input = np.random.rand(1, 640, 640, 3).astype(np.float32)
    print(f"Created dummy input shape: {dummy_input.shape}")

    models_to_test = ['onnx', 'tflite']

    for model_type in models_to_test:
        print(f"\n--- Testing {model_type.upper()} Model ---")

        try:
            # Initialize engine
            inference_engine = get_inference_engine(model_type)

            # Load model
            model_path = MODEL_PATHS[model_type]
            print(f"Loading model from: {model_path}")
            inference_engine.load_model(model_path)

            # Run inference multiple times for timing
            num_runs = 5
            times = []

            for i in range(num_runs):
                start_time = time.time()
                outputs = inference_engine.run_inference(dummy_input)
                end_time = time.time()
                times.append(end_time - start_time)
                print(".3f")

            avg_time = sum(times) / len(times)
            print(".3f")
            print(f"Output shape: {outputs.shape if hasattr(outputs, 'shape') else type(outputs)}")

        except Exception as e:
            print(f"❌ Error testing {model_type}: {e}")
            continue

    print("\nInference testing complete!")

if __name__ == "__main__":
    test_inference_only()