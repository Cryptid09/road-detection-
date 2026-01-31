#!/usr/bin/env python3
"""
Test postprocessing with actual model output
"""

import numpy as np
from src.postprocessing import Postprocessor
from src.inference import get_inference_engine
from src.config import MODEL_PATHS

def test_postprocessing():
    print("Testing Postprocessing with Real Model Output")
    print("=" * 45)

    # Load model and get output
    engine = get_inference_engine('tflite')
    engine.load_model(MODEL_PATHS['tflite'])

    # Create dummy input (but let's try with a simple pattern to see if model responds)
    dummy_input = np.random.rand(1, 640, 640, 3).astype(np.float32)
    # Make some pixels brighter to simulate potential features
    dummy_input[0, 320:330, 320:330, :] = 1.0  # Bright square in center

    output = engine.run_inference(dummy_input)
    print(f"Model output shape: {output.shape}")

    # Test postprocessing
    postprocessor = Postprocessor(confidence_threshold=0.001)  # Lower threshold for testing

    # Create a dummy original image
    original_image = np.zeros((480, 640, 3), dtype=np.uint8)

    detections = postprocessor.postprocess(output, original_image)

    print(f"Number of detections found: {len(detections)}")

    if detections:
        print("Sample detections:")
        for i, detection in enumerate(detections[:3]):  # Show first 3
            print(f"  Detection {i}: {detection}")
    else:
        print("No detections above threshold")

        # Let's check what the highest confidence detection looks like
        output_flat = output[0]  # Remove batch dimension
        confidences = output_flat[4, :]  # Confidence scores
        max_conf_idx = np.argmax(confidences)
        max_conf = confidences[max_conf_idx]

        print(f"Highest confidence: {max_conf:.6f} at index {max_conf_idx}")
        detection = output_flat[:, max_conf_idx]
        print(f"Full detection: {detection}")

if __name__ == "__main__":
    test_postprocessing()