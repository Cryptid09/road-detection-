#!/usr/bin/env python3
"""
Test script to verify model output format
Run this to understand how your model outputs class information
"""

import sys
import os
import numpy as np
import cv2

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.inference_tflite import TFLiteInferenceEngine
from src.preprocessing import preprocess_for_inference

def test_model_output():
    """Test and analyze model output format"""
    
    print("=" * 60)
    print("Model Output Format Verification")
    print("=" * 60)
    print()
    
    # Check if model exists
    model_path = config.MODEL_PATH_TFLITE
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please ensure your model file exists.")
        return
    
    print(f"Loading model: {model_path}")
    try:
        engine = TFLiteInferenceEngine(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    print("✅ Model loaded successfully")
    print()
    
    # Create a dummy test image (or use actual image if available)
    print("Creating test image...")
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    test_img.fill(128)  # Gray image
    
    # Preprocess
    print(f"Preprocessing with INPUT_SIZE={config.INPUT_SIZE}...")
    preprocessed, scale, pad = preprocess_for_inference(test_img, config.INPUT_SIZE)
    
    # Run inference
    print("Running inference...")
    output = engine.predict(preprocessed)
    
    print()
    print("=" * 60)
    print("OUTPUT ANALYSIS")
    print("=" * 60)
    print()
    
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print()
    
    # Analyze output structure
    num_predictions = output.shape[2]  # 8400
    num_channels = output.shape[1]      # 6 (or 4+num_classes)
    
    print(f"Number of predictions: {num_predictions}")
    print(f"Number of channels: {num_channels}")
    print()
    
    # Show first few predictions
    print("First 5 predictions (all channels):")
    print("-" * 60)
    sample = output[0, :, :5].T  # First 5, transposed for readability
    print("Format: [cx, cy, w, h, channel_4, channel_5]")
    print()
    for i, pred in enumerate(sample):
        print(f"Prediction {i}:")
        print(f"  cx: {pred[0]:.6f}  cy: {pred[1]:.6f}  w: {pred[2]:.6f}  h: {pred[3]:.6f}")
        print(f"  ch4: {pred[4]:.6f}  ch5: {pred[5]:.6f}")
    print()
    
    # Analyze each channel
    print("Channel Statistics (across all 8400 predictions):")
    print("-" * 60)
    channel_names = ["cx (center x)", "cy (center y)", "w (width)", "h (height)", 
                     "Channel 4", "Channel 5"]
    
    for i, name in enumerate(channel_names):
        channel_data = output[0, i, :]
        print(f"{name:20s}: min={channel_data.min():8.4f}, max={channel_data.max():8.4f}, "
              f"mean={channel_data.mean():8.4f}, std={channel_data.std():8.4f}")
    print()
    
    # Check for class information
    print("Class Detection Analysis:")
    print("-" * 60)
    ch4 = output[0, 4, :]
    ch5 = output[0, 5, :]
    
    # Check if channel 4 is "unused" (should be ~0)
    ch4_nonzero = np.count_nonzero(np.abs(ch4) > 0.001)
    ch4_percent = (ch4_nonzero / len(ch4)) * 100
    
    print(f"Channel 4 (unused?):")
    print(f"  Non-zero values: {ch4_nonzero}/{len(ch4)} ({ch4_percent:.2f}%)")
    if ch4_percent < 1:
        print("  ✅ Channel 4 appears to be unused (~0) - Format: [cx, cy, w, h, unused, score]")
    else:
        print("  ⚠️  Channel 4 has meaningful values - Might be class0_score")
    
    print()
    print(f"Channel 5 (score?):")
    print(f"  Range: [{ch5.min():.4f}, {ch5.max():.4f}]")
    print(f"  Values > 0.5: {np.count_nonzero(ch5 > 0.5)}/{len(ch5)}")
    if ch5.max() <= 1.0 and ch5.min() >= 0.0:
        print("  ✅ Channel 5 appears to be confidence scores [0, 1]")
    else:
        print("  ⚠️  Channel 5 values outside [0, 1] range")
    
    print()
    print("=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print()
    
    if num_channels == 6 and ch4_percent < 1:
        print("✅ Format appears to be: [cx, cy, w, h, unused, score]")
        print()
        print("⚠️  CLASS ID DETERMINATION:")
        print("   If format is [cx, cy, w, h, unused, score]:")
        print("   - Current code defaults all to class 0 (pothole)")
        print("   - Need to verify: Does your model have separate outputs?")
        print("   - Or: Is class determined by a different mechanism?")
        print()
        print("   Check your training configuration:")
        print("   - How many classes were defined?")
        print("   - Was it multi-class or binary classification?")
    elif num_channels >= 6:
        print("⚠️  Format might be: [cx, cy, w, h, class0_score, class1_score, ...]")
        print("   Need to update parsing to extract class from scores")
    else:
        print("❓ Unexpected format - Need manual inspection")
    
    print()
    print("=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    print()
    print("Based on this analysis:")
    print("1. Verify your training config to understand class handling")
    print("2. If model outputs separate predictions per class, update parsing")
    print("3. If single score per detection, verify how class is determined")
    print()
    print("Current implementation assumes format: [cx, cy, w, h, unused, score]")
    print("and defaults to class 0. If this is incorrect, update src/postprocessing.py")

if __name__ == "__main__":
    test_model_output()

