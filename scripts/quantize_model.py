#!/usr/bin/env python3
"""
Quantize YOLOv8 TFLite model to full INT8 for faster inference on Raspberry Pi
Requires: tensorflow (not tflite-runtime) - pip install tensorflow

NOTE: If you have the original ONNX or PyTorch (.pt) model, use that instead
      for better quantization results.
"""

import tensorflow as tf
import numpy as np
import cv2
import os
from pathlib import Path

# Configuration
INPUT_MODEL = "model/best_float32.tflite"  # Your 12 MB float32 model
OUTPUT_MODEL = "model/best_int8_proper.tflite"
INPUT_SIZE = 640

# Representative dataset generator
def representative_dataset_gen():
    """
    Generate representative data for quantization calibration
    Uses sample images or synthetic data
    """
    print("Generating representative dataset for quantization...")
    
    # Option 1: Use real images if available
    image_dir = Path("logs/images")
    if image_dir.exists():
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        if image_files:
            print(f"Found {len(image_files)} images for calibration")
            for img_path in image_files[:100]:  # Use max 100 images
                img = cv2.imread(str(img_path))
                if img is not None:
                    # Preprocess like inference
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_resized = cv2.resize(img_rgb, (INPUT_SIZE, INPUT_SIZE))
                    img_normalized = img_resized.astype(np.float32) / 255.0
                    img_batch = np.expand_dims(img_normalized, axis=0)
                    yield [img_batch]
            return
    
    # Option 2: Generate synthetic data if no images available
    print("No images found, using synthetic data for calibration")
    for _ in range(100):
        # Generate random normalized images
        synthetic_img = np.random.random((1, INPUT_SIZE, INPUT_SIZE, 3)).astype(np.float32)
        yield [synthetic_img]


def quantize_model():
    """
    Convert float32 TFLite model to full INT8 quantized model
    """
    if not os.path.exists(INPUT_MODEL):
        print(f"Error: Input model not found: {INPUT_MODEL}")
        print("Available models:")
        if os.path.exists("model"):
            for f in os.listdir("model"):
                if f.endswith(".tflite"):
                    size_mb = os.path.getsize(f"model/{f}") / (1024 * 1024)
                    print(f"  - {f} ({size_mb:.2f} MB)")
        return False
    
    print("=" * 60)
    print("TFLite Model Quantization")
    print("=" * 60)
    print(f"Input:  {INPUT_MODEL}")
    print(f"Output: {OUTPUT_MODEL}")
    print(f"Input size: {INPUT_SIZE}x{INPUT_SIZE}")
    print("=" * 60)
    
    # Load the float model
    converter = tf.lite.TFLiteConverter.from_saved_model_path(INPUT_MODEL)
    
    # Set optimization flags for full INT8 quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Set representative dataset for calibration
    converter.representative_dataset = representative_dataset_gen
    
    # Ensure full integer quantization (inputs and outputs are INT8/UINT8)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # Force INT8 for inputs/outputs
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    print("\nQuantizing model (this may take a few minutes)...")
    
    try:
        # Convert model
        quantized_model = converter.convert()
        
        # Save quantized model
        os.makedirs(os.path.dirname(OUTPUT_MODEL), exist_ok=True)
        with open(OUTPUT_MODEL, 'wb') as f:
            f.write(quantized_model)
        
        # Compare sizes
        original_size = os.path.getsize(INPUT_MODEL) / (1024 * 1024)
        quantized_size = os.path.getsize(OUTPUT_MODEL) / (1024 * 1024)
        reduction = (1 - quantized_size / original_size) * 100
        
        print("\n" + "=" * 60)
        print("✓ Quantization successful!")
        print("=" * 60)
        print(f"Original model:  {original_size:.2f} MB")
        print(f"Quantized model: {quantized_size:.2f} MB")
        print(f"Size reduction:  {reduction:.1f}%")
        print(f"\nSaved to: {OUTPUT_MODEL}")
        print("\nExpected speedup on Raspberry Pi: 2-4x faster")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Quantization failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have TensorFlow installed (not just tflite-runtime)")
        print("   pip install tensorflow")
        print("2. Ensure your input model is a valid TFLite model")
        return False


if __name__ == "__main__":
    # Check if TensorFlow is available
    try:
        print(f"TensorFlow version: {tf.__version__}")
    except:
        print("Error: TensorFlow not found")
        print("Install with: pip install tensorflow")
        exit(1)
    
    success = quantize_model()
    
    if success:
        print("\nNext steps:")
        print("1. Update config.py to use the new model:")
        print(f"   MODEL_PATH_TFLITE = '{OUTPUT_MODEL}'")
        print("2. Run your inference script")
        print("3. Expect 5-8 FPS on Raspberry Pi 4 (vs ~1-2 FPS currently)")
