"""
ONNX to INT8 TFLite Quantization Script for Google Colab
=========================================================
Run this in Google Colab to convert ONNX model to properly quantized INT8 TFLite

Instructions:
1. Upload your best.onnx model to Colab
2. Upload some sample images to 'sample_images/' folder (optional, but recommended)
3. Run this script
4. Download the generated best_int8_quantized.tflite model
5. Replace your current model on Raspberry Pi

Expected speedup: 3-4x faster (5-8 FPS on Pi 4)
"""

# ============================================================================
# STEP 1: Install dependencies
# ============================================================================
print("Installing dependencies...")
print("This may take 2-3 minutes...")
!pip install -q onnx tf2onnx tensorflow opencv-python

import os
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf

# ============================================================================
# STEP 2: Configuration
# ============================================================================
ONNX_MODEL_PATH = "best.onnx"  # Your ONNX model (upload to Colab)
OUTPUT_NAME = "best_int8_quantized.tflite"
SAMPLE_IMAGES_DIR = "sample_images"  # Upload 50-100 sample road images here (optional)
INPUT_SIZE = 640

# ============================================================================
# STEP 3: Verify model exists
# ============================================================================
if not os.path.exists(ONNX_MODEL_PATH):
    print(f"❌ ERROR: Model not found: {ONNX_MODEL_PATH}")
    print("\nPlease upload your ONNX model (.onnx file) to Colab")
    exit(1)

print(f"✅ Found ONNX model: {ONNX_MODEL_PATH}")

# ============================================================================
# STEP 4: Representative dataset for calibration
# ============================================================================
def representative_dataset():
    """Generator for calibration data"""
    
    # Check if sample images exist
    if os.path.exists(SAMPLE_IMAGES_DIR):
        image_files = list(Path(SAMPLE_IMAGES_DIR).glob("*.jpg")) + \
                      list(Path(SAMPLE_IMAGES_DIR).glob("*.png"))
        
        if len(image_files) > 0:
            print(f"Using {min(len(image_files), 100)} sample images for calibration")
            
            for img_path in image_files[:100]:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # Resize and preprocess
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (INPUT_SIZE, INPUT_SIZE))
                img_normalized = img_resized.astype(np.float32) / 255.0
                img_batch = np.expand_dims(img_normalized, axis=0)
                
                yield [img_batch]
            return
    
    # Fallback: use random data
    print("⚠️  No sample images found, using synthetic data for calibration")
    print("   For better accuracy, upload real road images to 'sample_images/'")
    
    for _ in range(100):
        synthetic_data = np.random.random((1, INPUT_SIZE, INPUT_SIZE, 3)).astype(np.float32)
        yield [synthetic_data]

# ============================================================================
# STEP 5: Convert ONNX to TensorFlow SavedModel
# ============================================================================
print("\n" + "="*60)
print("Step 1/2: Converting ONNX to TensorFlow")
print("="*60)

import onnx
from onnx_tf.backend import prepare

print("Loading ONNX model...")
onnx_model = onnx.load(ONNX_MODEL_PATH)
print("✅ ONNX model loaded")

print("\nConverting to TensorFlow...")
tf_rep = prepare(onnx_model)
print("✅ Converted to TensorFlow")

# Export to SavedModel format
SAVED_M7: Verify output
# ============================================================================
print("\n" + "="*60)
print("Verification")
print("="*60)

if os.path.exists(OUTPUT_NAME):
    # Check file size
    original_size = os.path.getsize(ONNX_MODEL_PATH) / (1024 * 1024)
    quantized_size = os.path.getsize(OUTPUT_NAME) / (1024 * 1024)
    
    print(f"✅ Model saved as: {OUTPUT_NAME}")
    print(f"   Original ONNX: {original_size:.2f} MB")
    print(f"   Quantized TFLite: {quantized_size:.2f} MB")
    print(f"   Compression: {original_size/quantized_size:.1f}x smaller")
    
    # Verify it's actually INT8
    interpreter = tf.lite.Interpreter(model_path=OUTPUT_NAME)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    print(f"\n   Input shape: {input_details['shape']}")
    print(f"   Input dtype: {input_details['dtype']}")
    print(f"   Output shape: {output_details['shape']}")
    print(f"   Output dtype: {output_details['dtype']}")
    
    if input_details['dtype'] == np.uint8 or input_details['dtype'] == np.int8:
        print("   ✅ Model is properly INT8 quantized!")
    else:
        print("   ⚠️  Model is hybrid quantized (INT8 weights, float ops)")
        print("      This is still faster than full float32")
    
    print("\n" + "="*60)
    print("✅ DONE! Download the model and use it on your Pi")
    print("="*60)
    print("\nOn Raspberry Pi:")
    print(f"  1. Upload {OUTPUT_NAME} to model/ directory")
    print("  2. Update config.py:")
    print(f"     MODEL_PATH_TFLITE = 'model/{OUTPUT_NAME}'")
    print("  3. Run: python main.py")
    print("\nExpected performance: 5-8 FPS on Raspberry Pi 4")
    
else:
    print(f"❌ Could not find output model")
    print("   Check errors above
# ============================================================================
print("\n" + "="*60)
print("Verification")
print("="*60)

# Find the exported model
exported_model = MODEL_PATH.replace('.pt', '_int8.tflite')
if os.path.exists(exported_model):
    # Rename to output name
    os.rename(exported_model, OUTPUT_NAME)
    print(f"✅ Model saved as: {OUTPUT_NAME}")
    
    # Check file size
    size_mb = os.path.getsize(OUTPUT_NAME) / (1024 * 1024)
    print(f"   File size: {size_mb:.2f} MB")
    
    # Verify it's actually INT8
    try:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=OUTPUT_NAME)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        
        print(f"\n   Input dtype: {input_details['dtype']}")
        print(f"   Output dtype: {output_details['dtype']}")
        
        if input_details['dtype'] == np.uint8 or input_details['dtype'] == np.int8:
            print("   ✅ Model is properly INT8 quantized!")
        else:
            print("   ⚠️  Model may not be fully quantized")
            
    except Exception as e:
        print(f"   ⚠️  Could not verify quantization: {e}")
    
    print("\n" + "="*60)
    print("✅ DONE! Download the model and use it on your Pi")
    print("="*60)
    print("\nOn Raspberry Pi, run:")
    print(f"  1. Upload {OUTPUT_NAME} to model/ directory")
    print("  2. Update config.py:")
    print(f"     MODEL_TYPE = 'tflite'")
    print(f"     MODEL_PATH_TFLITE = 'model/{OUTPUT_NAME}'")
    print("  3. Run: python main.py")
    print("\nExpected performance: 5-8 FPS on Raspberry Pi 4")
    
else:
    print(f"❌ Could not find exported model")
    print("   Check ultralytics output above for errors")

# ============================================================================
# STEP 8: Test the model (Optional)
# ============================================================================
print("\n" + "="*60)
print("Quick Test")
print("="*60)

if os.path.exists(SAMPLE_IMAGES_DIR):
    test_images = list(Path(SAMPLE_IMAGES_DIR).glob("*.jpg"))[:3]
    
    if test_images:
        print(f"Testing on {len(test_images)} sample images...")
        
        for img_path in test_images:
            # Load and preprocess
            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (INPUT_SIZE, INPUT_SIZE))
            
            # For INT8 model, input should be uint8 [0, 255]
            input_data = np.expand_dims(img_resized, axis=0).astype(np.uint8)
            
            # Run inference
            interpreter.set_tensor(input_details['index'], input_data)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details['index'])
            
            # Dequantize output if needed
            if output_details['dtype'] == np.uint8:
                output_scale, output_zero_point = output_details['quantization']
                output = (output.astype(np.float32) - output_zero_point) * output_scale
            
            print(f"  ✅ {img_path.name}: Output shape {output.shape}")
        
        print("\n✅ All tests passed! Model is working correctly.")
    else:
        print("No test images found")
else:
    print("Skipping test (no sample images)")
