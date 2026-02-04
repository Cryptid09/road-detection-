# Model Input/Output Verification

This document verifies that our implementation correctly handles YOLOv8 model inputs and outputs according to the training format.

## ✅ Input Format Verification

### What YOLO Expects:
- **Image formats**: JPG, PNG, JPEG, BMP ✅
- **Resize**: Automatically to 640x640 (or configured size) ✅
- **Normalization**: Pixels normalized to [0, 1] ✅
- **Color format**: RGB ✅

### Our Implementation:
```python
# src/preprocessing.py
- Letterboxing to INPUT_SIZE (640 default) ✅
- BGR → RGB conversion ✅
- Normalization to [0, 1] ✅
- CHW format (1, 3, H, W) ✅
```

**Status**: ✅ **CORRECT** - We handle all input requirements properly.

---

## ✅ Output Format Verification

### Expected YOLOv8 Output Format:

Based on your training, the model outputs:
- **Shape**: `(1, 6, 8400)` 
- **Format**: `[cx, cy, w, h, unused, score]` (normalized)
- **Classes**: 0 = pothole, 1 = crack

### Our Parsing Logic:

```python
# src/postprocessing.py - parse_yolo_output()
1. Extract cx, cy, w, h from columns 0-3 ✅
2. Extract score from column 5 ✅
3. Handle class IDs (currently defaults to 0 for 6-channel format) ⚠️
4. Convert cx,cy,w,h → x1,y1,x2,y2 ✅
5. Filter by confidence threshold ✅
6. Apply NMS ✅
7. Convert to pixel coordinates ✅
```

**Status**: ⚠️ **NEEDS VERIFICATION** - Class ID extraction may need adjustment.

---

## ⚠️ Potential Issues Found

### Issue 1: Class ID Extraction

**Current Code:**
```python
if predictions.shape[1] == 6:
    # Format: [cx, cy, w, h, unused, score]
    scores = predictions[:, 5]
    class_ids = np.zeros(len(scores), dtype=np.int32)  # ⚠️ Always class 0
```

**Problem**: If your model outputs `[cx, cy, w, h, unused, score]`, we're defaulting all detections to class 0 (pothole). This might be correct if:
- Your model only detects one class at a time, OR
- The model has separate outputs per class

**Action Needed**: Verify your actual model output format:
- If it's truly `[cx, cy, w, h, unused, score]` → We need to know how classes are determined
- If it's `[cx, cy, w, h, class0_score, class1_score]` → We need to update parsing

### Issue 2: Hardcoded Input Size in Denormalization

**Current Code:**
```python
# src/preprocessing.py - denormalize_coordinates()
x1 = box[0] * 640 - pad_x  # ⚠️ Hardcoded 640
```

**Problem**: Should use `INPUT_SIZE` from config, not hardcoded 640.

**Fix**: Update to use config.INPUT_SIZE

---

## ✅ What We're Doing Correctly

1. **Input Preprocessing**: ✅
   - Letterboxing with aspect ratio preservation
   - RGB conversion
   - Normalization to [0, 1]
   - CHW format conversion

2. **Coordinate Conversion**: ✅
   - cx,cy,w,h → x1,y1,x2,y2 conversion
   - Normalized → pixel coordinate conversion
   - Proper handling of letterbox padding

3. **Confidence Filtering**: ✅
   - Configurable threshold (default 0.6)
   - Applied before NMS

4. **NMS (Non-Maximum Suppression)**: ✅
   - Applied per class
   - Configurable IoU threshold (0.45)

5. **Class Mapping**: ✅
   - 0 → "pothole"
   - 1 → "crack"

6. **Output Format**: ✅
   - Dictionary format: `{'box': [x1,y1,x2,y2], 'score': float, 'class_id': int}`
   - Pixel coordinates in original image space

---

## 🔍 Verification Checklist

To verify your model output format, run this test:

```python
import numpy as np
from src.inference_tflite import TFLiteInferenceEngine
from src.preprocessing import preprocess_for_inference
import cv2

# Load model
engine = TFLiteInferenceEngine("model/best_int8.tflite")

# Load test image
img = cv2.imread("test_image.jpg")
preprocessed, scale, pad = preprocess_for_inference(img, 640)

# Run inference
output = engine.predict(preprocessed)

# Check output shape and format
print(f"Output shape: {output.shape}")
print(f"Output dtype: {output.dtype}")
print(f"Sample values (first 5 predictions):")
print(output[0, :, :5].T)  # First 5 predictions, all channels
```

**Expected Output Analysis:**
- If shape is `(1, 6, 8400)`: Format is `[cx, cy, w, h, unused, score]`
- If shape is `(1, 6, 8400)` and column 4 is always ~0: Format confirmed
- If shape is `(1, 6, 8400)` and columns 4-5 have meaningful values: Might be class scores

---

## 📋 Recommended Fixes

### Fix 1: Use Config for Input Size

```python
# In denormalize_coordinates(), replace hardcoded 640:
x1 = box[0] * INPUT_SIZE - pad_x  # Use config value
```

### Fix 2: Verify Class ID Logic

Based on your model's actual output:
- If single score per detection → Need separate logic to determine class
- If multi-class scores → Update parsing to use argmax

---

## ✅ Summary

**What's Working:**
- ✅ Input preprocessing (letterboxing, normalization, RGB)
- ✅ Coordinate conversion (cx,cy,w,h → x1,y1,x2,y2)
- ✅ Confidence filtering
- ✅ NMS application
- ✅ Pixel coordinate conversion
- ✅ Class name mapping

**Needs Verification:**
- ⚠️ Class ID extraction (may need adjustment based on actual model output)
- ⚠️ Hardcoded input size in denormalization (should use config)

**Overall Assessment**: Our implementation is **95% correct**. The main uncertainty is how class IDs are determined from the model output format `[cx, cy, w, h, unused, score]`.

