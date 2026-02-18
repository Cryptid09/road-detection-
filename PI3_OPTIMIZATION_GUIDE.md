# Raspberry Pi 3 Optimization Guide

## ✅ What Has Been Implemented

All Pi 3 optimizations from your training guide have been implemented in the codebase:

### 1. **Configuration Optimizations** (`config.py`)

- ✅ **Confidence Threshold**: Set to `0.7` (reduces false positives and CPU load)
- ✅ **Frame Skipping**: Set to `2` (process every 2nd frame - MANDATORY for Pi 3)
- ✅ **Display**: Disabled by default (`SHOW_DISPLAY = False`)
- ✅ **Image Logging**: Disabled by default (`LOG_INCLUDE_IMAGE = False`) to reduce I/O overhead
- ✅ **TFLite Threads**: Set to `2` (optimized for Pi 3)
- ✅ **Camera Resolution**: Already set to `640x480` (optimal for Pi 3)

### 2. **Auto-Detection of Model Input Size**

The system now automatically detects the model's input size and adjusts `config.INPUT_SIZE` if there's a mismatch. This prevents the "Dimension mismatch" error you encountered.

**How it works:**
- When the TFLite model loads, it reads the model's fixed input shape
- If `config.INPUT_SIZE` doesn't match, it auto-adjusts and prints a warning
- This means you can set `INPUT_SIZE = 320` in config, and if your model is still 640x640, it will auto-adjust to 640

### 3. **TFLite Thread Configuration**

The TFLite interpreter now uses the configured number of threads (`TFLITE_NUM_THREADS = 2` for Pi 3).

---

## ⚠️ CRITICAL: You Still Need to Export a 320×320 Model

**The current `best_int8.tflite` is likely still 640×640**, which is why you're getting low FPS.

### To Get the FPS Boost, You MUST:

1. **Export a new 320×320 INT8 TFLite model** using the Kaggle notebook workflow you shared:

```python
from ultralytics import YOLO

model = YOLO("/path/to/best.pt")

# Export for Pi 3
model.export(
    format="tflite",
    imgsz=320,        # FIXED 320×320 for Pi 3
    int8=True,        # INT8 quantization
    optimize=True     # Extra optimizations
)
```

2. **Download the exported model** (`best_int8.tflite` from `runs/detect/*/weights/best_saved_model/`)

3. **Replace your current model** on the Pi:
   ```bash
   # On Pi
   cp new_best_320_int8.tflite model/best_int8.tflite
   ```

4. **Update config** (optional - auto-detection will handle it):
   ```python
   INPUT_SIZE = 320  # Already set in config.py
   ```

---

## 🚀 Expected Performance After Model Export

| Configuration | Current (640×640) | After 320×320 Export |
|--------------|-------------------|----------------------|
| **FPS** | 0.3-0.5 | **3-6 FPS** ✅ |
| **Model Size** | ~6-7 MB | ~2-3 MB |
| **RAM Usage** | High | Low |
| **CPU Load** | Very High | Moderate |

---

## 📋 Current Optimized Settings (config.py)

```python
# Pi 3 Optimized Settings
CONFIDENCE_THRESHOLD = 0.7      # Higher threshold
INPUT_SIZE = 320                # Will auto-adjust if model mismatch
FRAME_SKIP = 2                   # Process every 2nd frame
SHOW_DISPLAY = False             # Headless mode
LOG_INCLUDE_IMAGE = False        # No image saving
TFLITE_NUM_THREADS = 2           # Pi 3 optimized
CAMERA_WIDTH = 640               # Optimal resolution
CAMERA_HEIGHT = 480
```

---

## 🧪 Testing on Pi 3

### 1. Test with Current 640×640 Model (Baseline)

```bash
python main.py --no-display --profile
```

**Expected:**
- FPS: ~0.3-0.5
- System will auto-detect 640×640 and adjust
- You'll see timing breakdown showing inference is the bottleneck

### 2. After Exporting 320×320 Model

```bash
# Replace model
cp /path/to/new/best_320_int8.tflite model/best_int8.tflite

# Run again
python main.py --no-display --profile
```

**Expected:**
- FPS: **3-6 FPS** ✅
- System will auto-detect 320×320
- Inference time should drop significantly

---

## 🔍 Verification Steps

### Check Model Input Size

After loading, you should see in the logs:

```
TFLite Model loaded: model/best_int8.tflite
Input shape: [1, 320, 320, 3]  # Should be 320, not 640
Quantized: True
Using input size: 320x320
```

### Check Performance

With `--profile` flag, you'll see timing breakdown every 30 frames:

```
FPS: 4.2 | Frames processed: 30
  Timing breakdown (avg over 30 frames):
    Preprocess: 15.2ms
    Inference:  180.5ms (85.2%)  # This should drop significantly with 320×320
    Postprocess: 12.3ms
    Coords:      2.1ms
    Total:       210.1ms
```

---

## 🎯 Realistic Expectations for Pi 3

- **FPS**: 3-6 FPS is **acceptable** for dashcam anomaly detection
- **Pothole detection**: Should work well
- **Crack detection**: May be unreliable (as mentioned in your guide)
- **Model size**: ~2-3 MB (much smaller than 640×640)

---

## 🔧 Troubleshooting

### If FPS is still low after 320×320 export:

1. **Check model is actually 320×320**:
   - Look for "Input shape: [1, 320, 320, 3]" in logs
   - If it shows 640, the export didn't work correctly

2. **Verify INT8 quantization**:
   - Should see "Quantized: True"
   - If False, the model isn't quantized

3. **Check thermal throttling**:
   ```bash
   vcgencmd measure_temp
   vcgencmd get_throttled
   ```
   - Temperature should be < 80°C
   - If throttling, add cooling

4. **Increase frame skipping**:
   ```python
   FRAME_SKIP = 3  # Process every 3rd frame
   ```

---

## 📝 Next Steps

1. ✅ **Code is ready** - All Pi 3 optimizations implemented
2. ⏳ **Export 320×320 model** - Use Kaggle notebook workflow
3. ⏳ **Test on Pi 3** - Run with `--profile` to verify FPS improvement
4. ⏳ **Fine-tune if needed** - Adjust `FRAME_SKIP` or `CONFIDENCE_THRESHOLD` based on results

---

## 💡 Optional: Pothole-Only Model

As mentioned in your guide, for Pi 3 you might want to:
- Drop crack detection
- Train/export a **pothole-only** model
- This will be faster and more reliable

This is a training decision, not a code change needed.

