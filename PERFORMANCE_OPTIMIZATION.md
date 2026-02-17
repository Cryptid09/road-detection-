# Performance Optimization Guide

## Current Issue: 0.36 FPS (Target: ≥5 FPS)

The camera is initialized at 15 FPS, but actual processing is only 0.36 FPS. This indicates the **bottleneck is in the processing pipeline, not the camera**.

## Step 1: Identify the Bottleneck

Run with profiling to see where time is spent:

```bash
python main.py --profile
```

This will show timing breakdown every 30 frames:
- **Preprocess**: Letterboxing, normalization
- **Inference**: Model inference (likely the main bottleneck)
- **Postprocess**: NMS, parsing
- **Coords**: Coordinate conversion

## Step 2: Quick Optimizations (Try These First)

### Option A: Reduce Input Size (BIGGEST IMPACT)

**Current**: `INPUT_SIZE = 640` (very slow on Pi)
**Recommended**: `INPUT_SIZE = 320` or `416`

Edit `config.py`:
```python
INPUT_SIZE = 320  # Reduced from 640 for better FPS
```

**Expected FPS improvement**: 4-8x faster
- 640x640: ~0.4 FPS
- 416x416: ~1.5-2 FPS  
- 320x320: ~3-5 FPS

**Note**: You may need to retrain/requantize the model for 320x320, OR the model might accept different input sizes if it was trained with dynamic shapes.

### Option B: Disable Display (Medium Impact)

Run headless mode:
```bash
python main.py --no-display
```

Or edit `config.py`:
```python
SHOW_DISPLAY = False
```

**Expected FPS improvement**: 10-20% faster

### Option C: Disable Image Saving (Medium Impact)

Edit `config.py`:
```python
LOG_INCLUDE_IMAGE = False  # Disable image saving for better FPS
```

**Expected FPS improvement**: 5-15% faster (especially on slow SD cards)

### Option D: All Optimizations Combined

```python
# config.py
INPUT_SIZE = 320  # Reduced from 640
SHOW_DISPLAY = False  # Disable display
LOG_INCLUDE_IMAGE = False  # Disable image saving
```

Then run:
```bash
python main.py --no-display --profile
```

## Step 3: Verify Model Quantization

Check if your model is actually INT8 quantized:

```python
# Check model info at startup
# Should see: "Quantized: True" and "Input dtype: uint8"
```

If it shows `Quantized: False` or `Input dtype: float32`, the model is NOT quantized and will be much slower.

## Step 4: Check System Resources

On Raspberry Pi, check:
```bash
# CPU temperature (should be < 80°C)
vcgencmd measure_temp

# Check for throttling
vcgencmd get_throttled

# CPU usage
top
```

If throttling is active, add cooling.

## Step 5: Model-Specific Optimizations

### For TFLite:
- Ensure using `tflite-runtime` (lighter than full TensorFlow)
- Verify INT8 quantization is working
- Consider using EdgeTPU if available (but project requires CPU-only)

### For ONNX:
- Already using CPU provider (correct for edge deployment)
- Graph optimizations are enabled

## Expected Performance After Optimizations

| Configuration | Expected FPS (Pi 4) | Expected FPS (Pi 5) |
|--------------|---------------------|---------------------|
| 640x640, display, images | 0.3-0.5 | 0.5-0.8 |
| 640x640, no display | 0.4-0.6 | 0.7-1.0 |
| 416x416, no display | 1.5-2.5 | 3-4 |
| **320x320, no display** | **3-5** ✅ | **6-10** ✅ |

## Troubleshooting

### If FPS is still low after reducing input size:

1. **Check model input size**: Some models are hardcoded to 640x640
   - Solution: Retrain/requantize model for 320x320 or 416x416

2. **Check if model is actually INT8**: 
   - Look for "Quantized: True" in startup logs
   - If False, requantize the model

3. **Check CPU usage**:
   - Should be near 100% on one core (single-threaded inference)
   - If low, something else is blocking

4. **Check SD card speed**:
   - Slow SD cards can bottleneck image saving
   - Use `LOG_INCLUDE_IMAGE = False` to test

5. **Check thermal throttling**:
   - High temperature causes CPU slowdown
   - Add heatsink/fan if needed

## Recommended Configuration for ≥5 FPS

```python
# config.py
INPUT_SIZE = 320  # or 416 if model doesn't support 320
SHOW_DISPLAY = False  # Disable for headless operation
LOG_INCLUDE_IMAGE = False  # Or set to True only when needed
CAMERA_FPS = 15  # Keep at 15 (camera can capture faster than we process)
```

Run with:
```bash
python main.py --no-display --profile
```

This should achieve **≥5 FPS** on Pi 4 and **≥8 FPS** on Pi 5.

