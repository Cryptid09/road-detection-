# Project Summary

## What This Project Does

Real-time road anomaly detection system that:
- Captures video from USB webcam or Raspberry Pi Camera
- Runs YOLOv8 inference to detect potholes and cracks
- Logs all detections to CSV with timestamps and coordinates
- Displays real-time video with bounding boxes (optional)
- Works fully offline on Raspberry Pi

## Key Files

### Core Application
- `main.py` - Main inference script
- `config.py` - All configuration parameters
- `detect_camera.py` - Camera detection utility

### Source Modules (`src/`)
- `camera.py` - Unified camera interface (USB + Pi Camera)
- `preprocessing.py` - Image preprocessing and letterboxing
- `inference_onnx.py` - ONNX model inference
- `inference_tflite.py` - TFLite model inference
- `postprocessing.py` - NMS and coordinate conversion
- `fps_counter.py` - Performance monitoring
- `event_logger.py` - CSV logging
- `visualization.py` - Drawing utilities

### Documentation
- `README.md` - Main documentation and quick start
- `DEPLOYMENT_PI.md` - Detailed Pi deployment guide
- `requirements.txt` - Dependencies (platform-agnostic)
- `requirements-pi.txt` - Dependencies for Pi only

## Quick Commands

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-pi.txt  # On Pi

# Run
python detect_camera.py  # Find cameras
python main.py            # Run inference
python main.py --no-display  # Headless mode
```

## Deployment Checklist

- [ ] Transfer project to Raspberry Pi
- [ ] Install system dependencies
- [ ] Create virtual environment
- [ ] Install Python packages
- [ ] Add model files to `model/` directory
- [ ] Run `detect_camera.py` to find cameras
- [ ] Configure `config.py`
- [ ] Test with `python main.py`
- [ ] (Optional) Set up systemd service for auto-start

## Performance Targets

- **Minimum FPS**: 5 FPS (achieved with INPUT_SIZE=320 on Pi 4)
- **Target FPS**: 10+ FPS (achieved with INPUT_SIZE=320 on Pi 5)
- **Memory Usage**: <2GB RAM
- **CPU Usage**: 70-90% on single core (normal for CPU inference)

## Model Requirements

- **Format**: ONNX or TFLite INT8
- **Input**: 640x640 RGB (can resize to 320/416 for speed)
- **Output**: (1, 6, 8400) format: [cx, cy, w, h, unused, score]
- **Classes**: 0=pothole, 1=crack

## Output Format

CSV file (`logs/detections.csv`) with columns:
- timestamp (ISO format)
- class_name (pothole/crack)
- class_id (0/1)
- confidence (0.0-1.0)
- x1, y1, x2, y2 (bounding box pixels)

## Next Steps

1. Deploy to Raspberry Pi following `DEPLOYMENT_PI.md`
2. Configure for your specific camera and environment
3. Monitor performance and adjust `INPUT_SIZE` as needed
4. Set up auto-start service for production use
5. (Optional) Add GPS coordinates for mobile deployment
6. (Optional) Set up log rotation and archival

