# Road Anomaly Detection System

Real-time edge AI system for detecting road anomalies (potholes and cracks) using YOLOv8 on Raspberry Pi. Designed for CPU-only inference with support for both USB webcams and Raspberry Pi Camera.

## 🚀 Features

- **Real-time inference** on CPU-only (no GPU required)
- **Dual camera support**: USB webcam or Raspberry Pi Camera (ribbon)
- **Multiple model formats**: ONNX and TensorFlow Lite INT8 quantized
- **CSV logging** of all detections with timestamps and coordinates
- **FPS monitoring** and performance tracking
- **Modular architecture** for easy customization and maintenance
- **Production-ready** code with comprehensive error handling

## 📋 System Requirements

### Hardware
- **Raspberry Pi 4** (4GB+ RAM recommended) or **Raspberry Pi 5**
- **Camera**: USB webcam or Raspberry Pi Camera Module (ribbon)
- **Storage**: At least 8GB free space

### Software
- **OS**: Raspberry Pi OS (64-bit recommended) or Ubuntu for Raspberry Pi
- **Python**: 3.11.5 (comes with Raspberry Pi OS)
- **RAM**: ~2GB available for inference

## 📦 Quick Start

### 1. Clone/Download Project

```bash
cd ~
git clone <your-repo-url> road_management
cd road_management
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 3. Install Dependencies

**On Raspberry Pi:**
```bash
pip install opencv-python>=4.8.0 onnxruntime>=1.16.0 numpy>=1.24.0
pip install tflite-runtime>=2.14.0
pip install picamera2  # Optional, for Pi Camera
```

**On Laptop/Desktop (Development):**
```bash
pip install opencv-python>=4.8.0 onnxruntime>=1.16.0 numpy>=1.24.0
pip install tensorflow>=2.14.0  # Includes tensorflow.lite
```

### 4. Add Model Files

Place your trained models in the `model/` directory:
- `model/best.onnx` - ONNX model
- `model/best_int8.tflite` - TensorFlow Lite INT8 quantized model (recommended for Pi)

### 5. Detect Cameras

```bash
source venv/bin/activate
python detect_camera.py
```

### 6. Configure System

Edit `config.py` based on your setup:

```python
# Camera configuration
CAMERA_INDEX = 0  # USB webcam index
USE_PI_CAMERA = False  # Set True for Pi Camera (ribbon)

# Performance settings
INPUT_SIZE = 320  # Use 320/416 for Pi, 640 for development
CONFIDENCE_THRESHOLD = 0.6
SHOW_DISPLAY = False  # Set False for headless operation on Pi
```

### 7. Run the System

```bash
source venv/bin/activate
python main.py
```

**Command line options:**
```bash
python main.py --camera 1        # Use specific camera
python main.py --no-display      # Headless mode
python main.py --model-type onnx # Use ONNX model
```

## 📁 Project Structure

```
road_management/
├── README.md                 # This file
├── DEPLOYMENT_PI.md          # Detailed Pi deployment guide
├── config.py                 # Configuration file
├── main.py                   # Main inference script
├── detect_camera.py          # Camera detection utility
├── run.sh                    # Quick run script
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
│
├── model/                    # Model files (add your models here)
│   ├── best.onnx
│   └── best_int8.tflite
│
├── logs/                     # Detection logs (auto-created)
│   └── detections.csv
│
└── src/                      # Source modules
    ├── __init__.py
    ├── camera.py             # Camera interface (USB + Pi Camera)
    ├── preprocessing.py      # Image preprocessing & letterboxing
    ├── inference_onnx.py     # ONNX inference engine
    ├── inference_tflite.py   # TFLite inference engine
    ├── postprocessing.py     # NMS & coordinate conversion
    ├── fps_counter.py        # FPS measurement
    ├── event_logger.py       # CSV logging
    └── visualization.py      # Drawing utilities
```

## ⚙️ Configuration

All settings are in `config.py`. Key parameters:

| Parameter | Description | Default | Recommended (Pi) |
|-----------|-------------|---------|-------------------|
| `MODEL_TYPE` | Model format: "onnx" or "tflite" | "tflite" | "tflite" |
| `INPUT_SIZE` | Model input size | 640 | 320 or 416 |
| `CONFIDENCE_THRESHOLD` | Detection confidence | 0.6 | 0.6 |
| `CAMERA_INDEX` | USB webcam index | 0 | 0 |
| `USE_PI_CAMERA` | Use Pi Camera (ribbon) | False | True/False |
| `SHOW_DISPLAY` | Show display window | True | False |

## 📊 Output

### Detection Logs

All detections are automatically logged to `logs/detections.csv`:

| Column | Description |
|--------|-------------|
| timestamp | ISO format timestamp |
| class_name | "pothole" or "crack" |
| class_id | 0 (pothole) or 1 (crack) |
| confidence | Detection confidence (0.0-1.0) |
| x1, y1, x2, y2 | Bounding box coordinates (pixels) |

### Console Output

- Real-time FPS updates every 30 frames
- Detection counts per frame
- System status messages

## 🎯 Performance

### Expected FPS on Raspberry Pi

| Pi Model | Input 640 | Input 416 | Input 320 |
|----------|-----------|----------|-----------|
| Pi 4 (4GB) | 2-3 FPS | 4-5 FPS | 6-8 FPS |
| Pi 5 | 5-7 FPS | 8-10 FPS | 12-15 FPS |

### Optimization Tips

1. **Reduce input size**: Use 320 or 416 instead of 640
2. **Use TFLite INT8**: Faster and lower memory than ONNX
3. **Headless mode**: Disable display for better performance
4. **Close other apps**: Free up CPU resources
5. **Monitor temperature**: Use `vcgencmd measure_temp`

## 🔧 Troubleshooting

### Camera Issues

**Problem**: Camera not detected
```bash
# Run camera detection
python detect_camera.py

# For USB webcam: Check permissions
sudo usermod -a -G video $USER
# Logout and login again

# For Pi Camera: Enable in raspi-config
sudo raspi-config
# Interface Options > Camera > Enable
```

**Problem**: "Failed to open camera"
- Try different camera indices: `python main.py --camera 1`
- Check if camera is in use: `lsof /dev/video0`
- Verify camera works: `libcamera-hello` (Pi Camera) or `v4l2-ctl --list-devices` (USB)

### Model Issues

**Problem**: Model file not found
- Verify model files exist in `model/` directory
- Check file paths in `config.py`
- System will try to fall back to alternative format if available

**Problem**: Low FPS
- Reduce `INPUT_SIZE` to 320 or 416
- Use TFLite INT8 model instead of ONNX
- Run in headless mode: `--no-display`
- Check CPU temperature and throttle status

### Installation Issues

**Problem**: "tflite-runtime not available"
- **On Pi**: `pip install tflite-runtime>=2.14.0`
- **On laptop**: `pip install tensorflow>=2.14.0` (includes tflite)

**Problem**: "externally-managed-environment"
- Use virtual environment: `python3 -m venv venv && source venv/bin/activate`

## 🚀 Deployment

For detailed Raspberry Pi deployment instructions, see **[DEPLOYMENT_PI.md](DEPLOYMENT_PI.md)**.

Quick deployment steps:
1. Transfer project to Pi
2. Set up virtual environment
3. Install dependencies
4. Configure camera and settings
5. Test system
6. (Optional) Set up systemd service for auto-start

## 📝 Development

### Running Tests

```bash
source venv/bin/activate
python detect_camera.py  # Test camera detection
python main.py --help    # See all options
```

### Code Structure

- **Modular design**: Each component in separate module
- **Easy to extend**: Add new inference engines or postprocessing
- **Well-documented**: Inline comments explain each step
- **Error handling**: Graceful failures with helpful messages

## 🔐 Security Notes

- System runs fully offline (no cloud dependencies)
- All data stored locally in `logs/` directory
- No network access required
- Camera data processed in real-time, not stored

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

- YOLOv8 model architecture
- OpenCV for computer vision utilities
- ONNX Runtime and TensorFlow Lite for inference engines
- Raspberry Pi Foundation for hardware platform

## 📞 Support

For issues:
1. Check logs: `logs/detections.csv` and console output
2. Review configuration in `config.py`
3. Test components: `python detect_camera.py`
4. Check system resources: `htop`, `df -h`, `vcgencmd measure_temp`

---

**Ready for deployment?** See [DEPLOYMENT_PI.md](DEPLOYMENT_PI.md) for complete Pi setup guide.
