# Raspberry Pi Deployment Guide

Complete step-by-step guide for deploying the Road Anomaly Detection System on Raspberry Pi.

## Prerequisites

- **Hardware**: Raspberry Pi 4 (4GB+ RAM) or Raspberry Pi 5
- **OS**: Raspberry Pi OS (64-bit recommended) or Ubuntu for Raspberry Pi
- **Python**: 3.11.5 (comes with Raspberry Pi OS)
- **Camera**: USB webcam or Raspberry Pi Camera Module (ribbon)
- **Storage**: At least 8GB free space (for models and dependencies)

## Step 1: Initial Pi Setup

### 1.1 Update System

```bash
sudo apt update
sudo apt upgrade -y
```

### 1.2 Install System Dependencies

```bash
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    libopencv-dev \
    python3-opencv \
    v4l-utils \
    libcamera-apps
```

### 1.3 Enable Camera (if using Pi Camera)

```bash
sudo raspi-config
# Navigate to: Interface Options > Camera > Enable
# Reboot after enabling
sudo reboot
```

## Step 2: Transfer Project to Pi

### Option A: Using Git (Recommended)

```bash
# On Pi
cd ~
git clone <your-repo-url> road_management
cd road_management
```

### Option B: Using SCP (from your laptop)

```bash
# On your laptop
scp -r /path/to/road_management pi@<pi-ip>:/home/pi/
```

### Option C: Using USB Drive

1. Copy project folder to USB drive
2. Insert USB drive into Pi
3. Copy to home directory: `cp -r /media/usb/road_management ~/`

## Step 3: Set Up Python Environment

```bash
cd ~/road_management

# Create virtual environment WITH system site packages (for OpenCV)
python3 -m venv --system-site-packages venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

## Step 4: Install Python Dependencies

```bash
# Make sure venv is activated (you should see (venv) in prompt)
source venv/bin/activate

# Install from Pi-specific requirements
pip install -r requirements-pi.txt

# Verify installation
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import tflite_runtime; print('TFLite: OK')"
python -c "import numpy; print('NumPy:', numpy.__version__)"
```

**Note**: This installation uses TFLite for inference (optimized for Pi). ONNX Runtime is NOT used as it doesn't support 32-bit ARM (armv7l).

**Note**: If `tflite-runtime` installation fails, you can use TensorFlow instead:
```bash
pip install tensorflow>=2.14.0  # Larger but works everywhere
```

## Step 5: Verify Model Files

Ensure your model files are in the `model/` directory:

```bash
ls -lh model/
# Should show:
# - best_int8.tflite (or best.onnx)
```

If models are missing, copy them from your training environment.

## Step 6: Configure System

### 6.1 Detect Cameras

```bash
source venv/bin/activate
python detect_camera.py
```

This will show available cameras. Note the camera index or Pi Camera status.

### 6.2 Edit Configuration

Edit `config.py`:

```python
# For USB webcam
CAMERA_INDEX = 0  # Use index from detect_camera.py
USE_PI_CAMERA = False

# For Pi Camera (ribbon)
USE_PI_CAMERA = True
CAMERA_INDEX = 0  # Ignored when USE_PI_CAMERA = True

# Performance settings for Pi
INPUT_SIZE = 320  # Use 320 or 416 for faster inference (640 is slower)
CONFIDENCE_THRESHOLD = 0.6
SHOW_DISPLAY = False  # Set to False for headless operation
```

### 6.3 Test Camera Access

**For USB webcam:**
```bash
# List video devices
v4l2-ctl --list-devices

# Test camera
libcamera-vid -t 5000 -o test.h264
```

**For Pi Camera:**
```bash
# Test camera
libcamera-hello -t 5000
```

## Step 7: Test the System

```bash
source venv/bin/activate

# Test with display (if connected to monitor)
python main.py

# Or test in headless mode
python main.py --no-display
```

## Step 8: Set Up Auto-Start (Optional)

### 8.1 Create Systemd Service

Create service file:

```bash
sudo nano /etc/systemd/system/road-detection.service
```

Add this content:

```ini
[Unit]
Description=Road Anomaly Detection System
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/road_management
Environment="PATH=/home/pi/road_management/venv/bin"
ExecStart=/home/pi/road_management/venv/bin/python /home/pi/road_management/main.py --no-display
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### 8.2 Enable and Start Service

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service (starts on boot)
sudo systemctl enable road-detection.service

# Start service
sudo systemctl start road-detection.service

# Check status
sudo systemctl status road-detection.service

# View logs
sudo journalctl -u road-detection.service -f
```

### 8.3 Stop/Disable Service

```bash
# Stop service
sudo systemctl stop road-detection.service

# Disable auto-start
sudo systemctl disable road-detection.service
```

## Step 9: Performance Optimization

### 9.1 Overclock Pi (Optional, Advanced)

Edit `/boot/config.txt`:

```bash
sudo nano /boot/config.txt
```

Add (for Pi 4):
```
over_voltage=2
arm_freq=2000
```

**Warning**: Overclocking can void warranty and cause instability. Use at your own risk.

### 9.2 Increase GPU Memory Split (if needed)

```bash
sudo nano /boot/config.txt
```

Set:
```
gpu_mem=128
```

### 9.3 Disable Unnecessary Services

```bash
# Disable Bluetooth (if not needed)
sudo systemctl disable bluetooth

# Disable WiFi power management
sudo iwconfig wlan0 power off
```

## Step 10: Monitoring and Logs

### View Detection Logs

```bash
# View recent detections
tail -f logs/detections.csv

# Count total detections
wc -l logs/detections.csv
```

### Monitor System Resources

```bash
# CPU and memory usage
htop

# Disk usage
df -h

# Temperature (important for Pi)
vcgencmd measure_temp
```

## Troubleshooting

### Camera Not Detected

1. **USB Webcam:**
   ```bash
   # Check permissions
   sudo usermod -a -G video $USER
   # Logout and login again
   
   # List devices
   ls -l /dev/video*
   v4l2-ctl --list-devices
   ```

2. **Pi Camera:**
   ```bash
   # Verify camera is enabled
   sudo raspi-config
   # Interface Options > Camera > Enable
   
   # Test camera
   libcamera-hello
   ```

### Low FPS

1. Reduce `INPUT_SIZE` to 320 or 416 in `config.py`
2. Use TFLite INT8 model instead of ONNX
3. Run in headless mode (`--no-display`)
4. Close other applications
5. Check CPU temperature: `vcgencmd measure_temp`

### Out of Memory

1. Reduce `INPUT_SIZE` in config
2. Use TFLite INT8 model (smaller memory footprint)
3. Increase swap space:
   ```bash
   sudo dphys-swapfile swapoff
   sudo nano /etc/dphys-swapfile
   # Set CONF_SWAPSIZE=2048
   sudo dphys-swapfile setup
   sudo dphys-swapfile swapon
   ```

### Service Won't Start

1. Check logs: `sudo journalctl -u road-detection.service -n 50`
2. Verify paths in service file are correct
3. Test manually: `source venv/bin/activate && python main.py --no-display`

## Performance Expectations

On Raspberry Pi 4 (4GB):
- **Input size 640**: ~2-3 FPS
- **Input size 416**: ~4-5 FPS
- **Input size 320**: ~6-8 FPS

On Raspberry Pi 5:
- **Input size 640**: ~5-7 FPS
- **Input size 416**: ~8-10 FPS
- **Input size 320**: ~12-15 FPS

## Next Steps

- Set up remote monitoring (SSH, VNC)
- Configure automatic log rotation
- Set up alerts for high detection counts
- Integrate with cloud storage for logs
- Add GPS coordinates to detections (if using mobile Pi setup)

## Support

For issues or questions:
1. Check logs: `logs/detections.csv` and system logs
2. Review configuration in `config.py`
3. Test components individually: `python detect_camera.py`
4. Check system resources: `htop`, `df -h`, `vcgencmd measure_temp`

