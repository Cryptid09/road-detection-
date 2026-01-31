# Road Anomaly Detection System

Real-time edge AI system for detecting road anomalies (potholes and cracks) using Raspberry Pi.

## Features

- Real-time video processing from USB webcam or Pi Camera
- YOLOv8 Nano model inference (ONNX or TFLite INT8)
- CPU-only execution, fully offline
- Bounding box visualization with confidence scores
- CSV logging of detected anomalies
- FPS monitoring and display
- Modular, production-ready code

## Requirements

- Raspberry Pi 4 or 5
- Python 3.7+
- USB webcam or Pi Camera

## Installation

1. Clone or copy the project to your Raspberry Pi:

```bash
cd /home/pi
git clone <repository-url> road_detection
cd road_detection
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

For Raspberry Pi, you may need to install OpenCV separately:

```bash
sudo apt-get update
sudo apt-get install python3-opencv
```

## Configuration

Edit `src/config.py` to adjust parameters:

- `MODEL_TYPE`: 'onnx' or 'tflite'
- `CONFIDENCE_THRESHOLD`: Detection confidence threshold (default: 0.6)
- `CAMERA_INDEX`: Camera device index (default: 0)
- Model paths in `MODEL_PATHS`

## Usage

1. Ensure models are in the `models/` directory:
   - `best.onnx` for ONNX inference
   - `best_int8.tflite` for TFLite inference

2. **Test the models (recommended first):**

   **Full pipeline test with camera:**
   ```bash
   python test_models.py
   ```
   This tests both ONNX and TFLite models using your laptop camera.

   **Inference-only test (no camera needed):**
   ```bash
   python test_inference.py
   ```
   This tests model loading and inference with dummy data.

   **Complete system validation:**
   ```bash
   python system_validation.py
   ```
   Validates all system components step by step.

   **Performance analysis:**
   ```bash
   python performance_analysis.py
   ```
   Provides detailed performance metrics and timing analysis.

   **Output format analysis:**
   ```bash
   python analyze_output.py
   ```
   Examines the TFLite model output structure.

3. **Run the full system:**

   ```bash
   python main.py
   ```

4. The system will:
   - Open camera feed
   - Process frames in real-time
   - Display annotated video with detections
   - Log anomalies to `logs/anomalies.csv`

4. Press 'q' to quit

## Project Structure

```
road_detection/
├── main.py                 # Main application script
├── requirements.txt        # Python dependencies
├── src/
│   ├── camera.py          # Camera capture module
│   ├── preprocessing.py   # Image preprocessing
│   ├── inference.py       # Inference engine (ONNX/TFLite)
│   ├── postprocessing.py  # Detection postprocessing
│   ├── logging.py         # CSV logging module
│   ├── fps_counter.py     # FPS measurement
│   └── config.py          # Configuration parameters
├── models/                # Model files
├── logs/                  # Log files
└── scripts/               # Utility scripts
```

## Performance

- Target FPS: ≥5
- Optimized for CPU-only execution
- Low memory footprint

## Troubleshooting

- If camera fails to open, check `CAMERA_INDEX` in config
- For TFLite issues, ensure `tflite-runtime` is installed
- Check logs for error messages

## License

[Add your license here]