#!/usr/bin/env python3
"""
Camera Detection Utility
Helps identify available cameras and their properties
"""

import cv2
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.camera import list_available_cameras

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False


def test_usb_camera(index):
    """Test a USB camera and return its properties"""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    
    # Try to read a frame
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return None
    
    # Get properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    backend = cap.getBackendName()
    
    cap.release()
    
    return {
        'index': index,
        'width': width,
        'height': height,
        'fps': fps,
        'backend': backend,
        'works': True
    }


def test_pi_camera():
    """Test Raspberry Pi Camera"""
    if not PICAMERA2_AVAILABLE:
        return None
    
    try:
        picam2 = Picamera2()
        config = picam2.create_video_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()
        
        # Try to capture
        frame = picam2.capture_array()
        picam2.stop()
        picam2.close()
        
        return {
            'type': 'Pi Camera (ribbon)',
            'width': frame.shape[1],
            'height': frame.shape[0],
            'works': True
        }
    except Exception as e:
        return {
            'type': 'Pi Camera (ribbon)',
            'works': False,
            'error': str(e)
        }


def main():
    """Main function to detect and list cameras"""
    print("=" * 60)
    print("Camera Detection Utility")
    print("=" * 60)
    print()
    
    # Test USB cameras
    print("Scanning for USB cameras...")
    print("-" * 60)
    
    available_cameras = list_available_cameras(max_test=10)
    
    if available_cameras:
        print(f"Found {len(available_cameras)} USB camera(s):")
        for idx in available_cameras:
            props = test_usb_camera(idx)
            if props:
                print(f"\n  Camera Index: {props['index']}")
                print(f"    Resolution: {props['width']}x{props['height']}")
                print(f"    FPS: {props['fps']}")
                print(f"    Backend: {props['backend']}")
                print(f"    Status: ✓ Working")
    else:
        print("  No USB cameras found")
    
    print()
    print("-" * 60)
    
    # Test Pi Camera
    print("\nTesting Raspberry Pi Camera (ribbon)...")
    print("-" * 60)
    
    pi_cam_result = test_pi_camera()
    if pi_cam_result:
        if pi_cam_result['works']:
            print("  Pi Camera: ✓ Available and working")
            print(f"    Resolution: {pi_cam_result['width']}x{pi_cam_result['height']}")
        else:
            print("  Pi Camera: ✗ Not available or error")
            if 'error' in pi_cam_result:
                print(f"    Error: {pi_cam_result['error']}")
    else:
        print("  Pi Camera: ✗ picamera2 not installed")
        print("    Install with: pip install picamera2")
    
    print()
    print("=" * 60)
    print("\nConfiguration Tips:")
    print("-" * 60)
    
    if available_cameras:
        print(f"  For USB camera, use: CAMERA_INDEX = {available_cameras[0]}")
        print(f"                      USE_PI_CAMERA = False")
    else:
        print("  No USB cameras found. Check camera connections.")
    
    if pi_cam_result and pi_cam_result.get('works'):
        print(f"  For Pi Camera, use: USE_PI_CAMERA = True")
        print(f"                     CAMERA_INDEX = 0 (ignored)")
    
    print()
    print("  You can also override in config.py or use command line:")
    print("    python main.py --camera 0")
    print("=" * 60)


if __name__ == "__main__":
    main()

