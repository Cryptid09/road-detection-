"""
Event Logger Module
Logs detected anomalies to CSV file and optionally saves images
"""

import csv
import os
from datetime import datetime
from typing import List, Dict, Optional
import threading
import cv2
import numpy as np


class EventLogger:
    """
    Thread-safe CSV logger for detection events
    """
    
    def __init__(self, log_dir: str = "logs", filename: str = "detections.csv", 
                 save_images: bool = False, images_dir: str = "logs/images"):
        """
        Initialize event logger
        
        Args:
            log_dir: Directory to save log files
            filename: CSV filename
            save_images: If True, save images with detections
            images_dir: Directory to save detection images
        """
        self.log_dir = log_dir
        self.filename = filename
        self.log_path = os.path.join(log_dir, filename)
        self.save_images = save_images
        self.images_dir = images_dir
        self.lock = threading.Lock()
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create images directory if saving images
        if self.save_images:
            os.makedirs(images_dir, exist_ok=True)
        
        # Initialize CSV file with headers if it doesn't exist
        self._initialize_csv()
    
    def _initialize_csv(self):
        """Initialize CSV file with headers"""
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                headers = [
                    'timestamp',
                    'class_name',
                    'class_id',
                    'confidence',
                    'x1',
                    'y1',
                    'x2',
                    'y2'
                ]
                if self.save_images:
                    headers.append('image_path')
                writer.writerow(headers)
    
    def log_detections(self, detections: List[Dict], class_names: Dict[int, str], 
                      frame: Optional[np.ndarray] = None):
        """
        Log detections to CSV file and optionally save images
        
        Args:
            detections: List of detections, each with 'box', 'score', 'class_id'
            class_names: Dictionary mapping class_id to class name
            frame: Optional frame image to save (BGR format)
        """
        if not detections:
            return
        
        with self.lock:
            timestamp = datetime.now()
            timestamp_str = timestamp.isoformat()
            
            # Save image if enabled and frame provided
            image_path = None
            if self.save_images and frame is not None:
                # Create unique filename with timestamp
                timestamp_filename = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
                image_filename = f"detection_{timestamp_filename}.jpg"
                image_path = os.path.join(self.images_dir, image_filename)
                
                # Draw detections on frame before saving
                from src.visualization import draw_detections
                annotated_frame = draw_detections(frame.copy(), detections, class_names)
                
                # Save image
                cv2.imwrite(image_path, annotated_frame)
                # Use relative path in CSV
                image_path = os.path.join("images", image_filename)
            
            # Write to CSV
            with open(self.log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                
                for det in detections:
                    box = det['box']
                    class_id = det['class_id']
                    class_name = class_names.get(class_id, f"class_{class_id}")
                    
                    row = [
                        timestamp_str,
                        class_name,
                        class_id,
                        f"{det['score']:.4f}",
                        f"{box[0]:.2f}",
                        f"{box[1]:.2f}",
                        f"{box[2]:.2f}",
                        f"{box[3]:.2f}"
                    ]
                    
                    if self.save_images:
                        row.append(image_path if image_path else "")
                    
                    writer.writerow(row)
    
    def get_log_path(self) -> str:
        """Get path to log file"""
        return self.log_path

