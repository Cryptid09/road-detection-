"""
Event Logger Module
Logs detected anomalies to CSV file
"""

import csv
import os
from datetime import datetime
from typing import List, Dict
import threading


class EventLogger:
    """
    Thread-safe CSV logger for detection events
    """
    
    def __init__(self, log_dir: str = "logs", filename: str = "detections.csv"):
        """
        Initialize event logger
        
        Args:
            log_dir: Directory to save log files
            filename: CSV filename
        """
        self.log_dir = log_dir
        self.filename = filename
        self.log_path = os.path.join(log_dir, filename)
        self.lock = threading.Lock()
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize CSV file with headers if it doesn't exist
        self._initialize_csv()
    
    def _initialize_csv(self):
        """Initialize CSV file with headers"""
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'class_name',
                    'class_id',
                    'confidence',
                    'x1',
                    'y1',
                    'x2',
                    'y2'
                ])
    
    def log_detections(self, detections: List[Dict], class_names: Dict[int, str]):
        """
        Log detections to CSV file
        
        Args:
            detections: List of detections, each with 'box', 'score', 'class_id'
            class_names: Dictionary mapping class_id to class name
        """
        if not detections:
            return
        
        with self.lock:
            with open(self.log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                timestamp = datetime.now().isoformat()
                
                for det in detections:
                    box = det['box']
                    class_id = det['class_id']
                    class_name = class_names.get(class_id, f"class_{class_id}")
                    
                    writer.writerow([
                        timestamp,
                        class_name,
                        class_id,
                        f"{det['score']:.4f}",
                        f"{box[0]:.2f}",
                        f"{box[1]:.2f}",
                        f"{box[2]:.2f}",
                        f"{box[3]:.2f}"
                    ])
    
    def get_log_path(self) -> str:
        """Get path to log file"""
        return self.log_path

