import csv
import os
from datetime import datetime
from src.config import LOG_FILE, LOG_HEADERS

class Logger:
    def __init__(self, log_file=LOG_FILE, headers=LOG_HEADERS):
        self.log_file = log_file
        self.headers = headers
        self._ensure_log_file()

    def _ensure_log_file(self):
        """Ensure the log file exists with headers."""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def log_detection(self, detection):
        """Log a single detection to CSV."""
        timestamp = datetime.now().isoformat()
        class_name = detection['class_name']
        confidence = detection['confidence']
        x1, y1, x2, y2 = detection['bbox']

        row = [timestamp, class_name, confidence, x1, y1, x2, y2]

        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)