import cv2
import numpy as np
from src.config import CONFIDENCE_THRESHOLD, CLASS_NAMES, INPUT_SIZE

class Postprocessor:
    def __init__(self, confidence_threshold=CONFIDENCE_THRESHOLD, input_size=INPUT_SIZE):
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size

    def postprocess(self, outputs, original_image):
        """
        Postprocess model outputs to extract detections.
        YOLOv8 TFLite output format: [batch, 6, 8400]
        Where each detection has: [x, y, w, h, conf, class_score]
        """
        detections = []

        # Output shape: (1, 6, 8400) for YOLOv8
        batch_size, num_values, num_detections = outputs.shape

        # Assuming single batch
        output = outputs[0]  # Shape: (6, 8400)

        for i in range(num_detections):
            detection = output[:, i]  # Get one detection: [x, y, w, h, conf, class_score]

            # Extract values
            x_center, y_center, w, h, confidence, class_score = detection

            if confidence < self.confidence_threshold:
                continue

            # For binary classification (pothole/crack), class_score represents the positive class
            # If class_score > 0.5, it's class 1 (crack), else class 0 (pothole)
            class_id = 1 if class_score > 0.5 else 0

            # YOLO format: x, y, w, h (center, width, height)
            x_center, y_center, w, h = detection[0], detection[1], detection[2], detection[3]

            # Convert to corner coordinates
            x1 = int((x_center - w / 2) * original_image.shape[1] / self.input_size)
            y1 = int((y_center - h / 2) * original_image.shape[0] / self.input_size)
            x2 = int((x_center + w / 2) * original_image.shape[1] / self.input_size)
            y2 = int((y_center + h / 2) * original_image.shape[0] / self.input_size)

            # Clip bounding box coordinates to image boundaries
            height, width = original_image.shape[:2]
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))

            # Skip invalid bounding boxes (after clipping)
            if x2 <= x1 or y2 <= y1:
                continue

            detections.append({
                'class_id': class_id,
                'class_name': CLASS_NAMES.get(class_id, 'unknown'),
                'confidence': float(confidence),
                'bbox': (x1, y1, x2, y2)
            })

        return detections

    def draw_detections(self, image, detections):
        """Draw bounding boxes and labels on the image."""
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']

            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image