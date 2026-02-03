"""
Visualization Module
Draws bounding boxes and labels on images
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple


# Color palette for classes (BGR format for OpenCV)
CLASS_COLORS = {
    0: (0, 165, 255),    # Orange for pothole
    1: (0, 255, 0),      # Green for crack
}

# Default color if class not in palette
DEFAULT_COLOR = (255, 255, 255)  # White


def draw_detections(img: np.ndarray,
                   detections: List[Dict],
                   class_names: Dict[int, str],
                   thickness: int = 2,
                   font_scale: float = 0.6) -> np.ndarray:
    """
    Draw bounding boxes and labels on image
    
    Args:
        img: Input BGR image
        detections: List of detections with 'box', 'score', 'class_id'
        class_names: Dictionary mapping class_id to class name
        thickness: Line thickness for bounding boxes
        font_scale: Font scale for labels
    
    Returns:
        Image with drawn detections
    """
    img_copy = img.copy()
    
    for det in detections:
        box = det['box']
        class_id = det['class_id']
        score = det['score']
        
        # Get color for this class
        color = CLASS_COLORS.get(class_id, DEFAULT_COLOR)
        
        # Convert box coordinates to integers
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        
        # Draw bounding box
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label text
        class_name = class_names.get(class_id, f"class_{class_id}")
        label = f"{class_name}: {score:.2f}"
        
        # Calculate label size for background
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        
        # Draw label background
        cv2.rectangle(
            img_copy,
            (x1, y1 - label_height - baseline - 5),
            (x1 + label_width, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            img_copy,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),  # Black text
            1
        )
    
    return img_copy


def draw_fps(img: np.ndarray, fps: float, position: Tuple[int, int] = (10, 30)) -> np.ndarray:
    """
    Draw FPS counter on image
    
    Args:
        img: Input BGR image
        fps: Current FPS value
        position: (x, y) position for FPS text
    
    Returns:
        Image with FPS drawn
    """
    img_copy = img.copy()
    fps_text = f"FPS: {fps:.1f}"
    
    # Draw background rectangle
    (text_width, text_height), baseline = cv2.getTextSize(
        fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    )
    cv2.rectangle(
        img_copy,
        (position[0] - 5, position[1] - text_height - 5),
        (position[0] + text_width + 5, position[1] + 5),
        (0, 0, 0),  # Black background
        -1
    )
    
    # Draw FPS text
    cv2.putText(
        img_copy,
        fps_text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),  # Green text
        2
    )
    
    return img_copy

