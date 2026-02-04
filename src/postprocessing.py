"""
Postprocessing Module
Handles NMS, confidence filtering, and coordinate conversion
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> np.ndarray:
    """
    Non-Maximum Suppression to remove overlapping detections
    
    Args:
        boxes: Bounding boxes in format (N, 4) as (x1, y1, x2, y2)
        scores: Confidence scores (N,)
        iou_threshold: IoU threshold for NMS
    
    Returns:
        Indices of boxes to keep
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)
    
    # Convert to format expected by cv2.dnn.NMSBoxes: (x, y, w, h)
    boxes_xywh = np.zeros_like(boxes)
    boxes_xywh[:, 0] = boxes[:, 0]  # x1
    boxes_xywh[:, 1] = boxes[:, 1]  # y1
    boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]  # w = x2 - x1
    boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]  # h = y2 - y1
    
    # Use OpenCV NMS
    indices = cv2.dnn.NMSBoxes(
        boxes_xywh.tolist(),
        scores.tolist(),
        score_threshold=0.0,  # We filter by confidence separately
        nms_threshold=iou_threshold
    )
    
    if len(indices) == 0:
        return np.array([], dtype=np.int32)
    
    return indices.flatten()


def parse_yolo_output(output: np.ndarray,
                     confidence_threshold: float = 0.6,
                     nms_threshold: float = 0.45,
                     num_classes: int = 2) -> List[Dict]:
    """
    Parse YOLOv8 raw output into detections
    
    Args:
        output: Raw model output (1, 6, 8400) or (1, 4+num_classes, 8400)
               Format: [cx, cy, w, h, unused, score] (normalized) OR
                       [cx, cy, w, h, class0_score, class1_score, ...]
        confidence_threshold: Minimum confidence score
        nms_threshold: IoU threshold for NMS
        num_classes: Number of classes (default: 2 for pothole and crack)
    
    Returns:
        List of detections, each as dict with:
        - 'box': (x1, y1, x2, y2) in normalized coordinates
        - 'score': confidence score
        - 'class_id': class ID (0 or 1)
    """
    # Output shape: (1, channels, 8400)
    # Reshape to (8400, channels)
    predictions = output[0].transpose(1, 0)  # (8400, channels)
    
    # Extract bbox components (always first 4)
    cx = predictions[:, 0]  # Center x (normalized)
    cy = predictions[:, 1]  # Center y (normalized)
    w = predictions[:, 2]   # Width (normalized)
    h = predictions[:, 3]   # Height (normalized)
    
    # Handle different output formats
    if predictions.shape[1] == 6:
        # Format: [cx, cy, w, h, unused, score]
        # This is a single-class or objectness score
        scores = predictions[:, 5]
        # Default to class 0 (pothole) if format doesn't include class info
        class_ids = np.zeros(len(scores), dtype=np.int32)
    elif predictions.shape[1] >= 4 + num_classes:
        # Format: [cx, cy, w, h, class0_score, class1_score, ...]
        # Get class scores (columns 4 to 4+num_classes-1)
        class_scores = predictions[:, 4:4+num_classes]
        # Get max class and score for each detection
        class_ids = np.argmax(class_scores, axis=1)
        scores = np.max(class_scores, axis=1)
    else:
        raise ValueError(f"Unexpected output shape: {predictions.shape}")
    
    # Filter by confidence threshold
    valid_mask = scores >= confidence_threshold
    
    if not np.any(valid_mask):
        return []
    
    # Get valid predictions
    valid_cx = cx[valid_mask]
    valid_cy = cy[valid_mask]
    valid_w = w[valid_mask]
    valid_h = h[valid_mask]
    valid_scores = scores[valid_mask]
    valid_class_ids = class_ids[valid_mask]
    
    # Convert to xyxy format (normalized)
    x1 = valid_cx - valid_w / 2
    y1 = valid_cy - valid_h / 2
    x2 = valid_cx + valid_w / 2
    y2 = valid_cy + valid_h / 2
    
    # Clip to [0, 1]
    x1 = np.clip(x1, 0, 1)
    y1 = np.clip(y1, 0, 1)
    x2 = np.clip(x2, 0, 1)
    y2 = np.clip(y2, 0, 1)
    
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    
    # Apply NMS per class to avoid suppressing detections of different classes
    all_detections = []
    for class_id in range(num_classes):
        class_mask = valid_class_ids == class_id
        if not np.any(class_mask):
            continue
        
        class_boxes = boxes[class_mask]
        class_scores = valid_scores[class_mask]
        
        # Apply NMS for this class
        nms_indices = nms(class_boxes, class_scores, nms_threshold)
        
        # Add detections for this class
        for idx in nms_indices:
            all_detections.append({
                'box': class_boxes[idx],
                'score': float(class_scores[idx]),
                'class_id': int(class_id)
            })
    
    return all_detections


def convert_to_pixel_coords(detections: List[Dict],
                           scale: float,
                           pad: Tuple[int, int],
                           original_shape: Tuple[int, int],
                           input_size: int = 640) -> List[Dict]:
    """
    Convert normalized coordinates to pixel coordinates
    
    Args:
        detections: List of detections with normalized coordinates
        scale: Scale factor from preprocessing
        pad: (pad_x, pad_y) padding offsets
        original_shape: (height, width) of original image
        input_size: Model input size (default: 640)
    
    Returns:
        List of detections with pixel coordinates
    """
    from src.preprocessing import denormalize_coordinates
    
    pixel_detections = []
    for det in detections:
        pixel_box = denormalize_coordinates(
            det['box'],
            scale,
            pad,
            original_shape,
            input_size
        )
        
        pixel_detections.append({
            'box': pixel_box,
            'score': det['score'],
            'class_id': det['class_id']
        })
    
    return pixel_detections

