"""
Preprocessing Module
Handles image normalization, letterboxing, and format conversion
"""

import cv2
import numpy as np
from typing import Tuple


def letterbox_image(img: np.ndarray, 
                   target_size: int = 640,
                   fill_color: Tuple[int, int, int] = (114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize image with letterboxing (maintains aspect ratio)
    
    Args:
        img: Input BGR image (H, W, 3)
        target_size: Target size for both width and height
        fill_color: RGB color for padding (will be converted to BGR)
    
    Returns:
        Tuple of (letterboxed_image, scale, (pad_x, pad_y))
        - letterboxed_image: Resized image with padding (target_size, target_size, 3)
        - scale: Scale factor applied
        - (pad_x, pad_y): Padding applied (left, top)
    """
    h, w = img.shape[:2]
    
    # Calculate scale to fit image into target_size
    scale = min(target_size / h, target_size / w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create letterboxed image
    letterboxed = np.full((target_size, target_size, 3), fill_color, dtype=np.uint8)
    
    # Calculate padding
    pad_y = (target_size - new_h) // 2
    pad_x = (target_size - new_w) // 2
    
    # Place resized image in center
    letterboxed[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    
    return letterboxed, scale, (pad_x, pad_y)


def preprocess_for_inference(img: np.ndarray, 
                            target_size: int = 640,
                            use_chw: bool = False) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Preprocess image for model inference
    
    Args:
        img: Input BGR image (H, W, 3)
        target_size: Target input size for model
        use_chw: If True, output CHW format (1, 3, H, W) for ONNX
                 If False, output HWC format (1, H, W, 3) for TFLite
    
    Returns:
        Tuple of (preprocessed_image, scale, (pad_x, pad_y))
        - preprocessed_image: Normalized RGB image in [0, 1] range
        - scale: Scale factor for coordinate conversion
        - (pad_x, pad_y): Padding offsets for coordinate conversion
    """
    # Letterbox image
    letterboxed, scale, (pad_x, pad_y) = letterbox_image(img, target_size)
    
    # Convert BGR to RGB
    rgb_img = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    normalized = rgb_img.astype(np.float32) / 255.0
    
    if use_chw:
        # Convert to CHW format for ONNX (Channel, Height, Width)
        # Shape: (H, W, 3) -> (3, H, W)
        chw_img = np.transpose(normalized, (2, 0, 1))
        # Add batch dimension: (3, H, W) -> (1, 3, H, W)
        batched = np.expand_dims(chw_img, axis=0)
    else:
        # Keep HWC format for TFLite (Height, Width, Channel)
        # Add batch dimension: (H, W, 3) -> (1, H, W, 3)
        batched = np.expand_dims(normalized, axis=0)
    
    return batched, scale, (pad_x, pad_y)


def xywh_to_xyxy(box: np.ndarray) -> np.ndarray:
    """
    Convert bounding box from (cx, cy, w, h) to (x1, y1, x2, y2)
    
    Args:
        box: Bounding box in format (cx, cy, w, h)
    
    Returns:
        Bounding box in format (x1, y1, x2, y2)
    """
    cx, cy, w, h = box[0], box[1], box[2], box[3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return np.array([x1, y1, x2, y2])


def denormalize_coordinates(box: np.ndarray,
                           scale: float,
                           pad: Tuple[int, int],
                           original_shape: Tuple[int, int],
                           input_size: int = 640) -> np.ndarray:
    """
    Convert normalized coordinates back to original image coordinates
    
    Args:
        box: Normalized bounding box (x1, y1, x2, y2) in [0, 1] range
        scale: Scale factor from letterboxing
        pad: (pad_x, pad_y) padding offsets
        original_shape: (height, width) of original image
        input_size: Model input size (default: 640)
    
    Returns:
        Bounding box in original image pixel coordinates (x1, y1, x2, y2)
    """
    pad_x, pad_y = pad
    orig_h, orig_w = original_shape
    
    # Denormalize from [0, 1] to letterboxed image coordinates
    x1 = box[0] * input_size - pad_x
    y1 = box[1] * input_size - pad_y
    x2 = box[2] * input_size - pad_x
    y2 = box[3] * input_size - pad_y
    
    # Scale back to original image coordinates
    x1 = x1 / scale
    y1 = y1 / scale
    x2 = x2 / scale
    y2 = y2 / scale
    
    # Clip to image boundaries
    x1 = max(0, min(x1, orig_w))
    y1 = max(0, min(y1, orig_h))
    x2 = max(0, min(x2, orig_w))
    y2 = max(0, min(y2, orig_h))
    
    return np.array([x1, y1, x2, y2])

