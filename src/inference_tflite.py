"""
TensorFlow Lite Inference Engine
Handles TFLite INT8 quantized model loading and inference
"""

import numpy as np
from typing import Tuple, Optional
import os

try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    try:
        import tensorflow.lite as tflite
        TFLITE_AVAILABLE = True
    except ImportError:
        TFLITE_AVAILABLE = False
        tflite = None


class TFLiteInferenceEngine:
    """
    TensorFlow Lite inference engine for YOLOv8 (INT8 quantized)
    """
    
    def __init__(self, model_path: str):
        """
        Initialize TFLite inference engine
        
        Args:
            model_path: Path to TFLite model file
        """
        if not TFLITE_AVAILABLE:
            raise RuntimeError(
                "TensorFlow Lite not available.\n"
                "For Raspberry Pi: pip install tflite-runtime>=2.14.0\n"
                "For development/laptop: pip install tensorflow>=2.14.0\n"
                "Note: tensorflow includes tensorflow.lite and works on all platforms"
            )
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"TFLite model not found: {model_path}")
        
        # Load TFLite model
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.input_index = self.input_details[0]['index']
        self.output_index = self.output_details[0]['index']
        
        # Get input/output shapes
        self.input_shape = self.input_details[0]['shape']
        self.output_shape = self.output_details[0]['shape']
        
        # Check if model is quantized
        self.is_quantized = self.input_details[0]['dtype'] != np.float32
        
        print(f"TFLite Model loaded: {model_path}")
        print(f"Input shape: {self.input_shape}")
        print(f"Input dtype: {self.input_details[0]['dtype']}")
        print(f"Output shape: {self.output_shape}")
        print(f"Quantized: {self.is_quantized}")
    
    def predict(self, preprocessed_img: np.ndarray) -> np.ndarray:
        """
        Run inference on preprocessed image
        
        Args:
            preprocessed_img: Preprocessed image
                - For INT8 quantized: (1, H, W, 3) uint8 in [0, 255] range
                - For float32: (1, H, W, 3) float32 in [0, 1] range
        
        Returns:
            Raw model output (1, 6, 8400) where columns are:
            [cx, cy, w, h, unused (~0), score] (all normalized)
        """
        # Set input tensor
        self.interpreter.set_tensor(self.input_index, preprocessed_img)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output = self.interpreter.get_tensor(self.output_index)
        
        # Handle dequantization if needed (for INT8 models)
        if self.output_details[0]['dtype'] != np.float32:
            output_scale, output_zero_point = self.output_details[0]['quantization']
            output = (output.astype(np.float32) - output_zero_point) * output_scale
        
        # Output shape: (1, 6, 8400)
        # Format: [cx, cy, w, h, unused, score]
        return output
    
    def __call__(self, preprocessed_img: np.ndarray) -> np.ndarray:
        """Make engine callable"""
        return self.predict(preprocessed_img)

