"""
ONNX Inference Engine
Handles ONNX model loading and inference
"""

import numpy as np
import onnxruntime as ort
from typing import Tuple, Optional
import os


class ONNXInferenceEngine:
    """
    ONNX Runtime inference engine for YOLOv8
    """
    
    def __init__(self, model_path: str, providers: list = None):
        """
        Initialize ONNX inference engine
        
        Args:
            model_path: Path to ONNX model file
            providers: ONNX Runtime execution providers (default: CPU)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        
        # Default to CPU provider for edge deployment
        if providers is None:
            providers = ['CPUExecutionProvider']
        
        # Create inference session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )
        
        # Get input/output details
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
        print(f"ONNX Model loaded: {model_path}")
        print(f"Input shape: {self.input_shape}")
        print(f"Providers: {self.session.get_providers()}")
    
    def predict(self, preprocessed_img: np.ndarray) -> np.ndarray:
        """
        Run inference on preprocessed image
        
        Args:
            preprocessed_img: Preprocessed image (1, 3, H, W) in [0, 1] range
        
        Returns:
            Raw model output (1, 6, 8400) where columns are:
            [cx, cy, w, h, unused (~0), score] (all normalized)
        """
        # Run inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: preprocessed_img}
        )
        
        # Output shape: (1, 6, 8400)
        # Format: [cx, cy, w, h, unused, score]
        return outputs[0]
    
    def __call__(self, preprocessed_img: np.ndarray) -> np.ndarray:
        """Make engine callable"""
        return self.predict(preprocessed_img)

