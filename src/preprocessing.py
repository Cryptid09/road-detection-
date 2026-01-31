import cv2
import numpy as np
from src.config import INPUT_SIZE

class Preprocessor:
    def __init__(self, input_size=INPUT_SIZE):
        self.input_size = input_size

    def preprocess(self, image):
        """
        Preprocess the input image for model inference.
        - Resize to input_size x input_size
        - Normalize to [0, 1]
        - Convert to float32
        """
        # Resize image
        resized = cv2.resize(image, (self.input_size, self.input_size))

        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        # Add batch dimension (1, H, W, C)
        preprocessed = np.expand_dims(normalized, axis=0)

        return preprocessed