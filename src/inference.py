import numpy as np
from abc import ABC, abstractmethod
from src.config import MODEL_PATHS, MODEL_TYPE, INPUT_SIZE

class InferenceEngine(ABC):
    @abstractmethod
    def load_model(self, model_path):
        pass

    @abstractmethod
    def run_inference(self, preprocessed_image):
        pass

class ONNXInference(InferenceEngine):
    def __init__(self):
        self.session = None

    def load_model(self, model_path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path)
        print(f"ONNX model loaded from {model_path}")

    def run_inference(self, preprocessed_image):
        # ONNX expects (N, C, H, W) format
        input_tensor = np.transpose(preprocessed_image, (0, 3, 1, 2))
        inputs = {self.session.get_inputs()[0].name: input_tensor}
        outputs = self.session.run(None, inputs)
        return outputs[0]  # Assuming single output

class TFLiteInference(InferenceEngine):
    def __init__(self):
        self.interpreter = None

    def load_model(self, model_path):
        import tensorflow.lite as tflite
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        print(f"TFLite model loaded from {model_path}")

    def run_inference(self, preprocessed_image):
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        # Set input tensor
        self.interpreter.set_tensor(input_details[0]['index'], preprocessed_image)

        # Run inference
        self.interpreter.invoke()

        # Get output tensor
        output = self.interpreter.get_tensor(output_details[0]['index'])
        return output

def get_inference_engine(model_type=MODEL_TYPE):
    if model_type == 'onnx':
        return ONNXInference()
    elif model_type == 'tflite':
        return TFLiteInference()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")