#!/usr/bin/env python3
"""
Check TFLite model input/output details
"""

import tensorflow.lite as tflite
from src.config import MODEL_PATHS

def check_model_details():
    print("Checking TFLite Model Details")
    print("=" * 30)

    # Load model
    interpreter = tflite.Interpreter(model_path=MODEL_PATHS['tflite'])
    interpreter.allocate_tensors()

    # Get input details
    input_details = interpreter.get_input_details()
    print("Input Details:")
    for i, detail in enumerate(input_details):
        print(f"  Input {i}: {detail}")

    # Get output details
    output_details = interpreter.get_output_details()
    print("\nOutput Details:")
    for i, detail in enumerate(output_details):
        print(f"  Output {i}: {detail}")

    print(f"\nInput shape: {input_details[0]['shape']}")
    print(f"Input dtype: {input_details[0]['dtype']}")
    print(f"Output shape: {output_details[0]['shape']}")
    print(f"Output dtype: {output_details[0]['dtype']}")

if __name__ == "__main__":
    check_model_details()