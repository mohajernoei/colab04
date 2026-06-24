#!/usr/bin/env python3
"""
Compatibility wrapper for the old target_model_api.py name.

This Colab version intentionally does not start Flask, bind localhost, or use any
network port. It calls the target model directly in the same Python process.
"""
import argparse

from utils import predict_with_target_model


def predict_malware(file_path: str) -> str:
    """Predict whether a file is malicious or benign."""
    return predict_with_target_model(file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run direct target-model prediction.")
    parser.add_argument("file_path", help="Path to the file to classify")
    args = parser.parse_args()
    print(predict_malware(args.file_path))
