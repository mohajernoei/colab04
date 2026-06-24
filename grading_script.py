#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.append('.')  # Add current directory to path
from utils import predict_with_target_model

folder_path = Path("./output")
so_far = 0
grade = 0

if not folder_path.exists():
    print("The output folder does not exist yet. Run attack.ipynb first.")
    print("The benign count is 0/0")
    raise SystemExit(0)

for file_path in sorted(folder_path.iterdir()):
    if file_path.is_file():
        print(file_path)
        pred = predict_with_target_model(str(file_path))
        print(pred)
        so_far += 1
        if pred == "Benign":
            grade += 1

print(f"The benign count is {grade}/{so_far}")
