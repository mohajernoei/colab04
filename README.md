# Colab04: Adversarial Malware Classification Lab

This project is a Google Colab-oriented adversarial machine-learning lab for malware classification research. It uses surrogate deep-learning models to generate modified executable samples and then evaluates those generated files with a MalConv-based target classifier.

The current version runs in **no-server mode**. It does not start Flask, does not open a local port, and does not call a localhost API. The target model is loaded directly inside Python when `predict_with_target_model(...)` is first called.

> **Safety note:** This project is intended only for coursework, defensive cybersecurity education, and controlled research environments. Do not use it to modify, distribute, execute, or deploy malware outside an authorized isolated lab.

## Repository Structure

```text
.
├── attack.ipynb              # Main Colab notebook
├── grading_script.py         # Evaluates generated files in ./output
├── requirements.txt          # Python dependencies
├── utils.py                  # Model loading, prediction, FGSM, and file-conversion helpers
├── README.md                 # Project documentation
├── samples/                  # Input sample files used by the notebook
├── output/                   # Generated files created by the notebook
└── .bts/
    └── models/
        ├── resnet50.pth      # ResNet50 surrogate model weights
        └── mobilenetv2.pth   # MobileNetV2 surrogate model weights
```

## Main Components

### `attack.ipynb`

This is the main notebook. It installs dependencies, imports `utils.py`, reads files from `./samples/`, generates adversarially modified outputs, saves them in `./output/`, and then runs `grading_script.py`.

The notebook uses two surrogate models:

* ResNet50
* MobileNetV2

For each sample file, the notebook first tries to craft a successful output with ResNet50. If that fails, it then tries MobileNetV2. At the end, the notebook runs the grading script automatically.

### `utils.py`

This file contains the core project logic:

* Lazy loading of the MalConv target model
* `predict_with_target_model(path)` for target-model prediction
* ResNet50 and MobileNetV2 surrogate model loading
* FGSM-style gradient perturbation helpers
* Executable-to-image conversion
* Image-to-executable conversion
* Colab-friendly warning suppression and progress-bar behavior

The target model is loaded only once per Python runtime, which avoids repeatedly reloading the classifier.

### `grading_script.py`

This script checks every file in the `./output/` directory. For each output file, it calls:

```python
predict_with_target_model(str(file_path))
```

It prints the prediction for each file and then reports how many generated files were classified as `Benign`.

Example output:

```text
output/sample1.exe
Benign
output/sample2.exe
Malicious
The benign count is 1/2
```

### `requirements.txt`

This file installs the main Python dependencies, including:

* PyTorch, TorchVision, and TorchAudio CUDA 12.1 builds
* NumPy
* tqdm and ipywidgets
* EMBER
* DEAP
* LIEF
* LightGBM
* scikit-learn
* pandas
* Pillow
* requests

The notebook also installs `secml_malware` and `secml` separately with `--no-deps`.

## Required Files and Folders

Before running the notebook, make sure the project contains these required files:

```text
attack.ipynb
utils.py
grading_script.py
requirements.txt
.bts/models/resnet50.pth
.bts/models/mobilenetv2.pth
samples/
```

The `.bts/models/` folder is required because `utils.py` loads the surrogate model weights from that location.

## File Name Warning

If the files were downloaded or uploaded with names like this:

```text
attack(11).ipynb
utils(24).py
grading_script(9).py
requirements(43).txt
README(2).md
```

rename them before running the project:

```text
attack.ipynb
utils.py
grading_script.py
requirements.txt
README.md
```

The notebook and scripts expect the standard names. For example, `grading_script.py` imports from `utils`, and the notebook installs dependencies from `requirements.txt`.

## Installation

In Google Colab or a compatible Linux/Python environment, install the dependencies with:

```bash
pip install -r requirements.txt
```

Then install the SecML malware packages as used by the notebook:

```bash
pip install --no-deps secml_malware
pip install --no-deps secml
```

If Colab asks you to restart the runtime after installation, restart it and then continue running the notebook.

## Usage

### 1. Prepare the project folder

Make sure the notebook, scripts, requirements file, `.bts/models/`, and `samples/` folder are in the same project directory.

### 2. Run the notebook

Open and run:

```text
attack.ipynb
```

The notebook reads input files from:

```text
samples/
```

and writes generated files to:

```text
output/
```

### 3. Grade the generated files

The notebook runs the grading script automatically at the end. You can also run it manually:

```bash
python grading_script.py
```

The script prints each generated file, its target-model prediction, and the final benign count.

## How the Workflow Operates

At a high level, the workflow is:

1. Load dependencies and helper functions.
2. Load the surrogate ResNet50 and MobileNetV2 models.
3. Read executable samples from `./samples/`.
4. Convert each executable into a fixed-size tensor representation.
5. Apply FGSM-style perturbations to the appended padding region.
6. Write the modified bytes back into an output executable file.
7. Query the in-process MalConv target model.
8. Stop early if the target model classifies the generated output as `Benign`.
9. Save generated outputs in `./output/`.
10. Run `grading_script.py` to summarize the results.

## No-Flask / No-Server Mode

This project does not require a background server.

The current implementation avoids:

* Flask
* localhost APIs
* opening port 5000
* HTTP requests to a target model

Instead, `utils.py` loads the target MalConv model directly inside the Python process. This makes the project easier to run in Colab and avoids port-related errors.

## Troubleshooting

### `ModuleNotFoundError: No module named 'utils'`

Make sure `utils.py` is in the same directory as `attack.ipynb` and `grading_script.py`.

### `FileNotFoundError` for `.bts/models/resnet50.pth` or `.bts/models/mobilenetv2.pth`

Make sure the `.bts/models/` folder exists and contains:

```text
resnet50.pth
mobilenetv2.pth
```

### `The output folder does not exist yet`

Run `attack.ipynb` first. The grading script only checks files that already exist in `./output/`.

### Dependency errors for `secml` or `secml_malware`

Run:

```bash
pip install --no-deps secml_malware
pip install --no-deps secml
```

Then restart the runtime if necessary.

### Dependency conflicts with NumPy

The requirements file pins NumPy below version 2. If you see NumPy compatibility errors, reinstall the requirements and restart the runtime:

```bash
pip install -r requirements.txt
```

## What Can Be Removed

The current project does not require a startup server script. If an old file such as `startup_server.py`, `startup_server.sh`, or `startup_script.sh` exists, it can be removed unless your platform specifically requires it.

Do not remove:

```text
attack.ipynb
utils.py
grading_script.py
requirements.txt
.bts/models/
samples/
```

## Ethics and Responsible Use

This repository is for defensive research and education only. Use it only with authorized samples, isolated environments, and systems you are allowed to test.

Do not use this project to:

* Evade real-world security products
* Modify or distribute malware
* Test against systems without permission
* Deploy generated files outside a controlled lab

## License

Add the license required by your course, instructor, institution, or repository policy.

