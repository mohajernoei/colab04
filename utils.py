import os
import warnings
from pathlib import Path

# Keep notebook/Colab output readable. These environment variables must be set
# before optional TensorFlow/secml imports happen.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# The project runs everything on CPU. This avoids noisy CUDA-driver warnings on
# machines/Colab runtimes with mismatched CUDA installations.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

warnings.filterwarnings(
    "ignore",
    message=r"CUDA initialization:.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

# Make `from tqdm import trange` in attack.ipynb use the notebook-aware progress
# display without editing the notebook. attack.ipynb imports utils before trange.
try:
    import tqdm as _tqdm_package
    from tqdm.auto import tqdm as _auto_tqdm

    def _notebook_friendly_trange(*args, **kwargs):
        kwargs.setdefault("dynamic_ncols", True)
        kwargs.setdefault("leave", True)
        kwargs.setdefault("mininterval", 0.1)
        return _auto_tqdm(range(*args), **kwargs)

    _tqdm_package.trange = _notebook_friendly_trange
except Exception:
    # Progress display is helpful but should never prevent the notebook from
    # running if tqdm behaves differently in an environment.
    pass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

_PROJECT_DIR = Path(__file__).resolve().parent


def _resolve_project_path(relative_path: str) -> Path:
    """
    Resolve a project-relative path in a Colab-friendly way.

    The notebook is usually run from the project directory, but this also works
    when utils.py is imported from another working directory.
    """
    cwd_path = Path(relative_path)
    module_path = _PROJECT_DIR / relative_path

    if cwd_path.exists():
        return cwd_path
    return module_path


transform = transforms.Compose([
    transforms.ToTensor(),
])


# ---------------------------------------------------------------------
# Direct in-process target model prediction, no Flask/server/ports
# ---------------------------------------------------------------------

_TARGET_MODEL_STATE = None


def _load_target_model_once():
    """
    Lazily load the MalConv target model once per Python runtime.

    This replaces the old localhost Flask API. Keeping the load lazy makes
    `import utils` fast and avoids importing secml_malware until the first
    actual call to predict_with_target_model(...).
    """
    global _TARGET_MODEL_STATE

    if _TARGET_MODEL_STATE is not None:
        return _TARGET_MODEL_STATE

    try:
        from secml.array import CArray
        from secml_malware.models.c_classifier_end2end_malware import (
            CClassifierEnd2EndMalware,
            End2EndModel,
        )
        from secml_malware.models.malconv import MalConv
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing secml/secml_malware dependencies. In Colab, run:\n"
            "    !pip install -r requirements.txt\n"
            "then restart the runtime if Colab asks you to."
        ) from exc

    print("[target] Loading MalConv target model once in-process...", flush=True)

    net = MalConv()
    classifier = CClassifierEnd2EndMalware(net)
    classifier.load_pretrained_model()
    max_len = classifier.get_input_max_length()

    print("[target] Target model ready. Continuing attack loop.", flush=True)

    _TARGET_MODEL_STATE = {
        "classifier": classifier,
        "max_len": max_len,
        "CArray": CArray,
        "End2EndModel": End2EndModel,
    }
    return _TARGET_MODEL_STATE


def predict_with_target_model(path: str) -> str:
    """
    Predict whether a file is malicious or benign using the target MalConv model.

    Colab/no-server version:
    - no Flask
    - no HTTP requests
    - no localhost port 5000
    - same public interface as before: predict_with_target_model(path)
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    state = _load_target_model_once()
    classifier = state["classifier"]
    max_len = state["max_len"]
    CArray = state["CArray"]
    End2EndModel = state["End2EndModel"]

    with file_path.open("rb") as file_handle:
        code = file_handle.read()

    x = End2EndModel.bytes_to_numpy(code, max_len, 256, False)
    x = np.asarray(x, dtype=np.uint8, order="C")
    sample = CArray(x, copy=True)

    _, confidence = classifier.predict(sample, True)

    return "Malicious" if confidence[0, 1].item() > 0.5 else "Benign"


# ---------------------------------------------------------------------
# Surrogate model loading helpers
# ---------------------------------------------------------------------

def load_resnet50():
    model_path = _resolve_project_path(".bts/models/resnet50.pth")

    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)

    model.load_state_dict(
        torch.load(model_path, map_location=torch.device("cpu"))
    )
    model.eval()

    return model


def load_mobilenetv2():
    model_path = _resolve_project_path(".bts/models/mobilenetv2.pth")

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)

    model.load_state_dict(
        torch.load(model_path, map_location=torch.device("cpu"))
    )
    model.eval()

    return model


# ---------------------------------------------------------------------
# FGSM helpers
# ---------------------------------------------------------------------

def loss_function_gradient(x, model):
    """
    Compute gradient of malware-class loss with respect to input x.
    """
    x = x.detach().clone()
    x.requires_grad_(True)

    y = model(x)
    malware_class = torch.ones_like(y)

    loss = F.binary_cross_entropy_with_logits(y, malware_class)
    grad = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0]

    return grad.detach()[0]


def _internal_fsgm_solver(grad, p_norm=np.inf, epsilon=0.5):
    """
    Compute an FGSM-style perturbation.
    """
    if torch.count_nonzero(grad).item() == 0:
        g = torch.zeros_like(grad)
    else:
        g = grad / torch.norm(grad)

    if p_norm == 2:
        return epsilon * g

    if p_norm == np.infty:
        return epsilon * torch.sign(g)

    raise NotImplementedError(f"{p_norm}-norm not yet implemented")


def pertubation_optimizer(grad, len_of_exe, x, padding_length):
    """
    Apply perturbation only to the appended padding region.
    """
    gradient_result = _internal_fsgm_solver(grad)

    adv_x = x[0].detach().clone().flatten()
    gradient_result = gradient_result.flatten()

    start = len_of_exe
    end = min(len_of_exe + padding_length, adv_x.numel())

    adv_x[start:end] += gradient_result[start:end]

    x = x.detach().clone()
    x[0] = adv_x.reshape(adv_x.shape[0] // 1048576, 1024, 1024)

    return x


def fgsm(model, image, len_of_exe, padding_length):
    """
    Apply one FGSM step.
    """
    grad = loss_function_gradient(image, model)
    image = pertubation_optimizer(grad, len_of_exe, image, padding_length)

    with torch.no_grad():
        prediction = torch.round(torch.sigmoid(model(image)))

    return prediction, image


# ---------------------------------------------------------------------
# EXE/image conversion helpers
# ---------------------------------------------------------------------

def img_to_exe(image, input_path, output_path, len_of_exe, padding_length):
    """
    Write modified padding bytes from tensor image back into an executable.
    """
    output_parent = Path(output_path).parent
    if str(output_parent) not in ("", "."):
        output_parent.mkdir(parents=True, exist_ok=True)

    image_array = image.detach().cpu().numpy()
    relevant_data = image_array.flatten()

    with open(input_path, "rb") as file_handle:
        code = bytearray(file_handle.read())

    binary_data = relevant_data.astype(np.uint8).tobytes()
    max_index = len(binary_data)

    if max_index >= len(code):
        code.extend([0] * (max_index + 1 - len(code)))

    start = len_of_exe
    end = len_of_exe + padding_length

    code[start:end] = binary_data[start:end]

    with open(output_path, "wb") as exe_file:
        exe_file.write(code)


def exe_to_img(code):
    """
    Convert executable bytes into a 1024x1024 3-channel tensor image.
    """
    x = bytes_to_numpy(code, 1048576, 256, False)
    x = x.reshape((1024, 1024))

    x = np.stack([x] * 3, axis=0)
    x = np.transpose(x, (1, 2, 0))

    img = Image.fromarray(x.astype("uint8"))
    img = transform(img).unsqueeze(0)

    return img, len(code)


def bytes_to_numpy(
    bytez: bytes,
    max_length: int,
    padding_value: int,
    shift_values: bool
) -> np.ndarray:
    """
    Convert bytes to a fixed-length numpy array.
    """
    b = np.ones((max_length,), dtype=np.uint16) * padding_value

    bytez = np.frombuffer(bytez[:max_length], dtype=np.uint8)
    bytez = bytez.astype(np.uint16) + shift_values

    b[: len(bytez)] = bytez

    return np.array(b, dtype=float)
