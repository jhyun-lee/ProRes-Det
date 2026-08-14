"""
Shared CLI plumbing, and the console-script entry points declared in setup.py.

demo.py / evaluate.py / train.py at the repo root are thin wrappers around
`demo_main` / `evaluate_main` / `train_main` here.
"""

import os
import sys
from datetime import datetime

from .config import PROJECT_ROOT, load_config, pick, resolve_path

# The bundled sample set is split three ways under data/SampleData (see
# data/README_data.md). demo.py and evaluate.py default to the validation split,
# because it is the only one that ships both surface/ and labels/ for every scene;
# train.py defaults to SAMPLE_TRAIN and never touches these two.
SAMPLE_ROOT = "data/SampleData"
SAMPLE_TRAIN = f"{SAMPLE_ROOT}/sample_train"
SAMPLE_EVAL = f"{SAMPLE_ROOT}/sample_eval"
SAMPLE_TEST = f"{SAMPLE_ROOT}/sample_test"

DEFAULT_INPUT = SAMPLE_EVAL
DEFAULT_GT = SAMPLE_EVAL
# The held-out projection source. data/live/train_light_{1,2,3}.mp4 are what
# collect.py turns into training light frames; --live defaults away from them so a
# demo never projects what the shipped checkpoint was fitted on.
DEFAULT_LIVE_VIDEO = "data/live/test_light.mp4"
DEFAULT_LIVE_BG = "data/live/BaseBackGround.jpg"
DEFAULT_OUTPUT = "output"


CUDA_INDEX = "https://download.pytorch.org/whl/cu128"


def detector_backends(det_cfg):
    """
    Which detection backends to run, from `detector.backend` in detection.yaml.

    A scalar names one; a list names several, which is what lets evaluate.py put one
    row per backend in the same report. demo.py restores and detects with a single
    detector, so it takes the first entry.
    """
    backend = pick(None, det_cfg, "detector", "backend", default="yolo")
    names = backend if isinstance(backend, (list, tuple)) else [backend]
    names = [str(b).strip() for b in names if str(b).strip()]
    if not names:
        raise SystemExit(
            "detector.backend is empty in configs/detection.yaml.\n"
            "    name one backend (backend: yolo) or several ([yolo, ssd]).")
    return names


def resolve_device(requested=None):
    if requested:
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def device_note(device) -> str:
    """
    One line naming what is about to run, and why it is not the GPU when it is not.

    Worth the lines: `pip install torch` hands Windows a CPU-only wheel, everything
    keeps working, and the only symptom is restoration running ~25x slower. Printing
    the build turns that into something you can see in the first second of a run.
    """
    try:
        import torch
    except ImportError:
        return "device: cpu (torch is not importable)"

    if str(device).startswith("cuda"):
        name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "?"
        return f"device: {device} - {name}, torch {torch.__version__}"
    if torch.version.cuda is None:
        return (f"device: cpu - torch {torch.__version__} is a CPU-only build, so "
                f"restoration runs ~25x slower than it has to.\n"
                f"    pip install --index-url {CUDA_INDEX} torch torchvision")
    if not torch.cuda.is_available():
        return (f"device: cpu - torch {torch.__version__} carries CUDA "
                f"{torch.version.cuda} but no usable GPU was found")
    return f"device: cpu - asked for, a GPU is available"


def build_models(device=None, need_detector=True, detector_backend=None):
    """
    Turn configs/*.yaml into (restorer, detector, info).

    Everything except the device comes from the YAML, so the entry points cannot
    drift apart on it. `detector_backend` is not a CLI override: it is how
    evaluate.py walks the list in `detector.backend` one entry at a time.
    """
    from .models import (
        build_detector, build_restorer, detector_names, restorer_names,
    )
    from .models.restoration import RestorationConfig

    # A malformed config or a missing PyYAML is the user's problem to fix, not a
    # traceback to read.
    try:
        rest_cfg = load_config("restoration")
        det_cfg = load_config("detection")
    except (FileNotFoundError, ImportError) as e:
        raise SystemExit(str(e)) from e

    device = resolve_device(device)
    print(device_note(device))

    input_size = tuple(pick(None, rest_cfg, "model", "input_size", default=[640, 360]))

    backend = pick(None, rest_cfg, "model", "backend", default="naf_se_unet")
    if backend not in restorer_names():
        raise SystemExit(
            f"unknown restorer '{backend}' in configs/restoration.yaml.\n"
            f"    available: {', '.join(restorer_names())}"
        )

    weights = resolve_path(pick(None, rest_cfg, "model", "weights"))
    if not weights or not os.path.exists(weights):
        raise SystemExit(
            f"restoration weights not found: {weights}\n"
            f"    set model.weights in configs/restoration.yaml, or drop the file "
            f"into {os.path.join(PROJECT_ROOT, 'weights')} "
            f"(see weights/README_weights.md)"
        )

    # A checkpoint embeds the config it was trained with, so an explicit config is
    # only needed for a legacy bare state_dict. A default-valued block stays None,
    # which is what keeps the checkpoint's own config winning.
    ablation = None
    if rest_cfg.get("ablation"):
        from_yaml = RestorationConfig.from_dict(rest_cfg["ablation"])
        ablation = None if from_yaml.is_default() else from_yaml

    try:
        restorer = build_restorer(backend, weights, device=device, cfg=ablation,
                                  input_size=input_size)
    except RuntimeError as e:
        raise SystemExit(f"could not load the restoration checkpoint.\n{e}") from e
    print(f"restorer: {restorer.describe()}")
    # The ablation config belongs to the shipped architecture, not to BaseRestorer.
    # Printing it unconditionally crashed any other registered backend.
    arch_cfg = getattr(restorer, "cfg", None)
    if arch_cfg is not None:
        print(f"    {arch_cfg.describe()}")

    detector = None
    if need_detector:
        backend = detector_backend or detector_backends(det_cfg)[0]
        # Before the weights lookup: an unrecognised backend has no weights entry, so
        # checking that first would blame the checkpoint for a misspelled name.
        if backend not in detector_names():
            raise SystemExit(
                f"unknown detector '{backend}' in configs/detection.yaml.\n"
                f"    available: {', '.join(detector_names())}"
            )
        conf = float(pick(None, det_cfg, "detector", "conf", default=0.25))
        names = det_cfg.get("names")
        det_weights = None
        if backend != "none":
            det_weights = resolve_path((det_cfg.get("weights") or {}).get(backend))
            if not det_weights or not os.path.exists(det_weights):
                raise SystemExit(
                    f"detector weights for '{backend}' not found: {det_weights}\n"
                    f"    set weights.{backend} in configs/detection.yaml, or set "
                    f"detector.backend to 'none' to run restoration only."
                )
        detector = build_detector(backend, weights=det_weights, class_names=names,
                                  conf=conf, device=device,
                                  imgsz=pick(None, det_cfg, "detector", "imgsz"))
        print(f"detector: {detector.describe()}")

    info = {"device": device, "input_size": input_size,
            "restoration_config": rest_cfg, "detection_config": det_cfg}
    return restorer, detector, info


def box_filter_kwargs(det_cfg):
    """The box size gate, from configs/detection.yaml."""
    return {
        "min_width": int(pick(None, det_cfg, "detector", "min_width", default=20)),
        "min_height": int(pick(None, det_cfg, "detector", "min_height", default=20)),
        "min_area": int(pick(None, det_cfg, "detector", "min_area", default=500)),
    }


def run_dir(base, name=None):
    """output/<name or timestamp>/, created."""
    name = name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    path = os.path.join(resolve_path(base) or base, name)
    os.makedirs(path, exist_ok=True)
    return path


def _delegate(module_name):
    """Import a root-level script's main(). Only works from a source checkout."""
    import importlib
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    try:
        return importlib.import_module(module_name).main
    except ImportError as e:
        raise SystemExit(
            f"could not import {module_name}.py from {PROJECT_ROOT}.\n"
            f"    The pdf-* console scripts require an editable install of a "
            f"checkout: pip install -e .\n"
            f"    From a plain wheel, use the projector_distortion package directly."
        ) from e


def demo_main(argv=None):
    return _delegate("demo")(argv)


def evaluate_main(argv=None):
    return _delegate("evaluate")(argv)


def train_main(argv=None):
    return _delegate("train")(argv)
