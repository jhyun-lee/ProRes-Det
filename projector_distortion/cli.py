"""
Shared CLI plumbing, and the console-script entry points declared in setup.py.

demo.py / evaluate.py / train.py at the repo root are thin wrappers around
`demo_main` / `evaluate_main` / `train_main` here.
"""

import os
import sys
from datetime import datetime

from .config import PROJECT_ROOT, load_config, pick, resolve_path

DEFAULT_INPUT = "data/sample_input"
DEFAULT_GT = "data/sample_gt"
DEFAULT_LIVE_VIDEO = "data/live/BeamVideo.mp4"
DEFAULT_LIVE_BG = "data/live/BaseBackGround.jpg"
DEFAULT_OUTPUT = "output"


def add_common_args(p):
    """Flags every entry point understands."""
    g = p.add_argument_group("models")
    g.add_argument("--restorer-weights", default=None,
                   help="restoration checkpoint (default: from configs/restoration.yaml)")
    g.add_argument("--detector", default=None, choices=["yolo", "ssd", "none"],
                   help="detection backend (default: from configs/detection.yaml)")
    g.add_argument("--det-weights", default=None,
                   help="detector checkpoint (default: per-backend from the config)")
    g.add_argument("--conf", type=float, default=None, help="detector confidence floor")
    g.add_argument("--classes", default=None,
                   help="dataset.yaml to take class names from")
    g.add_argument("--device", default=None, help="cuda | cpu (default: cuda if present)")
    g.add_argument("--fp16", action="store_true", help="autocast the restorer on CUDA")

    c = p.add_argument_group("config")
    c.add_argument("--restoration-config", default=None,
                   help="YAML merged over configs/restoration.yaml")
    c.add_argument("--detection-config", default=None,
                   help="YAML merged over configs/detection.yaml")
    return p


def resolve_device(requested=None):
    if requested:
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def build_models(args, need_detector=True):
    """
    Turn parsed args + YAML into (restorer, detector, info).

    Config precedence lives here so the three entry points stay identical on it.
    """
    from .data import load_class_names
    from .models import build_detector, build_restorer
    from .models.restoration import RestorationConfig, config_from_args

    rest_cfg = load_config("restoration", args.restoration_config)
    det_cfg = load_config("detection", args.detection_config)
    device = resolve_device(args.device)

    weights = resolve_path(pick(args.restorer_weights, rest_cfg, "model", "weights"))
    if not weights or not os.path.exists(weights):
        raise SystemExit(
            f"restoration weights not found: {weights}\n"
            f"    pass --restorer-weights <path>, or drop the file into "
            f"{os.path.join(PROJECT_ROOT, 'weights')} (see weights/README.md)"
        )

    # Ablation flags only exist on the parsers that registered them (demo/train).
    ablation = None
    if any(hasattr(args, a) for a in ("no_ca", "base_dim")):
        from_cli = config_from_args(args)
        ablation = from_cli if from_cli != RestorationConfig() else None
    if ablation is None and rest_cfg.get("ablation"):
        from_yaml = RestorationConfig.from_dict(rest_cfg["ablation"])
        ablation = from_yaml if from_yaml != RestorationConfig() else None

    input_size = pick(None, rest_cfg, "model", "input_size", default=[640, 360])
    fp16 = bool(args.fp16 or rest_cfg.get("model", {}).get("fp16"))
    restorer = build_restorer(weights, device=device, cfg=ablation,
                              input_size=tuple(input_size), fp16=fp16)
    print(f"restorer: {restorer.describe()}")
    print(f"    {restorer.cfg.describe()}")

    detector = None
    if need_detector:
        backend = pick(args.detector, det_cfg, "detector", "backend", default="yolo")
        conf = float(pick(args.conf, det_cfg, "detector", "conf", default=0.25))
        names = load_class_names(args.classes) if args.classes else det_cfg.get("names")
        det_weights = None
        if backend != "none":
            det_weights = resolve_path(
                args.det_weights or (det_cfg.get("weights") or {}).get(backend))
            if not det_weights or not os.path.exists(det_weights):
                raise SystemExit(
                    f"detector weights for '{backend}' not found: {det_weights}\n"
                    f"    pass --det-weights <path>, or use --detector none to run "
                    f"restoration only."
                )
        detector = build_detector(backend, weights=det_weights, class_names=names,
                                  conf=conf, device=device,
                                  imgsz=pick(None, det_cfg, "detector", "imgsz"))
        print(f"detector: {detector.describe()}")

    info = {"device": device, "restoration_config": rest_cfg, "detection_config": det_cfg}
    return restorer, detector, info


def box_filter_kwargs(args, det_cfg):
    """The box size gate + best_per_class, CLI over YAML."""
    best = getattr(args, "best_per_class", False) or pick(
        None, det_cfg, "detector", "best_per_class", default=False)
    return {
        "min_width": int(pick(None, det_cfg, "detector", "min_width", default=20)),
        "min_height": int(pick(None, det_cfg, "detector", "min_height", default=20)),
        "min_area": int(pick(None, det_cfg, "detector", "min_area", default=500)),
        "best_per_class": bool(best),
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
