"""Swappable model layer: restoration + detection behind two small interfaces."""

from .base import (
    BaseDetector,
    BaseRestorer,
    Detection,
    NullDetector,
    detector_names,
    get_detector_class,
    register_detector,
)
from .detection import (
    CLASS_NAMES,
    SsdDetector,
    YoloDetector,
    build_detector,
    filter_detections,
)
from .restoration import (
    ARCH_NAME,
    CHECKPOINT_FORMAT,
    TOGGLES,
    CALayer,
    LayerNorm2d,
    NAFBlock,
    RestorationConfig,
    RestorationNet,
    RestormerLikeBlock,
    RestormerLikeRestorer,
    SimpleGate,
    add_ablation_args,
    build_network,
    build_restorer,
    config_from_args,
    count_parameters,
    load_checkpoint,
    save_checkpoint,
)

__all__ = [
    "BaseRestorer", "BaseDetector", "Detection", "NullDetector",
    "register_detector", "detector_names", "get_detector_class",
    "RestorationConfig", "RestorationNet", "RestormerLikeRestorer", "TOGGLES",
    "LayerNorm2d", "SimpleGate", "CALayer", "NAFBlock", "RestormerLikeBlock",
    "build_network", "build_restorer", "save_checkpoint", "load_checkpoint",
    "count_parameters", "add_ablation_args", "config_from_args",
    "ARCH_NAME", "CHECKPOINT_FORMAT",
    "CLASS_NAMES", "YoloDetector", "SsdDetector", "build_detector",
    "filter_detections",
]
