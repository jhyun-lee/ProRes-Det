"""
Swappable model layer: restoration + detection behind two small interfaces.

What is re-exported here is the layer's contract - the base classes, the detector
registry, the builders and the checkpoint helpers. The restoration network's own
building blocks (LayerNorm2d, SimpleGate, CALayer, NAFBlock, NAFSEBlock,
RestorationNet) are one architecture's internals, so they stay in
`projector_distortion.models.restoration` and are imported from there when needed.
"""

from .base import (
    BaseDetector,
    BaseRestorer,
    Detection,
    NullDetector,
    detector_names,
    get_detector_class,
    get_restorer_class,
    register_detector,
    register_restorer,
    restorer_names,
)
from .detection import (
    CLASS_NAMES,
    SsdDetector,
    YoloDetector,
    build_detector,
    filter_detections,
)
from .restoration import (
    TOGGLES,
    NAFSEUNetRestorer,
    RestorationConfig,
    build_network,
    build_restorer,
    count_parameters,
    load_checkpoint,
    save_checkpoint,
)

__all__ = [
    "BaseRestorer", "BaseDetector", "Detection", "NullDetector",
    "register_detector", "detector_names", "get_detector_class",
    "register_restorer", "restorer_names", "get_restorer_class",
    "RestorationConfig", "NAFSEUNetRestorer", "TOGGLES",
    "build_network", "build_restorer", "save_checkpoint", "load_checkpoint",
    "count_parameters",
    "CLASS_NAMES", "YoloDetector", "SsdDetector", "build_detector",
    "filter_detections",
]
