"""
Remove projector light from a camera capture, then compare object detection
before and after restoration.

    from projector_distortion import build_restorer, build_detector
    from projector_distortion.data import find_samples
    from projector_distortion.pipeline import process_sample

    restorer = build_restorer("weights/restorer_restormerlike.pt")
    detector = build_detector("ssd", "weights/detector_ssdlite.pth")
    for i, s in enumerate(find_samples("data/sample_input", "data/sample_gt")):
        r = process_sample(s, restorer, detector, frame_id=i)
        print(s.name_id, len(r.det_distorted), "->", len(r.det_restored))
"""

__version__ = "0.1.0"

from .config import PROJECT_ROOT, load_config, resolve_path
from .models import (
    CLASS_NAMES,
    BaseDetector,
    BaseRestorer,
    Detection,
    RestorationConfig,
    build_detector,
    build_restorer,
    detector_names,
    filter_detections,
)

__all__ = [
    "__version__",
    "BaseRestorer", "BaseDetector", "Detection",
    "RestorationConfig", "build_restorer", "build_detector",
    "detector_names", "filter_detections", "CLASS_NAMES",
    "load_config", "resolve_path", "PROJECT_ROOT",
]
