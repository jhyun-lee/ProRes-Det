"""
The two contracts that make restoration and detection swappable.

    Restorer.restore(pro_bgr, beam_bgr) -> (restored_bgr, residual_bgr)
    Detector.detect(bgr)                -> list[Detection]

`pro` is the camera's view of the projected screen, `beam` is the frame the
projector emitted at that moment. Restorers predict the *residual* to subtract,
not the clean image directly.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Detection:
    """One box; `box` is (x1, y1, x2, y2) in pixels of the image passed to detect()."""

    cls_id: int
    name: str
    conf: float
    box: Sequence[int]

    def as_row(self) -> dict:
        x1, y1, x2, y2 = (int(v) for v in self.box)
        return {"cls_id": self.cls_id, "name": self.name, "conf": round(self.conf, 4),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2}

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.box
        return max(0, int(x2) - int(x1)) * max(0, int(y2) - int(y1))


class BaseRestorer(ABC):
    """Removes projector light from a camera capture."""

    name = "base-restorer"
    input_size: Tuple[int, int] = (640, 360)

    @abstractmethod
    def restore(self, pro_bgr: np.ndarray, beam_bgr: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (restored_bgr, residual_bgr), both uint8 at `input_size`."""

    def restore_full(self, pro_bgr: np.ndarray, beam_bgr: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        restore() plus mean |residual| in the network's own [-1, 1] units.

        The residual image is colourmapped, so the scalar cannot be recovered from
        it. The default returns 0.0 rather than a wrong number; subclasses that
        know the value should override.
        """
        restored, residual = self.restore(pro_bgr, beam_bgr)
        return restored, residual, 0.0

    def __call__(self, pro_bgr, beam_bgr):
        return self.restore(pro_bgr, beam_bgr)

    def describe(self) -> str:
        return self.name

    def info(self) -> Dict:
        return {"name": self.name, "input_size": list(self.input_size)}


class BaseDetector(ABC):
    """Finds objects in an image."""

    name = "base-detector"

    def __init__(self, class_names: Sequence[str] = (), conf: float = 0.25,
                 device: str = "cpu"):
        self.class_names = list(class_names)
        self.conf = float(conf)
        self.device = device

    @abstractmethod
    def detect(self, bgr: np.ndarray) -> List[Detection]:
        """Return every detection at or above `self.conf`."""

    def __call__(self, bgr):
        return self.detect(bgr)

    def label_of(self, cls_id: int) -> str:
        if 0 <= cls_id < len(self.class_names):
            return str(self.class_names[cls_id])
        return str(cls_id)

    def describe(self) -> str:
        return f"{self.name} (conf>={self.conf}, {len(self.class_names)} classes)"

    def info(self) -> Dict:
        return {"name": self.name, "conf": self.conf, "device": self.device,
                "class_names": self.class_names}


class NullDetector(BaseDetector):
    """`--detector none`: run restoration only."""

    name = "none"

    def detect(self, bgr):
        return []

    def describe(self):
        return "none (detection disabled)"


_DETECTORS: Dict[str, type] = {}


def register_detector(key: str):
    """Decorator adding a BaseDetector subclass to the backend registry."""
    def deco(cls):
        if not issubclass(cls, BaseDetector):
            raise TypeError(f"{cls.__name__} must subclass BaseDetector")
        _DETECTORS[key] = cls
        return cls
    return deco


def detector_names() -> List[str]:
    return sorted(_DETECTORS)


def get_detector_class(key: str) -> type:
    if key not in _DETECTORS:
        raise ValueError(f"unknown detector '{key}'. available: {detector_names()}")
    return _DETECTORS[key]


register_detector("none")(NullDetector)
