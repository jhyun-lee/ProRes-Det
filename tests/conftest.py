"""Shared fixtures. Tests that need a weight file skip cleanly when it is absent."""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

WEIGHTS = os.path.join(ROOT, "weights")
DATA = os.path.join(ROOT, "data")

RESTORER_W = os.path.join(WEIGHTS, "restorer_restormerlike.pt")
YOLO_W = os.path.join(WEIGHTS, "detector_yolo11s.pt")
SSD_W = os.path.join(WEIGHTS, "detector_ssdlite.pth")

SAMPLE_INPUT = os.path.join(DATA, "sample_input")
SAMPLE_GT = os.path.join(DATA, "sample_gt")


def _skip_missing(path, what):
    return pytest.mark.skipif(not os.path.exists(path),
                              reason=f"{what} not present at {path}")


needs_restorer = _skip_missing(RESTORER_W, "restoration weights")
needs_yolo = _skip_missing(YOLO_W, "yolo weights")
needs_ssd = _skip_missing(SSD_W, "ssd weights")
needs_samples = _skip_missing(SAMPLE_INPUT, "sample data")


def has_module(name):
    import importlib.util
    return importlib.util.find_spec(name) is not None


needs_ultralytics = pytest.mark.skipif(not has_module("ultralytics"),
                                       reason="ultralytics not installed")


@pytest.fixture(scope="session")
def root():
    return ROOT


@pytest.fixture
def bgr_image():
    """Deterministic 640x360 BGR frame with structure, not pure noise."""
    rng = np.random.default_rng(0)
    img = rng.integers(30, 90, (360, 640, 3), dtype=np.uint8)
    img[80:200, 120:340] = (180, 190, 200)
    img[220:320, 400:560] = (60, 120, 220)
    return img


@pytest.fixture
def pro_beam(bgr_image):
    """A (pro, beam) pair; beam is a different pattern so the 6ch input is not doubled."""
    rng = np.random.default_rng(1)
    beam = rng.integers(20, 200, (360, 640, 3), dtype=np.uint8)
    return bgr_image, beam
