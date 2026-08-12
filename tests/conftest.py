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

RESTORER_W = os.path.join(WEIGHTS, "restorer_nafse_unet.pt")
YOLO_W = os.path.join(WEIGHTS, "detector_yolo11s.pt")
SSD_W = os.path.join(WEIGHTS, "detector_ssdlite.pth")

# The three splits under data/SampleData. sample_train has no labels/ (training never
# reads them); sample_eval labels in YOLO txt, sample_test in LabelMe json.
SAMPLE_DATA = os.path.join(DATA, "SampleData")
SAMPLE_TRAIN = os.path.join(SAMPLE_DATA, "sample_train")
SAMPLE_EVAL = os.path.join(SAMPLE_DATA, "sample_eval")
SAMPLE_TEST = os.path.join(SAMPLE_DATA, "sample_test")

# What demo.py / evaluate.py default to, and what the pipeline tests score against.
SAMPLE_INPUT = SAMPLE_EVAL


def _skip_missing(path, what):
    return pytest.mark.skipif(not os.path.exists(path),
                              reason=f"{what} not present at {path}")


needs_restorer = _skip_missing(RESTORER_W, "restoration weights")
needs_yolo = _skip_missing(YOLO_W, "yolo weights")
needs_ssd = _skip_missing(SSD_W, "ssd weights")
needs_samples = _skip_missing(SAMPLE_INPUT, "sample data")
needs_train_split = _skip_missing(SAMPLE_TRAIN, "training split")
needs_test_split = _skip_missing(SAMPLE_TEST, "test split")


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
def distorted_light(bgr_image):
    """
    A (distorted, light) pair.

    `light` is a different pattern so the 6ch input is not the same image doubled.
    """
    rng = np.random.default_rng(1)
    light = rng.integers(20, 200, (360, 640, 3), dtype=np.uint8)
    return bgr_image, light
