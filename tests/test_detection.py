"""Detection: the registry, both wrappers, label conventions, the box filter."""

import numpy as np
import pytest

from conftest import SSD_W, YOLO_W, needs_ssd, needs_ultralytics, needs_yolo
from projector_distortion.models import (
    CLASS_NAMES, BaseDetector, Detection, build_detector, filter_detections,
)
from projector_distortion.models.base import detector_names, register_detector


# --- registry -----------------------------------------------------------------

def test_registry_exposes_the_expected_backends():
    assert set(detector_names()) == {"none", "yolo", "ssd"}


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="unknown detector"):
        build_detector("nope")


def test_backends_that_need_weights_say_so():
    with pytest.raises(ValueError, match="needs a weights path"):
        build_detector("ssd")


def test_none_backend_needs_nothing_and_detects_nothing():
    det = build_detector("none")
    assert det.detect(np.zeros((10, 10, 3), np.uint8)) == []
    assert "none" in det.describe()


def test_registering_a_new_backend_needs_no_other_change():
    @register_detector("dummy-test-backend")
    class _Dummy(BaseDetector):
        name = "dummy"

        def __init__(self, weights, **kw):
            super().__init__(kw.get("class_names") or ["a"], kw.get("conf", 0.25),
                             kw.get("device", "cpu"))

        def detect(self, bgr):
            return [Detection(0, "a", 0.9, (1, 2, 30, 40))]

    try:
        assert "dummy-test-backend" in detector_names()
        det = build_detector("dummy-test-backend", weights="ignored")
        out = det.detect(np.zeros((50, 50, 3), np.uint8))
        assert len(out) == 1 and out[0].name == "a"
    finally:
        from projector_distortion.models.base import _DETECTORS
        _DETECTORS.pop("dummy-test-backend", None)


def test_registering_a_non_detector_is_refused():
    with pytest.raises(TypeError):
        register_detector("bad")(object)


# --- Detection dataclass ------------------------------------------------------

def test_detection_area_and_row():
    d = Detection(3, "Kiwi", 0.815, (10, 20, 40, 60))
    assert d.area == 30 * 40
    row = d.as_row()
    assert row["conf"] == 0.815 and row["x2"] == 40 and row["name"] == "Kiwi"


def test_label_of_falls_back_to_the_id():
    det = build_detector("none", class_names=["a", "b"])
    assert det.label_of(1) == "b"
    assert det.label_of(99) == "99"


# --- box filter ---------------------------------------------------------------

def _dets():
    return [
        Detection(0, "Apple", 0.90, (0, 0, 100, 100)),
        Detection(0, "Apple", 0.50, (10, 10, 110, 110)),
        Detection(3, "Kiwi", 0.70, (0, 0, 100, 100)),
        Detection(5, "Melon", 0.99, (0, 0, 5, 5)),        # below the size gate
        Detection(6, "Orange", 0.99, (0, 0, 100, 4)),     # too short
    ]


def test_size_gate_drops_small_and_thin_boxes():
    kept = filter_detections(_dets())
    assert len(kept) == 3
    assert {d.cls_id for d in kept} == {0, 3}


def test_filter_defaults_keep_every_box_for_metrics():
    """Metric code must not silently lose duplicates."""
    kept = filter_detections(_dets())
    assert sum(1 for d in kept if d.cls_id == 0) == 2


def test_min_area_is_configurable():
    dets = [Detection(0, "Apple", 0.9, (0, 0, 30, 30))]     # 900 px
    assert filter_detections(dets, min_area=500)
    assert not filter_detections(dets, min_area=2000)


# --- wrappers -----------------------------------------------------------------

@needs_ssd
def test_ssd_loads_and_returns_valid_detections(bgr_image):
    det = build_detector("ssd", weights=SSD_W, conf=0.05, device="cpu")
    out = det.detect(bgr_image)
    assert isinstance(out, list)
    for d in out:
        assert isinstance(d, Detection)
        assert 0 <= d.cls_id < len(det.class_names), "torchvision 1..N must map to 0..N-1"
        assert d.conf >= 0.05
        assert d.box[2] > d.box[0] and d.box[3] > d.box[1]
        assert d.name == det.class_names[d.cls_id]


@needs_ssd
def test_ssd_honours_the_confidence_floor(bgr_image):
    low = build_detector("ssd", weights=SSD_W, conf=0.01, device="cpu").detect(bgr_image)
    high = build_detector("ssd", weights=SSD_W, conf=0.9, device="cpu").detect(bgr_image)
    assert len(high) <= len(low)
    assert all(d.conf >= 0.9 for d in high)


@needs_ssd
def test_ssd_uses_the_configured_class_names():
    det = build_detector("ssd", weights=SSD_W, class_names=CLASS_NAMES, device="cpu")
    assert det.class_names == CLASS_NAMES
    assert len(det.class_names) == 17


@needs_ssd
def test_ssd_info_is_json_serialisable():
    import json
    json.dumps(build_detector("ssd", weights=SSD_W, device="cpu").info())


@needs_yolo
@needs_ultralytics
def test_yolo_loads_and_returns_valid_detections(bgr_image):
    det = build_detector("yolo", weights=YOLO_W, conf=0.05, device="cpu")
    out = det.detect(bgr_image)
    for d in out:
        assert 0 <= d.cls_id < len(det.class_names)
        assert d.conf >= 0.05
        assert d.box[2] > d.box[0] and d.box[3] > d.box[1]


@needs_yolo
@needs_ultralytics
def test_yolo_prefers_the_names_baked_into_the_checkpoint():
    """Explicit names must still win, or detection.yaml's `names:` would be ignored."""
    default = build_detector("yolo", weights=YOLO_W, device="cpu")
    assert default.class_names, "the checkpoint should carry its own names"
    forced = build_detector("yolo", weights=YOLO_W, class_names=["x", "y"], device="cpu")
    assert forced.class_names == ["x", "y"]


def test_missing_weights_raise_filenotfound():
    with pytest.raises(FileNotFoundError):
        build_detector("ssd", weights="no/such/file.pth")
