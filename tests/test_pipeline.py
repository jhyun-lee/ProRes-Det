"""Pipeline, data discovery, recording, metrics, and the CLI wiring."""

import csv
import json
import os
import tempfile

import numpy as np
import pytest

from conftest import RESTORER_W, SAMPLE_GT, SAMPLE_INPUT, SSD_W, needs_restorer, \
    needs_samples, needs_ssd
from projector_distortion.data import (
    beam_id, find_samples, load_yolo_labels, ori_id, resolve_dirs,
)
from projector_distortion.utils.image import iou, psnr, resize, ssim
from projector_distortion.utils.recording import (
    DEFAULT_FRAME_KINDS, FRAME_KINDS, RunRecorder, estimate_footprint_mb, parse_kinds,
)


# --- filename convention ------------------------------------------------------

def test_ids_are_parsed_from_the_filename():
    pro = "projected_0409001429_0404023332_294_75.jpg"
    assert ori_id(pro) == "0409001429"
    assert beam_id(pro) == "0404023332_294_75", "beamId may contain underscores"
    assert ori_id("Ori0409001429.jpg") == "0409001429"
    assert beam_id("output_video_0404023332_294_75.jpg") == "0404023332_294_75"


def test_unrecognised_layout_is_reported():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(FileNotFoundError, match="no recognised data layout"):
            resolve_dirs(tmp)


# --- sample discovery ---------------------------------------------------------

@needs_samples
def test_sample_discovery_pairs_everything():
    samples = find_samples(SAMPLE_INPUT, gt_root=SAMPLE_GT)
    assert samples, "no pairs discovered"
    for s in samples:
        assert os.path.exists(s.pro) and os.path.exists(s.beam)
        assert beam_id(s.pro) == beam_id(s.beam)
        assert ori_id(s.pro) == s.ori_id


@needs_samples
def test_ground_truth_is_attached_when_present():
    samples = find_samples(SAMPLE_INPUT, gt_root=SAMPLE_GT)
    with_gt = [s for s in samples if s.clean]
    assert with_gt, "no sample resolved a clean target"
    for s in with_gt:
        assert ori_id(s.clean) == s.ori_id
        if s.label:
            assert ori_id(s.label) == s.ori_id


@needs_samples
def test_limit_truncates_and_order_is_stable():
    a = find_samples(SAMPLE_INPUT, gt_root=SAMPLE_GT, limit=3)
    b = find_samples(SAMPLE_INPUT, gt_root=SAMPLE_GT, limit=3)
    assert len(a) == 3
    assert [s.name_id for s in a] == [s.name_id for s in b]


@needs_samples
def test_labels_land_inside_the_image():
    samples = find_samples(SAMPLE_INPUT, gt_root=SAMPLE_GT)
    labelled = [s for s in samples if s.label]
    assert labelled, "no detection labels found"
    boxes = load_yolo_labels(labelled[0].label, 640, 360)
    assert boxes, "label file parsed to nothing"
    for cls_id, (x1, y1, x2, y2) in boxes:
        assert 0 <= cls_id <= 16
        assert 0 <= x1 < x2 <= 640
        assert 0 <= y1 < y2 <= 360


def test_missing_label_file_is_not_an_error():
    assert load_yolo_labels("no/such.txt", 100, 100) == []


# --- metrics ------------------------------------------------------------------

def test_psnr_and_ssim_are_perfect_on_identical_images(bgr_image):
    assert psnr(bgr_image, bgr_image) == float("inf")
    assert ssim(bgr_image, bgr_image) == pytest.approx(1.0, abs=1e-6)


def test_psnr_drops_as_noise_grows(bgr_image):
    rng = np.random.default_rng(2)
    a = np.clip(bgr_image.astype(int) + rng.integers(-5, 5, bgr_image.shape), 0, 255
                ).astype(np.uint8)
    b = np.clip(bgr_image.astype(int) + rng.integers(-60, 60, bgr_image.shape), 0, 255
                ).astype(np.uint8)
    assert psnr(bgr_image, a) > psnr(bgr_image, b)
    assert ssim(bgr_image, a) > ssim(bgr_image, b)


def test_metrics_resize_mismatched_inputs(bgr_image):
    half = resize(bgr_image, (320, 180))
    assert psnr(bgr_image, half) > 0        # must not raise


def test_iou_edges():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(5 / 15)


# --- recording ----------------------------------------------------------------

def test_all_kinds_are_saved_by_default():
    assert DEFAULT_FRAME_KINDS == FRAME_KINDS
    assert "raw" in DEFAULT_FRAME_KINDS


def test_parse_kinds_validates():
    assert parse_kinds("restored,panel") == ("restored", "panel")
    with pytest.raises(ValueError, match="unknown image kinds"):
        parse_kinds("restored,nope")


def test_footprint_estimate_scales_with_interval():
    every1 = estimate_footprint_mb(FRAME_KINDS, 1, 1000)
    every10 = estimate_footprint_mb(FRAME_KINDS, 10, 1000)
    assert every1 == pytest.approx(every10 * 10, rel=0.01)
    assert estimate_footprint_mb(FRAME_KINDS, 0, 1000) == 0.0


def _fake_result(frame_id=0, n_boxes=1):
    from projector_distortion.models import Detection
    from projector_distortion.pipeline.offline import FrameResult
    img = np.full((180, 320, 3), 64, np.uint8)
    dets = [Detection(0, "Apple", 0.9, (10, 10, 80, 80))] * n_boxes
    return FrameResult(frame_id=frame_id, name_id=f"f{frame_id:03d}", beam=img.copy(),
                       captured=img.copy(), captured_det=img.copy(),
                       restored=img.copy(), restored_det=img.copy(),
                       residual=img.copy(), residual_mean=0.12,
                       det_captured=dets, det_restored=dets,
                       t_restore=0.01, t_detect=0.02)


def test_recorder_honours_save_every_and_logs_every_frame():
    with tempfile.TemporaryDirectory() as tmp:
        with RunRecorder(tmp, save_every=3) as rec:
            for i in range(9):
                r = _fake_result(i)
                saved = rec.should_save(i)
                if saved:
                    rec.save_frame_images(r)
                rec.log_frame(r, saved=saved)
            rec.finish(note="test")

        with open(os.path.join(tmp, "frames.csv"), encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 9, "csv must cover every frame, not just the saved ones"
        assert sum(int(r["saved"]) for r in rows) == 3

        saved_ids = sorted({int(f.split("_")[0][1:])
                            for f in os.listdir(os.path.join(tmp, "frames"))})
        assert saved_ids == [0, 3, 6]


def test_recorder_writes_one_detection_row_per_box():
    with tempfile.TemporaryDirectory() as tmp:
        with RunRecorder(tmp, save_every=0) as rec:
            rec.log_frame(_fake_result(0, n_boxes=2), saved=False)
            rec.finish()
        with open(os.path.join(tmp, "detections.csv"), encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 4          # 2 boxes x captured + restored
        assert {r["source"] for r in rows} == {"captured", "restored"}


def test_save_every_zero_writes_no_images():
    with tempfile.TemporaryDirectory() as tmp:
        with RunRecorder(tmp, save_every=0) as rec:
            r = _fake_result(0)
            assert rec.should_save(0) is False
            rec.log_frame(r, saved=False)
            meta = rec.finish()
        assert meta["saved_frames"] == 0
        assert not os.path.isdir(os.path.join(tmp, "frames"))


def test_max_saved_frames_caps_output():
    with tempfile.TemporaryDirectory() as tmp:
        with RunRecorder(tmp, save_every=1, max_saved_frames=2) as rec:
            for i in range(6):
                if rec.should_save(i):
                    rec.save_frame_images(_fake_result(i))
            assert rec.saved_frames == 2


def test_run_meta_is_valid_json():
    with tempfile.TemporaryDirectory() as tmp:
        with RunRecorder(tmp, save_every=0) as rec:
            rec.set(mode="offline", nested={"a": [1, 2]})
            rec.finish(images=3)
        meta = json.load(open(os.path.join(tmp, "run_meta.json"), encoding="utf-8"))
        assert meta["mode"] == "offline" and meta["images"] == 3
        assert "started_at" in meta and "finished_at" in meta


# --- end to end ---------------------------------------------------------------

@needs_restorer
@needs_ssd
@needs_samples
def test_offline_run_produces_clean_and_annotated_images():
    """The regression this framework exists to prevent: clean copies must survive."""
    from projector_distortion.models import build_detector
    from projector_distortion.models.restoration import RestormerLikeRestorer
    from projector_distortion.pipeline import run_offline

    restorer = RestormerLikeRestorer(RESTORER_W, device="cpu", input_size=(320, 180))
    detector = build_detector("ssd", weights=SSD_W, conf=0.05, device="cpu")

    with tempfile.TemporaryDirectory() as tmp:
        with RunRecorder(tmp, save_every=1) as rec:
            summary = run_offline(SAMPLE_INPUT, restorer, detector, rec,
                                  gt_root=SAMPLE_GT, limit=2, progress=False)
            rec.finish(**summary)

        assert summary["images"] == 2
        assert summary["with_ground_truth"] == 2
        q = summary["quality"]
        for key in ("psnr_captured", "psnr_restored", "ssim_captured", "ssim_restored"):
            assert np.isfinite(q[key]), key

        # name_id contains underscores, so derive the stem from the known kinds
        # rather than splitting on '_'.
        frames_dir = os.path.join(tmp, "frames")
        files = os.listdir(frames_dir)
        stems = set()
        for f in files:
            base = os.path.splitext(f)[0]
            for kind in sorted(FRAME_KINDS, key=len, reverse=True):
                if base.endswith("_" + kind):
                    stems.add(base[: -len(kind) - 1])
                    break
        assert len(stems) == 2, stems
        for stem in stems:
            for kind in ("restored", "restored_det", "captured", "captured_det",
                         "beam", "residual", "panel"):
                assert os.path.exists(os.path.join(frames_dir, f"{stem}_{kind}.jpg")), \
                    f"{stem}_{kind}.jpg missing"

        meta = json.load(open(os.path.join(tmp, "run_meta.json"), encoding="utf-8"))
        assert meta["images"] == 2


@needs_restorer
@needs_ssd
@needs_samples
def test_annotated_image_differs_once_boxes_exist():
    from projector_distortion.models import build_detector
    from projector_distortion.models.restoration import RestormerLikeRestorer
    from projector_distortion.pipeline import process_sample

    restorer = RestormerLikeRestorer(RESTORER_W, device="cpu", input_size=(320, 180))
    detector = build_detector("ssd", weights=SSD_W, conf=0.01, device="cpu")
    samples = find_samples(SAMPLE_INPUT, gt_root=SAMPLE_GT)

    for s in samples[:5]:
        r = process_sample(s, restorer, detector, min_area=1)
        if r.det_restored:
            assert not np.array_equal(r.restored, r.restored_det)
            assert not np.array_equal(r.captured, r.captured_det) or not r.det_captured
            return
    pytest.skip("the detector found nothing on the sample set at conf 0.01")


# --- CLI ----------------------------------------------------------------------

def test_demo_parser_defaults_are_coherent():
    import demo
    args = demo.build_parser().parse_args([])
    assert args.save_every == 1 and args.live is False
    assert parse_kinds(args.save_kinds) == FRAME_KINDS


def test_evaluate_parser_accepts_multiple_backends():
    import evaluate
    args = evaluate.build_parser().parse_args(["--detectors", "yolo,ssd", "--iou", "0.75"])
    assert args.detectors == "yolo,ssd" and args.iou == 0.75


def test_demo_rejects_an_unknown_save_kind():
    import demo
    args = demo.build_parser().parse_args(["--save-kinds", "restored,bogus"])
    with pytest.raises(ValueError):
        parse_kinds(args.save_kinds)


def test_average_precision_is_sane():
    from evaluate import average_precision
    assert average_precision([1, 1, 1], [0.9, 0.8, 0.7], 3) == pytest.approx(1.0)
    assert average_precision([0, 0], [0.9, 0.8], 2) == pytest.approx(0.0)
    partial = average_precision([1, 0], [0.9, 0.8], 2)
    assert 0.0 < partial < 1.0


def test_average_precision_with_no_ground_truth_is_nan():
    from evaluate import average_precision
    assert np.isnan(average_precision([1], [0.9], 0))
