"""Pipeline, data discovery, recording, metrics, and the CLI wiring."""

import csv
import json
import os
import tempfile
import time

import cv2
import numpy as np
import pytest

from conftest import RESTORER_W, SAMPLE_EVAL, SAMPLE_INPUT, SAMPLE_TEST, \
    SAMPLE_TRAIN, SSD_W, needs_restorer, needs_samples, needs_ssd, \
    needs_test_split, needs_train_split
from projector_distortion.data import (
    find_samples, light_id, load_labels, load_yolo_labels, resolve_dirs, surface_id,
)
from projector_distortion.utils.image import iou, psnr, resize, ssim
from projector_distortion.utils.recording import FRAME_KINDS, RunRecorder, parse_kinds


# --- filename convention ------------------------------------------------------

def test_ids_are_parsed_from_the_filename():
    distorted = "distorted_0409001429_0404023332_294_75.jpg"
    assert surface_id(distorted) == "0409001429"
    assert light_id(distorted) == "0404023332_294_75", "lightId may contain underscores"
    assert surface_id("surface_0409001429.jpg") == "0409001429"
    assert light_id("light_0404023332_294_75.jpg") == "0404023332_294_75"


def test_pre_rename_filenames_are_still_parsed():
    """Sessions collected before the rename must keep loading."""
    legacy = "projected_0409001429_0404023332_294_75.jpg"
    assert surface_id(legacy) == "0409001429"
    assert light_id(legacy) == "0404023332_294_75"
    assert surface_id("Ori0409001429.jpg") == "0409001429"
    assert light_id("output_video_0404023332_294_75.jpg") == "0404023332_294_75"


def test_unrecognised_layout_is_reported():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(FileNotFoundError, match="no recognised data layout"):
            resolve_dirs(tmp)


# --- sample discovery ---------------------------------------------------------

@needs_samples
def test_sample_discovery_pairs_everything():
    samples = find_samples(SAMPLE_INPUT, gt_root=SAMPLE_INPUT)
    assert samples, "no pairs discovered"
    for s in samples:
        assert os.path.exists(s.distorted) and os.path.exists(s.light)
        assert light_id(s.distorted) == light_id(s.light)
        assert surface_id(s.distorted) == s.surface_id


@needs_samples
def test_ground_truth_is_attached_when_present():
    samples = find_samples(SAMPLE_INPUT, gt_root=SAMPLE_INPUT)
    with_gt = [s for s in samples if s.surface]
    assert with_gt, "no sample resolved a surface target"
    for s in with_gt:
        assert surface_id(s.surface) == s.surface_id
        if s.label:
            assert surface_id(s.label) == s.surface_id


@needs_samples
def test_limit_truncates_and_order_is_stable():
    a = find_samples(SAMPLE_INPUT, gt_root=SAMPLE_INPUT, limit=3)
    b = find_samples(SAMPLE_INPUT, gt_root=SAMPLE_INPUT, limit=3)
    assert len(a) == 3
    assert [s.name_id for s in a] == [s.name_id for s in b]


@needs_samples
def test_labels_land_inside_the_image():
    samples = find_samples(SAMPLE_INPUT, gt_root=SAMPLE_INPUT)
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
    assert load_labels("no/such.json", 100, 100) == []


# --- the three splits under data/SampleData -----------------------------------

@needs_train_split
@needs_test_split
@needs_samples
def test_every_split_resolves_a_layout():
    """
    sample_train / sample_test carry the pre-rename filenames, sample_eval the
    current ones. All three must pair without any per-split handling.
    """
    for root in (SAMPLE_TRAIN, SAMPLE_EVAL, SAMPLE_TEST):
        layout, dirs = resolve_dirs(root)
        assert layout == "flat", root
        assert os.path.isdir(dirs["surface"]), root


@needs_train_split
def test_train_split_indexes_triplets():
    from projector_distortion.data import index_triplets
    triplets = index_triplets(SAMPLE_TRAIN)
    assert len(triplets) > 900, "the training split should pair nearly all 1,000 pairs"
    for distorted, surface, light in triplets[:5]:
        assert surface_id(distorted) == surface_id(surface)
        assert light_id(distorted) == light_id(light)


@needs_train_split
def test_train_split_has_no_detection_labels():
    """Training only ever fits restoration, so the split ships no labels/ at all."""
    assert not os.path.isdir(os.path.join(SAMPLE_TRAIN, "labels"))


@needs_test_split
def test_labelme_labels_are_read_from_the_test_split():
    samples = find_samples(SAMPLE_TEST, gt_root=SAMPLE_TEST)
    labelled = [s for s in samples if s.label]
    assert labelled, "no detection labels found in the test split"
    assert labelled[0].label.endswith(".json")

    boxes = load_labels(labelled[0].label, 640, 360)
    assert boxes, "LabelMe file parsed to nothing"
    for cls_id, (x1, y1, x2, y2) in boxes:
        assert 0 <= cls_id <= 16
        assert 0 <= x1 < x2 <= 640
        assert 0 <= y1 < y2 <= 360


def test_labelme_points_rescale_to_the_requested_size():
    """The annotation's own imageWidth/Height is what the points are read against."""
    doc = {"imageWidth": 640, "imageHeight": 360, "shapes": [
        {"label": "Apple", "shape_type": "rectangle",
         "points": [[320.0, 180.0], [160.0, 90.0]]},          # deliberately unordered
        {"label": "NotAClass", "shape_type": "rectangle",
         "points": [[0.0, 0.0], [10.0, 10.0]]},
        {"label": "Apple", "shape_type": "polygon", "points": [[0, 0], [1, 1], [2, 2]]},
    ]}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "surface_0409001429.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f)
        boxes = load_labels(path, 320, 180, class_names=["Apple", "BlueBerry"])

    assert boxes == [(0, (80, 45, 160, 90))], "unknown labels and polygons are skipped"


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

def test_default_kinds_keep_the_un_annotated_footage():
    """captures/ is what survives for a later re-analysis, so it must stay on."""
    assert {"distorted", "restored"} <= set(FRAME_KINDS)


def test_derived_views_are_not_written_as_separate_files():
    """They are tiles of the panel; a jpg each cost encodes per frame and bought nothing."""
    assert set(FRAME_KINDS) == {"distorted", "restored", "panel"}
    for gone in ("light", "distorted_det", "restored_det", "residual", "raw"):
        assert gone not in FRAME_KINDS


def test_parse_kinds_validates():
    assert parse_kinds("restored,panel") == ("restored", "panel")
    with pytest.raises(ValueError, match="unknown image kinds"):
        parse_kinds("restored,nope")


def _fake_result(frame_id=0, n_boxes=1):
    from projector_distortion.models import Detection
    from projector_distortion.pipeline.offline import FrameResult
    img = np.full((180, 320, 3), 64, np.uint8)
    dets = [Detection(0, "Apple", 0.9, (10, 10, 80, 80))] * n_boxes
    return FrameResult(frame_id=frame_id, name_id=f"f{frame_id:03d}", light=img.copy(),
                       distorted=img.copy(), distorted_det=img.copy(),
                       restored=img.copy(), restored_det=img.copy(),
                       residual=img.copy(), residual_mean=0.12,
                       det_distorted=dets, det_restored=dets,
                       t_restore=0.01, t_detect=0.02)


def test_recorder_honours_save_every_but_logs_every_frame():
    with tempfile.TemporaryDirectory() as tmp:
        with RunRecorder(tmp, save_every=3) as rec:
            for i in range(9):
                r = _fake_result(i)
                if rec.should_save(i):
                    rec.save_frame_images(r)
                rec.log_detections(r)
            rec.finish(note="test")

        with open(os.path.join(tmp, "detections.csv"), encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        logged = sorted({int(r["frame_id"]) for r in rows})
        assert logged == list(range(9)), "csv must cover every frame, not just saved ones"

        saved_ids = sorted({int(f.split("_")[0][1:])
                            for f in os.listdir(os.path.join(tmp, "captures"))})
        assert saved_ids == [0, 3, 6]


def test_images_are_split_across_two_directories():
    img = np.full((180, 320, 3), 64, np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        with RunRecorder(tmp, save_every=1) as rec:
            rec.save_frame_images(_fake_result(0), panel=img)

        assert sorted(os.listdir(os.path.join(tmp, "captures"))) == [
            "f000_distorted.jpg", "f000_restored.jpg"], "un-annotated footage"
        assert os.listdir(os.path.join(tmp, "frames_all")) == ["f000_panel.jpg"]
        assert not os.path.isdir(os.path.join(tmp, "frames")), "no derived-view jpgs"


def test_demo_does_not_write_a_metrics_csv():
    """PSNR/SSIM need ground truth; scoring belongs to evaluate.py, not demo.py."""
    with tempfile.TemporaryDirectory() as tmp:
        with RunRecorder(tmp, save_every=0) as rec:
            rec.log_detections(_fake_result(0))
            rec.finish()
        assert not os.path.exists(os.path.join(tmp, "frames.csv"))
        assert os.path.exists(os.path.join(tmp, "detections.csv"))


def test_live_analyses_an_evenly_spaced_subset():
    """
    The projector must not wait for the model, and the frames it does analyse must be
    evenly spaced - an irregular subset is what makes the recorded panel stutter.
    """
    from projector_distortion.pipeline import live

    clip_fps, stride, frames = 30, 3, 60
    submitted = []
    in_flight = 0

    # the loop's own rule, lifted out of the hardware it normally needs
    for frame_count in range(frames):
        if frame_count % stride == 0 and in_flight < live.MAX_IN_FLIGHT:
            submitted.append(frame_count)
            in_flight += 1
        if in_flight:                       # the worker finishes one per iteration
            in_flight -= 1

    gaps = {b - a for a, b in zip(submitted, submitted[1:])}
    assert gaps == {stride}, f"uneven spacing: {sorted(gaps)}"
    assert len(submitted) == frames // stride
    assert live.MAX_IN_FLIGHT >= 2, "1 would make the worker handoff synchronous again"


def _write_triplet(root, sub, name, img):
    d = os.path.join(root, sub)
    os.makedirs(d, exist_ok=True)
    cv2.imwrite(os.path.join(d, name), img)


def test_training_set_can_span_several_directories(bgr_image):
    """Real captures are date-partitioned and the light frames sit apart from them."""
    from projector_distortion.data import index_triplets

    img = cv2.resize(bgr_image, (64, 36))
    with tempfile.TemporaryDirectory() as tmp:
        for date, sid, light in (("0520", "0407005538", "0404023034_1008_142"),
                                 ("0529", "0529131031", "0404023034_1032_invert")):
            _write_triplet(tmp, f"Warp_{date}_distorted",
                           f"distorted_{sid}_{light}.jpg", img)
            _write_triplet(tmp, f"Warp_{date}_surface", f"surface_{sid}.jpg", img)
            _write_triplet(tmp, "lights", f"light_{light}.jpg", img)

        triplets = index_triplets(distorted=os.path.join(tmp, "Warp_*_distorted"),
                                  surface=os.path.join(tmp, "Warp_*_surface"),
                                  light=os.path.join(tmp, "lights"))
        assert len(triplets) == 2, triplets
        for distorted, surface, light in triplets:
            assert surface_id(distorted) == surface_id(surface)
            assert light_id(distorted) == light_id(light)


def test_distorted_without_a_surface_is_skipped_not_fatal(bgr_image):
    from projector_distortion.data import index_triplets

    img = cv2.resize(bgr_image, (64, 36))
    with tempfile.TemporaryDirectory() as tmp:
        _write_triplet(tmp, "distorted", "distorted_0409001429_aaa_1.jpg", img)
        # no surface
        _write_triplet(tmp, "distorted", "distorted_9999999999_aaa_1.jpg", img)
        _write_triplet(tmp, "surface", "surface_0409001429.jpg", img)
        _write_triplet(tmp, "lights", "light_aaa_1.jpg", img)

        triplets = index_triplets(distorted=os.path.join(tmp, "distorted"),
                                  surface=os.path.join(tmp, "surface"),
                                  light=os.path.join(tmp, "lights"))
        assert len(triplets) == 1
        assert surface_id(triplets[0][0]) == "0409001429"


def test_eval_dir_is_named_after_the_input_dataset():
    import evaluate
    assert evaluate._eval_dir_name("data/SampleData/sample_eval") == "Eval_sample_eval"
    assert evaluate._eval_dir_name("data/live_20260803_161234/") == \
        "Eval_live_20260803_161234"


def test_recorder_writes_one_detection_row_per_box():
    with tempfile.TemporaryDirectory() as tmp:
        with RunRecorder(tmp, save_every=0) as rec:
            rec.log_detections(_fake_result(0, n_boxes=2))
            rec.finish()
        with open(os.path.join(tmp, "detections.csv"), encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 4          # 2 boxes x distorted + restored
        assert {r["source"] for r in rows} == {"distorted", "restored"}


def test_save_every_zero_writes_no_images():
    with tempfile.TemporaryDirectory() as tmp:
        with RunRecorder(tmp, save_every=0) as rec:
            r = _fake_result(0)
            assert rec.should_save(0) is False
            rec.log_detections(r)
            meta = rec.finish()
        assert meta["saved_frames"] == 0
        assert not os.path.isdir(os.path.join(tmp, "frames"))


def test_writer_thread_drains_its_queue_before_stopping():
    """
    The live worker hands finished frames to `_writer` and moves on.

    Anything still queued when the run ends is the only copy there is, so the
    sentinel has to be processed after the backlog, not instead of it.
    """
    import queue
    import threading

    from projector_distortion.pipeline.live import _writer

    img = np.full((180, 320, 3), 64, np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        with RunRecorder(tmp, save_every=1) as rec:
            q, errors = queue.Queue(), []
            thread = threading.Thread(target=_writer, args=(q, rec, errors), daemon=True)
            thread.start()
            for i in range(5):
                q.put((_fake_result(i), img))
            q.put(None)
            thread.join(timeout=10)
            assert not thread.is_alive(), "writer ignored the sentinel"

        assert errors == []
        assert len(os.listdir(os.path.join(tmp, "captures"))) == 10   # 5 x 2 kinds
        assert len(os.listdir(os.path.join(tmp, "frames_all"))) == 5
        with open(os.path.join(tmp, "detections.csv"), encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert sorted({int(r["frame_id"]) for r in rows}) == [0, 1, 2, 3, 4]


def test_save_every_thins_what_lands_on_disk():
    with tempfile.TemporaryDirectory() as tmp:
        with RunRecorder(tmp, save_every=3) as rec:
            for i in range(9):
                if rec.should_save(i):
                    rec.save_frame_images(_fake_result(i))
            assert rec.saved_frames == 3, "frames 0, 3 and 6"


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
    """The regression this framework exists to prevent: un-annotated copies must survive."""
    from projector_distortion.models import build_detector
    from projector_distortion.models.restoration import NAFSEUNetRestorer
    from projector_distortion.pipeline import run_offline

    restorer = NAFSEUNetRestorer(RESTORER_W, device="cpu", input_size=(320, 180))
    detector = build_detector("ssd", weights=SSD_W, conf=0.05, device="cpu")

    with tempfile.TemporaryDirectory() as tmp:
        # every kind, not the default subset: this test guards the un-annotated copies
        with RunRecorder(tmp, save_every=1, frame_kinds=FRAME_KINDS) as rec:
            summary = run_offline(SAMPLE_INPUT, restorer, detector, rec,
                                  limit=2, progress=False)
            rec.finish(**summary)

        assert summary["images"] == 2
        assert "quality" not in summary, "demo.py must not score restoration quality"

        # name_id contains underscores, so derive the stem from the known kinds
        # rather than splitting on '_'.
        stems = set()
        for f in os.listdir(os.path.join(tmp, "captures")):
            base = os.path.splitext(f)[0]
            for kind in sorted(FRAME_KINDS, key=len, reverse=True):
                if base.endswith("_" + kind):
                    stems.add(base[: -len(kind) - 1])
                    break
        assert len(stems) == 2, stems
        for stem in stems:
            for kind in ("distorted", "restored"):
                assert os.path.exists(
                    os.path.join(tmp, "captures", f"{stem}_{kind}.jpg")), kind
            assert os.path.exists(os.path.join(tmp, "frames_all", f"{stem}_panel.jpg"))
        assert not os.path.isdir(os.path.join(tmp, "frames"))

        meta = json.load(open(os.path.join(tmp, "run_meta.json"), encoding="utf-8"))
        assert meta["images"] == 2


@needs_restorer
@needs_ssd
@needs_samples
def test_annotated_image_differs_once_boxes_exist():
    from projector_distortion.models import build_detector
    from projector_distortion.models.restoration import NAFSEUNetRestorer
    from projector_distortion.pipeline import process_sample

    restorer = NAFSEUNetRestorer(RESTORER_W, device="cpu", input_size=(320, 180))
    detector = build_detector("ssd", weights=SSD_W, conf=0.01, device="cpu")
    samples = find_samples(SAMPLE_INPUT, gt_root=SAMPLE_INPUT)

    for s in samples[:5]:
        r = process_sample(s, restorer, detector, min_area=1)
        if r.det_restored:
            assert not np.array_equal(r.restored, r.restored_det)
            assert not np.array_equal(r.distorted, r.distorted_det) or not r.det_distorted
            return
    pytest.skip("the detector found nothing on the sample set at conf 0.01")


# --- CLI ----------------------------------------------------------------------

def test_demo_parser_defaults_are_coherent():
    import demo
    args = demo.build_parser().parse_args([])
    assert args.save_every == 1 and args.live is False
    assert parse_kinds(args.save_kinds) == FRAME_KINDS


def test_cuda_requirements_pin_matched_torch_torchvision_pairs():
    """
    `pip install -e ".[all]" -r requirements-cuda.txt` has to stay one working command.

    The pins are load-bearing: PyPI usually carries a newer torch than the CUDA index
    does, so an unpinned resolve walks straight back to the CPU wheel. A torch without
    its matching torchvision is the other way this file silently rots.
    """
    from conftest import ROOT

    pairs = {"2.8": "0.23", "2.9": "0.24", "2.10": "0.25", "2.11": "0.26"}
    path = os.path.join(ROOT, "requirements-cuda.txt")
    lines = [ln.strip() for ln in open(path, encoding="utf-8")
             if ln.strip() and not ln.lstrip().startswith("#")]

    assert any(ln.startswith("--extra-index-url") and "download.pytorch.org" in ln
               for ln in lines), "the CUDA index has to be named"

    pinned = {}
    for line in (ln for ln in lines if not ln.startswith("-")):
        req, _, marker = line.partition(";")
        name, _, version = req.strip().partition("==")
        assert "+cu" in version, f"{name} is not pinned to a CUDA build: {line}"
        pinned.setdefault(marker.strip(), {})[name] = version.split("+")[0]

    assert pinned, "no pins at all"
    for marker, packages in pinned.items():
        assert set(packages) == {"torch", "torchvision"}, f"{marker}: {packages}"
        major_minor = ".".join(packages["torch"].split(".")[:2])
        assert packages["torchvision"].startswith(pairs[major_minor]), \
            f"{marker}: torch {packages['torch']} does not pair with " \
            f"torchvision {packages['torchvision']}"


def test_cpu_only_torch_build_says_how_to_fix_itself(monkeypatch):
    """
    The failure mode this guards: `pip install torch` on Windows is CPU-only, nothing
    errors, and the run is 25x slower with no hint as to why.
    """
    import torch

    from projector_distortion.cli import device_note

    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    note = device_note("cpu")
    assert "CPU-only" in note
    assert "download.pytorch.org" in note, "the note has to be actionable"


def test_device_note_names_the_gpu_when_there_is_one():
    import torch

    from projector_distortion.cli import device_note, resolve_device

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device on this machine")
    assert resolve_device() == "cuda"
    assert torch.cuda.get_device_name(0) in device_note("cuda")


def test_evaluate_parser_accepts_multiple_backends():
    import evaluate
    args = evaluate.build_parser().parse_args(["--detector", "yolo,ssd", "--iou", "0.75"])
    assert args.detector == "yolo,ssd" and args.iou == 0.75


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


def test_live_worker_consumes_the_frame_queue_tuple():
    """
    The live worker's queue payload, with stubs standing in for the rig.

    It is (frame_id, distorted, light) - the raw camera frame used to ride along
    unused, pinning a full-resolution image per queue slot. Nothing else exercises
    live.py without a projector, so the contract is checked here.
    """
    import queue
    import threading

    from projector_distortion.models.base import Detection
    from projector_distortion.pipeline.live import _worker

    class StubRestorer:
        input_size = (640, 360)

        def restore_full(self, distorted, light):
            blank = np.zeros((360, 640, 3), np.uint8)
            return blank, blank.copy(), 0.25

    class StubDetector:
        name = "stub"

        def __call__(self, bgr):
            return [Detection(0, "Apple", 0.9, (10, 10, 200, 200))]

    frame_queue, result_queue, write_queue = queue.Queue(3), queue.Queue(5), queue.Queue(8)
    stop = threading.Event()
    worker = threading.Thread(
        target=_worker, daemon=True,
        args=(frame_queue, result_queue, write_queue, StubRestorer(), StubDetector(),
              stop))
    worker.start()
    try:
        frame_queue.put((7, np.full((720, 1280, 3), 40, np.uint8),
                         np.full((360, 640, 3), 90, np.uint8)))
        result, panel = result_queue.get(timeout=15)
    finally:
        stop.set()
        worker.join(timeout=3)

    assert result.frame_id == 7 and result.name_id == "frame_000007"
    assert len(result.det_distorted) == 1 and len(result.det_restored) == 1
    assert result.residual_mean == 0.25
    assert panel.ndim == 3 and panel.size > 0
    assert not write_queue.empty(), "the writer thread has to be handed the frame"


def test_live_worker_reports_a_result_it_could_not_hand_back():
    """
    A full result queue must not lose the frame silently.

    The main loop counts handed-over frames in `in_flight` and only ever decrements
    on receipt, so a dropped result left the count permanently high: the final drain
    then waited out its whole 10s timeout and blamed the straggler on a missed
    deadline. The worker reports the id instead.
    """
    import queue
    import threading

    from projector_distortion.pipeline.live import _worker

    class StubRestorer:
        input_size = (640, 360)

        def restore_full(self, distorted, light):
            blank = np.zeros((360, 640, 3), np.uint8)
            return blank, blank.copy(), 0.0

    frame_queue = queue.Queue(3)
    result_queue = queue.Queue(1)
    result_queue.put(("already", "full"))          # nobody is draining
    lost = []
    stop = threading.Event()
    worker = threading.Thread(
        target=_worker, daemon=True,
        args=(frame_queue, result_queue, queue.Queue(8), StubRestorer(),
              lambda bgr: [], stop, lost))
    worker.start()
    try:
        frame_queue.put((11, np.zeros((360, 640, 3), np.uint8),
                         np.zeros((360, 640, 3), np.uint8)))
        deadline = time.time() + 20
        while not lost and time.time() < deadline:
            time.sleep(0.05)
    finally:
        stop.set()
        worker.join(timeout=3)

    assert lost == [11], f"the dropped frame id should be reported, got {lost}"


@needs_restorer
def test_an_unknown_backend_is_named_as_such_not_blamed_on_its_weights():
    """
    `--detector` is validated against the registry, and before the weights lookup.

    An unrecognised backend has no `weights:` entry either, so checking the
    checkpoint first reported a missing file for a merely misspelled name.
    """
    import demo
    from projector_distortion.cli import build_models

    args = demo.build_parser().parse_args(["--detector", "bogus"])
    with pytest.raises(SystemExit) as e:
        build_models(args, need_detector=True)
    message = str(e.value)
    assert "unknown detector 'bogus'" in message
    assert "yolo" in message, "the message has to list what is available"
