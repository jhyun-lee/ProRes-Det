"""projector_distortion/pipeline/replay.py - two mp4s in, one or more mp4s out."""

import json
import os
import tempfile

import cv2
import numpy as np
import pytest

from conftest import RESTORER_W, needs_restorer
from projector_distortion.models import build_detector
from projector_distortion.pipeline.replay import (
    VIDEO_OUT_KINDS, Alignment, ClipReader, build_alignment, build_frame,
    calibration_frames, clip_for, detect_screen, load_sidecar, normalise_kinds,
    output_size, replay_fps, run_replay, video_targets,
)
from projector_distortion.utils.recording import RunRecorder

# The reference rig, from the sidecar of data/recordings/rec_0828142253.mp4: the
# camera delivered 10 fps while the projector managed 28, which is the case index
# pairing gets wrong and timestamp pairing gets right.
CAM_FPS, CLIP_FPS = 10.03, 28.01

# Where the synthetic screen sits inside the synthetic camera frame.
QUAD = [(160, 90), (480, 90), (500, 300), (140, 300)]
CAM_SIZE = (640, 360)


# ------------------------------------------------------------------- fixtures

def _write_video(path, frames, fps=30.0, size=CAM_SIZE):
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert writer.isOpened(), "no mp4v encoder"
    for f in frames:
        writer.write(f)
    writer.release()


def _clip_frame(i):
    """A frame whose content changes every step, so the screen has motion in it."""
    rng = np.random.default_rng(i)
    img = np.full((360, 640, 3), 20, np.uint8)
    img[:] = rng.integers(0, 255, 3, dtype=np.uint8)
    img[60:300, 80:560] = rng.integers(0, 255, (1, 1, 3), dtype=np.uint8)
    return img


def _camera_frame(i, lit=True):
    """
    A lit trapezoid on a bright, static surround.

    The surround is deliberately as bright as the projection: that is the reference
    rig (a white board in a lit room), and it is what makes brightness alone useless
    for finding the screen.
    """
    frame = np.full((360, 640, 3), 200, np.uint8)
    quad = np.array(QUAD, np.int32)
    if lit:
        cv2.fillPoly(frame, [quad], [int(v) for v in _clip_frame(i)[0, 0]])
    else:
        cv2.fillPoly(frame, [quad], (200, 200, 200))
    return frame


@pytest.fixture
def rig(tmp_path):
    """A recording, the clip it was shot against, and the sidecar between them."""
    recording = str(tmp_path / "rec_test.mp4")
    clip = str(tmp_path / "clip.mp4")

    _write_video(recording, [_camera_frame(i) for i in range(30)], fps=CAM_FPS)
    _write_video(clip, [_clip_frame(i) for i in range(90)], fps=30.0)

    with open(str(tmp_path / "rec_test.json"), "w", encoding="utf-8") as f:
        json.dump({"video": {"measured_fps": CAM_FPS, "frames": 30},
                   "projector": {"fps_projector": CLIP_FPS},
                   "clip": {"path": clip, "fps": 30.0}}, f)
    return recording, clip


# -------------------------------------------------------------- the sidecar json

def test_sidecar_is_read_from_beside_the_mp4(rig):
    recording, clip = rig
    meta = load_sidecar(recording)
    assert meta["video"]["measured_fps"] == CAM_FPS
    assert meta["projector"]["fps_projector"] == CLIP_FPS


def test_a_missing_sidecar_is_not_an_error(tmp_path):
    assert load_sidecar(str(tmp_path / "nothing.mp4")) == {}


def test_replay_fps_is_the_measured_rate_not_the_header(rig):
    """
    A writer stamped 30 would play a 10 fps recording back 3x fast.

    The mp4 header itself is written by the encoder from whatever it was told, so
    the measured rate in the sidecar is the one that means anything.
    """
    recording, _ = rig
    assert replay_fps(recording) == pytest.approx(CAM_FPS, abs=0.01)


def test_the_clip_comes_from_the_recording_that_was_shot_against_it(rig):
    """
    Which clip was projected is a fact about the recording, and `record` wrote it down.

    Defaulting to test_light.mp4 instead would quietly restore a train_light session
    against light the camera never saw - a run that completes and means nothing.
    """
    recording, clip = rig
    found, source = clip_for(recording, fallback="datasets/live/test_light.mp4")
    assert found == clip
    assert "json" in source


def test_an_explicit_clip_beats_the_recording(rig):
    recording, clip = rig
    found, source = clip_for(recording, explicit="other.mp4")
    assert (found, source) == ("other.mp4", "--clip")


def test_a_recording_with_no_sidecar_falls_back(tmp_path):
    path = str(tmp_path / "bare.mp4")
    _write_video(path, [_camera_frame(i) for i in range(4)])
    found, source = clip_for(path, fallback="datasets/live/test_light.mp4")
    assert found == "datasets/live/test_light.mp4"
    assert source == "the default"


def test_a_clip_recorded_elsewhere_says_which_one_it_wants(tmp_path, capsys):
    """
    Shot on the rig, replayed on another machine: the path is gone, the name is not.

    Falling back silently would pair it with test_light.mp4 and call that normal.
    """
    path = str(tmp_path / "moved.mp4")
    _write_video(path, [_camera_frame(i) for i in range(4)])
    with open(str(tmp_path / "moved.json"), "w", encoding="utf-8") as f:
        json.dump({"clip": {"path": "/rig/only/train_light_2.mp4"}}, f)

    found, _ = clip_for(path, fallback="fallback.mp4")
    assert found == "fallback.mp4"
    assert "train_light_2.mp4" in capsys.readouterr().out


def test_replay_fps_falls_back_to_the_file_header(tmp_path):
    path = str(tmp_path / "bare.mp4")
    _write_video(path, [_camera_frame(i) for i in range(5)], fps=15.0)
    assert replay_fps(path) == pytest.approx(15.0, abs=0.5)


# ------------------------------------------------------------------- the pairing

def test_alignment_prefers_the_measured_rates(rig, tmp_path):
    recording, clip = rig
    cam, clip_cap = cv2.VideoCapture(recording), cv2.VideoCapture(clip)
    try:
        align = build_alignment(load_sidecar(recording), cam, clip_cap)
    finally:
        cam.release()
        clip_cap.release()
    assert align.camera_fps == pytest.approx(CAM_FPS)
    assert align.clip_rate == pytest.approx(CLIP_FPS)
    assert align.source == "measured"


def test_alignment_falls_back_to_headers_without_a_sidecar(rig):
    recording, clip = rig
    cam, clip_cap = cv2.VideoCapture(recording), cv2.VideoCapture(clip)
    try:
        align = build_alignment({}, cam, clip_cap)
    finally:
        cam.release()
        clip_cap.release()
    assert align.camera_fps > 0 and align.clip_rate > 0
    assert align.source == "clip header"


def test_pairing_is_by_time_not_by_index():
    """
    The whole point of the mode: camera frame i is not clip frame i.

    At 10 against 28 fps, camera frame 100 is 10 seconds in and the projector is 280
    frames along by then. Index pairing would hand it frame 100 - a third of the way
    back - and the run would restore against light the camera never saw.
    """
    align = Alignment(camera_fps=10.0, clip_rate=28.0)
    assert align.clip_index(0) == 0
    assert align.clip_index(100) == 280
    assert align.clip_index(10) == 28


def test_latency_shifts_the_pairing_backwards():
    """The camera sees a frame late, so the light it shows was emitted earlier."""
    align = Alignment(camera_fps=10.0, clip_rate=28.0, offset_sec=0.2)
    # 10 s of camera time minus 200 ms of capture lag, at 28 projected fps.
    assert align.clip_index(100) == round((10.0 - 0.2) * 28.0)


def test_pairing_never_runs_off_the_front_of_the_clip():
    """The first frames are within one latency of t=0, and there is no frame -5."""
    align = Alignment(camera_fps=10.0, clip_rate=28.0, offset_sec=0.5)
    assert align.clip_index(0) == 0
    assert align.clip_index(2) == 0


def test_clip_reader_walks_forward_and_holds_the_last_frame(rig):
    recording, clip = rig
    cap = cv2.VideoCapture(clip)
    try:
        reader = ClipReader(cap)
        first = reader.at(0)
        later = reader.at(20)
        assert first is not None and later is not None
        assert not np.array_equal(first, later), "walking the clip returned one frame"
        # Past the end the last decoded frame is held rather than None returned, so
        # a recording that outlasts its clip degrades instead of crashing.
        assert reader.at(10_000) is not None
    finally:
        cap.release()


# ------------------------------------------------------------------ the geometry

def test_screen_is_found_against_an_equally_bright_surround(rig):
    """
    The case brightness cannot solve: the wall is as bright as the projection.

    Only the quad changes between frames, so the motion map is what carries it.
    """
    recording, _ = rig
    cap = cv2.VideoCapture(recording)
    try:
        composite, motion = calibration_frames(cap)
        assert composite is not None and motion is not None
        points = detect_screen(motion)
    finally:
        cap.release()

    assert points is not None, "no screen found"
    for found, expected in zip(points, QUAD):
        assert abs(found[0] - expected[0]) <= 12
        assert abs(found[1] - expected[1]) <= 12


def test_a_still_recording_has_no_screen_to_find(tmp_path):
    """Nothing projected means nothing moves, and a quad must not be invented."""
    path = str(tmp_path / "still.mp4")
    _write_video(path, [_camera_frame(0, lit=False) for _ in range(20)])
    cap = cv2.VideoCapture(path)
    try:
        _, motion = calibration_frames(cap)
    finally:
        cap.release()
    assert detect_screen(motion) is None


def test_detect_screen_survives_an_empty_motion_map():
    assert detect_screen(None) is None
    assert detect_screen(np.zeros((360, 640), np.uint8)) is None


# -------------------------------------------------------------------- the output

def test_output_size_matches_what_build_frame_produces():
    """
    A writer opened at the wrong canvas size drops every frame silently.

    demo.py has to size the VideoWriter before the first frame exists, so the two
    are computed apart and this is what keeps them agreeing.
    """
    result = _fake_result(640, 360)
    for kind in VIDEO_OUT_KINDS:
        frame = build_frame(result, kind)
        assert (frame.shape[1], frame.shape[0]) == output_size(kind, (640, 360)), kind


def test_unknown_video_out_is_rejected_by_name(rig, tmp_path):
    recording, clip = rig
    with tempfile.TemporaryDirectory() as out:
        with RunRecorder(out, save_every=0) as rec:
            with pytest.raises(ValueError, match="mosaic"):
                run_replay(recording, clip, None, None, rec, video_out="mosaic")


def test_one_kind_keeps_the_plain_result_mp4():
    """Every other mode writes result.mp4, and a single-video replay must too."""
    assert list(video_targets("restored", (640, 360))) == [None]
    assert list(video_targets(["panel"], (640, 360))) == [None]


def test_several_kinds_are_named_so_they_can_sit_side_by_side():
    targets = video_targets(["restored", "panel"], (640, 360))
    assert set(targets) == {"restored", "panel"}
    assert targets["restored"] == output_size("restored", (640, 360))
    assert targets["panel"] == output_size("panel", (640, 360))


def test_repeating_a_kind_does_not_open_two_writers_for_it():
    assert normalise_kinds(["panel", "panel", "restored"]) == ("panel", "restored")


def test_asking_for_no_video_at_all_is_refused():
    """A replay exists to write a video; silently writing none would be worse."""
    with pytest.raises(ValueError, match="no video_out"):
        normalise_kinds([])


def test_a_prompt_window_is_sized_to_its_image(monkeypatch):
    """
    WINDOW_NORMAL opens at OpenCV's own small default, not at the image.

    The calibration figure is what the corners get judged on, so it arriving in a
    ~640 box is the difference between a usable prompt and a useless one.
    """
    from projector_distortion.pipeline import replay

    sized = {}
    monkeypatch.setattr(replay.cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(replay.cv2, "resizeWindow",
                        lambda title, w, h: sized.update(w=w, h=h))

    replay.open_window("t", np.zeros((430, 1292, 3), np.uint8))
    assert (sized["w"], sized["h"]) == (1292, 430)


def test_an_oversized_figure_is_capped_without_distorting_it(monkeypatch):
    from projector_distortion.pipeline import replay

    sized = {}
    monkeypatch.setattr(replay.cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(replay.cv2, "resizeWindow",
                        lambda title, w, h: sized.update(w=w, h=h))

    replay.open_window("t", np.zeros((2000, 4000, 3), np.uint8))
    assert sized["w"] <= replay.WINDOW_MAX_W and sized["h"] <= replay.WINDOW_MAX_H
    assert sized["w"] / sized["h"] == pytest.approx(2.0, abs=0.01)


def _fake_result(w, h):
    from projector_distortion.pipeline import FrameResult

    img = np.zeros((h, w, 3), np.uint8)
    return FrameResult(frame_id=0, name_id="f", light=img.copy(),
                       distorted=img.copy(), distorted_det=img.copy(),
                       restored=img.copy(), restored_det=img.copy(),
                       residual=img.copy())


# ----------------------------------------------------------------------- the run

@needs_restorer
def test_replay_writes_one_video_frame_per_camera_frame(rig):
    """End to end, with detection off - the restoration-only pass."""
    from projector_distortion.models import build_restorer

    recording, clip = rig
    restorer = build_restorer("naf_se_unet", RESTORER_W, device="cpu",
                              input_size=(320, 180))
    detector = build_detector("none")

    with tempfile.TemporaryDirectory() as out:
        with RunRecorder(out, save_every=0,
                         video_size=output_size("restored", (320, 180)),
                         video_fps=CAM_FPS) as rec:
            summary = run_replay(recording, clip, restorer, detector, rec,
                                 limit=6, video_out="restored", review_calib=False,
                                 detector_name="none")
        video = os.path.join(out, "result.mp4")
        assert os.path.exists(video)
        cap = cv2.VideoCapture(video)
        try:
            written = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()

    assert summary["frames_processed"] == 6
    assert written == 6
    # No ground truth exists for a replay, so it must not claim to have scored one.
    assert "psnr_restored" not in summary


@needs_restorer
def test_two_kinds_write_two_videos_from_one_pass(rig):
    """
    The frames are restored once; only the compositing is done twice.

    Both files must therefore carry the same frame count, and neither may land on
    the plain result.mp4 name that a single-video run uses.
    """
    from projector_distortion.models import build_restorer

    recording, clip = rig
    restorer = build_restorer("naf_se_unet", RESTORER_W, device="cpu",
                              input_size=(320, 180))
    with tempfile.TemporaryDirectory() as out:
        with RunRecorder(out, save_every=0,
                         video_size=video_targets(["restored", "panel"], (320, 180)),
                         video_fps=CAM_FPS) as rec:
            summary = run_replay(recording, clip, restorer, build_detector("none"),
                                 rec, limit=4, video_out=["restored", "panel"],
                                 review_calib=False, detector_name="none")
        written = sorted(f for f in os.listdir(out) if f.endswith(".mp4"))
        counts = []
        for name in written:
            cap = cv2.VideoCapture(os.path.join(out, name))
            try:
                counts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            finally:
                cap.release()

    assert written == ["result_panel.mp4", "result_restored.mp4"]
    assert counts == [4, 4]
    assert summary["frames_processed"] == 4


@needs_restorer
def test_replay_with_detection_off_reports_no_boxes(rig):
    from projector_distortion.models import build_restorer

    recording, clip = rig
    restorer = build_restorer("naf_se_unet", RESTORER_W, device="cpu",
                              input_size=(320, 180))
    with tempfile.TemporaryDirectory() as out:
        with RunRecorder(out, save_every=0) as rec:
            summary = run_replay(recording, clip, restorer, build_detector("none"),
                                 rec, limit=3, video_out="panel",
                                 review_calib=False, detector_name="none")
    assert summary["detections_total"] == {"distorted": 0, "restored": 0}
    assert "detection_delta_pct" not in summary


def test_mismatched_clip_is_called_out(rig, tmp_path, capsys):
    """
    Replaying against the wrong clip completes and means nothing.

    It is the one failure here that looks like success - every frame restores, the
    video is written, and the light was never on that screen - so it gets a warning
    rather than being left to the reader of the run_meta.
    """
    from projector_distortion.pipeline.replay import _warn_on_clip_mismatch

    recording, clip = rig
    other = str(tmp_path / "train_light_1.mp4")
    _warn_on_clip_mismatch(load_sidecar(recording), other)
    assert "not train_light_1.mp4" in capsys.readouterr().out


def test_the_right_clip_says_nothing(rig, capsys):
    from projector_distortion.pipeline.replay import _warn_on_clip_mismatch

    recording, clip = rig
    _warn_on_clip_mismatch(load_sidecar(recording), clip)
    assert capsys.readouterr().out == ""
