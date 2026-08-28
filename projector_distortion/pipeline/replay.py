"""
Replay pipeline: two mp4s in, one or more mp4s out. No hardware.

`Data.py record` shoots the camera's view of a projected clip and writes one raw,
unrectified mp4. This is what turns that file back into a run:

    recording  rec_<t>.mp4   the camera's view          -> `distorted`
    clip       test_light.mp4 the frames it projected   -> `light`

The offline pipeline reads folders of already-rectified, id-matched stills, so a
recording had nowhere to go. Two things stand between an mp4 pair and that pipeline,
and both are handled here:

  Geometry.  The recording is raw: the screen is a trapezoid somewhere inside a
             1280x720 camera frame. The projected area is found once, in what moves
             across the whole recording, and every frame is rectified with that one
             homography - the same single-calibration model `demo.py --live` uses.

  Time.      The camera and the projector run at unrelated rates (10.03 fps against
             28.01 fps on the reference recording), so frame i of one is not frame i
             of the other. Pairing is by timestamp, not by index, and the clip is
             read forward-only because the mapping is monotonic.

The .json `record` writes beside the mp4 is what makes this a one-argument command:
it carries the measured rates *and* the clip that was projected, so the pairing does
not have to be told either. Without it the rates fall back to the file headers and
the clip has to be named with --clip.

Nothing here needs ground truth, so no PSNR/SSIM: there is no surface shot of a
scene the projector was lighting the whole time. It restores, optionally detects, and
writes the video - or several views of it, composited off the one restoration pass.
"""

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..utils.visualize import PANEL_GAP, caption, panel_size
from .offline import build_panel, process_pair

# `live` is imported inside the functions that warp, not here: it drags in the Win32
# display plumbing, and demo.py has to be able to read VIDEO_OUT_KINDS and
# output_size() just to build its parser.

# Frames sampled across the recording to find the screen in. Spread over the whole
# clip rather than taken from its start, so a passage that happens to be dark does
# not decide the geometry on its own.
CALIB_FRAMES = 48

# Smallest quad accepted as the screen, as a fraction of the camera frame. A
# projection filling less than this is a mis-aimed rig, not a screen.
CALIB_MIN_AREA_FRAC = 0.02

# Closing kernel that rejoins the slabs a dark passage splits the lit mask into,
# in pixels of the camera frame. Sized by measurement: at 31 the reference recording
# still separates into 207k + 92k px, and the larger slab alone reads as a screen
# covering the bottom half - a plausible quad that is wrong. At 51 they merge into
# one 410k px region and the corners stop moving; 51, 71 and 91 agree to the pixel.
CALIB_CLOSE_KERNEL = 51

VIDEO_OUT_KINDS = ("panel", "restored", "compare")

CALIB_WINDOW = "Replay_Calibration"

# Largest a prompt window is opened at. Under a 1920x1080 panel with room for the
# taskbar and the title bar, so the figure arrives whole on the smallest screen this
# is likely to run on rather than hanging off the bottom of it.
WINDOW_MAX_W, WINDOW_MAX_H = 1800, 950

# Per-tile width of the calibration figure. warp_before_after defaults to 640, which
# makes a 1292-wide figure; the corners are being judged against a 1280-wide camera
# frame, so the left tile is then a half-scale view of the thing under review.
REVIEW_TILE_W = 760


# --------------------------------------------------------------- the sidecar json

def load_sidecar(video_path) -> Dict:
    """
    The .json `Data.py record` writes beside the mp4, or {} when it is absent.

    Never fatal: a recording that lost its sidecar still replays off the file
    headers, it just cannot know the projector's measured rate.
    """
    path = os.path.splitext(str(video_path))[0] + ".json"
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        print(f"warning: could not read {os.path.basename(path)} ({e}); "
              f"falling back to the video headers")
        return {}


# ------------------------------------------------------------------- the pairing

@dataclass
class Alignment:
    """
    Which clip frame was on screen when camera frame `i` was taken.

        t     = i / camera_fps - offset_sec
        index = t * clip_rate

    `clip_rate` is the rate the projector actually achieved, which is not the clip's
    own fps: the reference recording asked for 30 and got 28.01. Reading it from the
    sidecar is what keeps a 30-second recording aligned at its end and not only at
    its start - a 7% rate error is 2 seconds of drift by then.

    `offset_sec` is the projector -> camera latency, in seconds. It is a property of
    the capture chain - the sensor, the USB pipeline, the driver's buffering - so it
    is a wall-clock delay, not a frame count. `rig.offset` in configs/live.yaml
    expresses the same lag in frames of the live loop, which runs at CAM_FPS; a
    recording that came off the camera at 10 fps has the identical delay, just fewer
    frames spanning it, which is why the conversion is at CAM_FPS and not at the
    recording's own rate.
    """

    camera_fps: float
    clip_rate: float
    offset_sec: float = 0.0
    source: str = "headers"

    def clip_index(self, camera_index: int) -> int:
        t = camera_index / self.camera_fps - self.offset_sec
        return max(0, int(round(t * self.clip_rate)))

    def describe(self) -> str:
        return (f"camera {self.camera_fps:.2f} fps, projector {self.clip_rate:.2f} fps "
                f"({self.source}), latency {self.offset_sec * 1000:.0f} ms")


def build_alignment(meta, cam_cap, clip_cap, offset_sec=0.0) -> Alignment:
    """Rates from the sidecar when it has them, from the file headers when it does not."""
    video = meta.get("video") or {}
    projector = meta.get("projector") or {}
    clip = meta.get("clip") or {}

    camera_fps = _positive(video.get("measured_fps")) or \
        _positive(video.get("fps_header")) or \
        _positive(cam_cap.get(cv2.CAP_PROP_FPS)) or 30.0

    clip_rate = _positive(projector.get("fps_projector"))
    source = "measured"
    if clip_rate is None:
        clip_rate = _positive(clip.get("fps")) or _positive(clip_cap.get(cv2.CAP_PROP_FPS))
        source = "clip header"
    if clip_rate is None:
        clip_rate, source = camera_fps, "assumed = camera"

    return Alignment(camera_fps=camera_fps, clip_rate=clip_rate,
                     offset_sec=float(offset_sec), source=source)


def clip_for(recording, explicit=None, fallback=None):
    """
    (path, where it came from) for the clip this recording was shot against.

    An explicit --clip wins. Otherwise the sidecar is asked, because it recorded the
    answer at capture time and nothing else knows it: the right clip is a property of
    the recording, not a default, and a wrong one produces a run that completes and
    means nothing. `fallback` is the last resort - a recording with no sidecar, or one
    naming a path that does not exist on this machine.

    Returns (None, reason) when nothing can be resolved, so the caller can say why.
    """
    if explicit:
        return explicit, "--clip"

    recorded = (load_sidecar(recording).get("clip") or {}).get("path")
    if recorded and os.path.exists(recorded):
        return recorded, "the recording's .json"
    if recorded:
        # Shot on the rig machine, replayed somewhere else. The name is still the
        # useful half of the answer, so say it rather than silently defaulting.
        print(f"warning: the .json names {os.path.basename(recorded)}, which is not "
              f"at the path it was recorded from.\n    Pass --clip to point at your "
              f"copy of it.")
    return fallback, "the default" if fallback else "nothing"


def replay_fps(recording) -> float:
    """
    The rate the camera actually recorded at, for the result.mp4 header.

    One camera frame in is one frame out, so stamping the writer with anything else
    is what makes a replay play back in fast or slow motion - the reference recording
    came off the camera at 10.03 fps, and the 30 a writer defaults to would play it
    3x fast. `Data.py record` measures the rate rather than trusting the driver for
    exactly this reason; this is the other half of that.

    Called before the run opens anything, so it reads the sidecar and falls back to
    the file header.
    """
    meta = load_sidecar(recording)
    video = meta.get("video") or {}
    fps = _positive(video.get("measured_fps")) or _positive(video.get("fps_header"))
    if fps:
        return fps

    cap = cv2.VideoCapture(str(recording))
    try:
        return _positive(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    finally:
        cap.release()


def _positive(value) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


class ClipReader:
    """
    Forward-only random access into the projected clip.

    `Alignment.clip_index` is non-decreasing, so the clip is walked with grab() to
    the wanted index rather than seeked. Seeking a long-GOP mp4 per frame decodes the
    preceding GOP every time; walking it decodes each frame once.
    """

    def __init__(self, cap):
        self.cap = cap
        self.index = -1
        self.frame = None

    def at(self, target: int) -> Optional[np.ndarray]:
        """The clip frame at `target`, or the last one once the clip has run out."""
        while self.index < target:
            ok = self.cap.grab()
            if not ok:
                return self.frame
            self.index += 1
            if self.index == target:
                ok, frame = self.cap.retrieve()
                if not ok:
                    return self.frame
                self.frame = frame
        return self.frame


# ----------------------------------------------------------------- the geometry

def calibration_frames(cap, frames=CALIB_FRAMES):
    """
    (composite, motion) over frames sampled evenly across the recording.

    `composite` is the per-pixel max - a bright, readable still to show a human.
    `motion` is the mean absolute difference between consecutive samples, and it is
    what the screen is actually found in.

    The live rig finds the screen by flashing black then white and diffing the two
    shots; a finished recording cannot be asked to flash. Brightness does not stand
    in for it - the reference rig projects onto a white board in a lit room, so the
    surround is as bright as the projection and an Otsu split on the max composite
    returns the entire camera frame. Neither does plain temporal deviation: the
    camera's auto-exposure rides the clip's brightness, so the whole frame breathes
    together and the wall's variance lands in the same range as the screen's
    (measured 233/255 of the surround above the Otsu threshold - no boundary at all).

    Frame-to-frame absolute difference does separate them, because AE drifts across
    seconds while the projected content changes between one sample and the next. On
    the reference recording that is 150/255 inside the quad against 2/255 outside.

    Accumulated one frame at a time, so the cost is a couple of planes however many
    frames are sampled.
    """
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, total // max(1, frames)) if total else 1

    composite, previous, motion_sum, index, taken = None, None, None, -1, 0
    while taken < frames:
        ok = cap.grab()
        if not ok:
            break
        index += 1
        if index % step:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if composite is None:
            composite, motion_sum = frame, np.zeros_like(gray)
        else:
            composite = np.maximum(composite, frame)
            motion_sum += np.abs(gray - previous)
        previous = gray
        taken += 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if composite is None or taken < 2:
        return composite, None

    motion = motion_sum / (taken - 1)
    peak = float(motion.max())
    if peak <= 0:
        return composite, None
    return composite, (motion / peak * 255.0).astype(np.uint8)


def detect_screen(motion, min_area_frac=CALIB_MIN_AREA_FRAC):
    """
    The projected quad in the motion map as TL/TR/BR/BL, or None.

    Closing rejoins the slabs; the convex hull of the largest one is what fills in
    the holes and edge bites the dark passages left behind - 30% of the frame lit
    where the screen covers 46%. A screen is convex, so the hull adds back exactly
    that and nothing else. Hulling every component together instead was worse: the
    surround's own 3k px specks then drag the corners out to the frame edge.
    """
    from .live import order_corners

    if motion is None:
        return None
    blurred = cv2.GaussianBlur(motion, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (CALIB_CLOSE_KERNEL, CALIB_CLOSE_KERNEL))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    h, w = mask.shape[:2]
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    if cv2.contourArea(hull) < min_area_frac * h * w:
        return None

    peri = cv2.arcLength(hull, True)
    for r in (0.01, 0.02, 0.03, 0.05):
        approx = cv2.approxPolyDP(hull, r * peri, True)
        if len(approx) == 4:
            return order_corners(approx.reshape(4, 2))
    return None


def open_window(title, image):
    """
    A resizable window actually sized to what it is about to show.

    WINDOW_NORMAL alone opens at OpenCV's own small default and ignores the image,
    which puts a 1292x430 calibration figure in a ~640 wide box - the corners are
    what this window exists to let someone judge, and at that scale they cannot be.
    AUTOSIZE fits the image but then refuses to be resized at all, and the figure is
    wider than a laptop screen. So: NORMAL, then sized here, capped to something that
    fits a display and keeping the aspect ratio.
    """
    h, w = image.shape[:2]
    scale = min(1.0, WINDOW_MAX_W / w, WINDOW_MAX_H / h)
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, max(1, int(w * scale)), max(1, int(h * scale)))
    return title


def click_corners(image, title=CALIB_WINDOW, max_width=1280):
    """
    Four clicks on the composite, in any order. 'u' undoes, 'q' aborts.

    The image is shown scaled to fit a screen; the clicks are scaled back, so the
    corners returned are always in full camera-frame coordinates.
    """
    from .live import order_corners

    scale = min(1.0, max_width / image.shape[1])
    view = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1.0 else image.copy()
    picked: List[Tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(picked) < 4:
            picked.append((int(x / scale), int(y / scale)))

    open_window(title, view)
    cv2.setMouseCallback(title, on_mouse)
    try:
        while True:
            canvas = view.copy()
            for i, (px, py) in enumerate(picked):
                p = (int(px * scale), int(py * scale))
                cv2.circle(canvas, p, 6, (0, 255, 255), -1)
                cv2.putText(canvas, str(i + 1), (p[0] + 8, p[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            caption(canvas, f"click the 4 screen corners ({len(picked)}/4)  "
                            f"u undo   q abort")
            cv2.imshow(title, canvas)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                return None
            if key == ord("u") and picked:
                picked.pop()
            if len(picked) == 4:
                return order_corners(picked)
    finally:
        cv2.destroyWindow(title)


def review_quad(composite, points, warped, title=CALIB_WINDOW):
    """
    Hold the found quad and what it rectifies to until it is accepted.

    The homography is estimated once and reused for every frame after it, so a quad
    off by a corner poisons the whole run quietly. enter/a accept, m click by hand,
    q abort.
    """
    from ..utils.visualize import warp_before_after

    figure = warp_before_after(composite, warped, points, tile_w=REVIEW_TILE_W,
                               hint="enter/a accept   m click the corners   q abort")
    open_window(title, figure)
    try:
        while True:
            cv2.imshow(title, figure)
            key = cv2.waitKey(30) & 0xFF
            if key in (13, ord("a")):
                return "accept"
            if key == ord("m"):
                return "manual"
            if key == ord("q"):
                return "quit"
    finally:
        cv2.destroyWindow(title)


def calibrate(cap, out_size, manual=False, review=True, recorder=None):
    """
    Find the screen once, and return (warp_fn, points, composite, rectified).

    `warp_fn` takes a raw camera frame and returns it rectified to `out_size`.
    """
    from .live import homography, warp, warp_size

    composite, motion = calibration_frames(cap)
    if composite is None:
        raise SystemExit("could not read any frame out of the recording")

    points = None if manual else detect_screen(motion)
    if points is None and not manual:
        print("warning: no screen boundary found in the recording; clicking it instead")
        manual = True

    while True:
        if manual or points is None:
            points = click_corners(composite)
            if points is None:
                raise SystemExit("calibration aborted")

        w, h = warp_size(points)
        matrix = homography(points, w, h)
        rectified = warp(composite, matrix, w, h, out_size)

        if not review:
            break
        verdict = review_quad(composite, points, rectified)
        if verdict == "accept":
            break
        if verdict == "quit":
            raise SystemExit("calibration aborted")
        manual, points = True, None

    if recorder is not None:
        recorder.save_calibration(points, raw_frame=composite, warped=rectified)

    return (lambda frame: warp(frame, matrix, w, h, out_size)), points, composite, rectified


# -------------------------------------------------------------------- the output

def build_compare(result, detector_name="detector") -> np.ndarray:
    """distorted | restored, side by side, captioned on the image."""
    left = result.distorted_det.copy()
    right = result.restored_det.copy()
    caption(left, "distorted")
    caption(right, "restored")
    gap = np.full((left.shape[0], PANEL_GAP, 3), 255, np.uint8)
    return np.hstack([left, gap, right])


def build_frame(result, kind, detector_name="detector") -> np.ndarray:
    """The image this run writes to result.mp4, per --video-out."""
    if kind == "restored":
        return result.restored
    if kind == "compare":
        return build_compare(result, detector_name)
    return build_panel(result, detector_name)


def normalise_kinds(video_out) -> Tuple[str, ...]:
    """
    One kind or several, de-duplicated and validated, in the order asked for.

    A run may want more than one view of the same frames - the 2x2 panel to read
    alongside the restored footage on its own - and they come off one pass, so the
    loop takes a tuple rather than being run twice.
    """
    kinds = (video_out,) if isinstance(video_out, str) else tuple(video_out or ())
    if not kinds:
        raise ValueError(f"no video_out given; expected any of "
                         f"{', '.join(VIDEO_OUT_KINDS)}")
    ordered = list(dict.fromkeys(kinds))
    unknown = [k for k in ordered if k not in VIDEO_OUT_KINDS]
    if unknown:
        raise ValueError(f"unknown video_out {', '.join(unknown)}; "
                         f"expected any of {', '.join(VIDEO_OUT_KINDS)}")
    return tuple(ordered)


def video_targets(video_out, input_size) -> Dict:
    """
    {writer name: canvas size} for RunRecorder, from the kinds a run asked for.

    A single kind keeps the unnamed writer, so it lands in result.mp4 the way every
    other mode's video does; several are named, so they can sit beside each other.
    """
    kinds = normalise_kinds(video_out)
    if len(kinds) == 1:
        return {None: output_size(kinds[0], input_size)}
    return {kind: output_size(kind, input_size) for kind in kinds}


def output_size(kind, input_size) -> Tuple[int, int]:
    """Canvas size `build_frame` produces, as (width, height). Fixed per run."""
    w, h = int(input_size[0]), int(input_size[1])
    if kind == "restored":
        return (w, h)
    if kind == "compare":
        return (w * 2 + PANEL_GAP, h)
    return panel_size(w, h)


# ---------------------------------------------------------------------- the run

def run_replay(recording, clip, restorer, detector, recorder, limit=0,
               offset_sec=None, offset_frames=0, video_out="panel", manual_calib=False,
               review_calib=True, min_area=500, min_width=20, min_height=20,
               detector_name="detector", progress=True) -> Dict:
    """
    Restore every camera frame of `recording` against the clip frame it was showing.

    One pass over the recording. The clip is walked forward alongside it, so the two
    files are each decoded exactly once however far apart their rates are.

    `offset_sec` wins when given; otherwise `offset_frames` - `rig.offset` from
    configs/live.yaml - is read as a lag in live-loop frames and converted at CAM_FPS.
    """
    kinds = normalise_kinds(video_out)

    cam = cv2.VideoCapture(str(recording))
    if not cam.isOpened():
        raise SystemExit(f"could not open the recording: {recording}")
    clip_cap = cv2.VideoCapture(str(clip))
    if not clip_cap.isOpened():
        cam.release()
        raise SystemExit(f"could not open the projected clip: {clip}")

    try:
        meta = load_sidecar(recording)
        _warn_on_clip_mismatch(meta, clip)

        from .live import CAM_FPS

        align = build_alignment(meta, cam, clip_cap)
        align.offset_sec = (float(offset_sec) if offset_sec is not None
                            else float(offset_frames) / CAM_FPS)
        print(f"alignment: {align.describe()}")

        size = tuple(restorer.input_size)
        rectify, points, _composite, _rectified = calibrate(
            cam, size, manual=manual_calib, review=review_calib, recorder=recorder)
        print(f"calibration: screen at {[tuple(p) for p in points]}")

        total = int(cam.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if limit:
            total = min(total, limit) if total else limit
        reader = ClipReader(clip_cap)

        recorder.set(alignment={"camera_fps": align.camera_fps,
                                "clip_rate": align.clip_rate,
                                "clip_rate_source": align.source,
                                "offset_sec": align.offset_sec},
                     calibration={"points_tl_tr_br_bl": [list(p) for p in points]},
                     replay={"recording": os.path.abspath(str(recording)),
                             "clip": os.path.abspath(str(clip)),
                             "video_out": list(kinds)})

        return _loop(cam, reader, align, rectify, restorer, detector, recorder,
                     total=total, limit=limit, kinds=kinds,
                     detector_name=detector_name, progress=progress,
                     gate=dict(min_area=min_area, min_width=min_width,
                               min_height=min_height))
    finally:
        cam.release()
        clip_cap.release()


def _loop(cam, reader, align, rectify, restorer, detector, recorder, total, limit,
          kinds, detector_name, progress, gate) -> Dict:
    totals = {"distorted": 0, "restored": 0}
    sums = {"residual": 0.0, "t_restore": 0.0, "t_detect": 0.0}
    processed, no_light = 0, 0
    t0 = time.time()

    bar = None
    if progress:
        try:
            from tqdm import tqdm
            bar = tqdm(total=total or None, desc="replay", unit="frame")
        except ImportError:
            pass

    index = -1
    while True:
        ok, frame = cam.read()
        if not ok:
            break
        index += 1
        if limit and processed >= limit:
            break

        light = reader.at(align.clip_index(index))
        if light is None:
            no_light += 1
            continue

        result = process_pair(rectify(frame), light, restorer, detector,
                              frame_id=index, **gate)
        # One restoration, however many views of it are being written.
        views = {kind: build_frame(result, kind, detector_name) for kind in kinds}

        if recorder.should_save(index):
            recorder.save_frame_images(
                result, panel=views.get("panel", views[kinds[0]]))
        recorder.log_detections(result)
        for kind, view in views.items():
            recorder.write_video(view, name=None if len(kinds) == 1 else kind)

        totals["distorted"] += len(result.det_distorted)
        totals["restored"] += len(result.det_restored)
        sums["residual"] += result.residual_mean
        sums["t_restore"] += result.t_restore
        sums["t_detect"] += result.t_detect
        processed += 1
        if bar is not None:
            bar.update(1)

    if bar is not None:
        bar.close()
    if no_light:
        print(f"warning: {no_light} camera frame(s) had no clip frame to pair with; "
              f"the recording outlasts the clip")

    n = max(processed, 1)
    summary = {
        "frames_read": index + 1,
        "frames_processed": processed,
        "frames_without_light": no_light,
        "elapsed_sec": round(time.time() - t0, 2),
        "detections_total": totals,
        "detections_per_frame": {k: round(v / n, 3) for k, v in totals.items()},
        "residual_mean": round(sums["residual"] / n, 5),
        "avg_restore_ms": round(sums["t_restore"] / n * 1000, 2),
        "avg_detect_ms": round(sums["t_detect"] / n * 1000, 2),
    }
    if totals["distorted"]:
        delta = (totals["restored"] - totals["distorted"]) / totals["distorted"] * 100
        summary["detection_delta_pct"] = round(delta, 1)
    return summary


def _warn_on_clip_mismatch(meta, clip):
    """
    The clip has to be the one that was projected, and only the sidecar knows which.

    Pairing a recording with a different clip produces a run that completes, scores
    nothing, and restores against light the camera never saw - the one failure here
    that looks like success.
    """
    recorded = ((meta.get("clip") or {}).get("path"))
    if not recorded:
        return
    if os.path.normcase(os.path.abspath(str(clip))) != \
            os.path.normcase(os.path.abspath(str(recorded))):
        print(f"warning: this recording was shot against "
              f"{os.path.basename(recorded)}, not {os.path.basename(str(clip))}.\n"
              f"    `light` will not be the frames the camera actually saw.")
