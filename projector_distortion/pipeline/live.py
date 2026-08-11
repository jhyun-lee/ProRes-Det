"""
Live pipeline: webcam + projector rig. Needs real hardware.

    projector shows BeamVideo frame N
    webcam captures it                  -> 4-point homography -> `distorted`
    the frame shown --offset frames ago -> `light`
    restore(distorted, light) -> detect both -> 2x2 panel + recorder

The first frame's warp input and output are written to calib/ and shown once, so the
rectification the whole run depends on can be checked without stopping it.

Getting the window onto the projector's display is its own problem, and one that
collect.py and record.py share; it lives in utils/display.py.
"""

import math
import queue
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

from ..models import filter_detections
from ..utils.display import Monitor, place_fullscreen
from ..utils.image import resize
from ..utils.visualize import draw_quad, grid_2x2

WINDOW = "Projector_Display"
PANEL = "Combined_View"
DEBUG_WINDOW = "PreWarp_Debug"
WARP_WINDOW = "Warp_FirstFrame"

CLICK_ORDER = "TL -> TR -> BR -> BL"

CAM_BACKENDS = {"auto": None, "any": cv2.CAP_ANY, "dshow": cv2.CAP_DSHOW,
                "msmf": cv2.CAP_MSMF, "v4l2": cv2.CAP_V4L2}

# Frames allowed to be waiting on the worker. Deep enough to ride out a slow frame,
# shallow enough that the panel never trails the projector by much.
MAX_IN_FLIGHT = 3

# Finished frames allowed to be waiting on the writer. Disk is an order of magnitude
# faster than the models, so this only has to absorb a stalled write, not a backlog.
MAX_PENDING_WRITES = 8

# Analysed frames spent measuring the machine before the stride is fixed. Re-tuning
# it mid-run is what makes the analysed video uneven, so it is set once and kept.
# The first frames are discarded outright: CUDA autotuning makes frame 1 ~50x the
# steady-state cost, and a median over the rest ignores whatever else stalls once.
STRIDE_WARMUP = 12
STRIDE_DISCARD = 2


def _step(msg):
    print(f"   . {msg}", flush=True)


def _hold_until(deadline):
    """
    Block until `deadline` (a perf_counter value).

    Plain sleep, deliberately: busy-waiting the last millisecond would hold the GIL
    and slow the restore/detect thread far more than the timer jitter it removes.
    """
    remaining = deadline - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)


def order_corners(pts):
    """
    Sort 4 arbitrary points into TL, TR, BR, BL, so click order stops mattering.

    TL minimises x+y and BR maximises it; TR minimises y-x and BL maximises it.
    Holds for any convex quad, including strong perspective trapezoids.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    ordered = (pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)])
    return [tuple(int(round(v)) for v in p) for p in ordered]


def warp_size(points):
    """Target resolution implied by the corners. Rejects degenerate quads loudly."""
    if len(points) != 4:
        raise ValueError(f"need exactly 4 points, got {len(points)}")
    w = max(abs(points[0][0] - points[1][0]), abs(points[2][0] - points[3][0]))
    h = max(abs(points[0][1] - points[2][1]), abs(points[1][1] - points[3][1]))
    if w < 2 or h < 2:
        raise ValueError(f"degenerate quad: w={w}, h={h}. Corners must span an area.")
    return int(w), int(h)


def homography(points, w, h):
    src = np.array(points, dtype=np.float32)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def warp(frame, matrix, w, h, out_size):
    """Rectify with a precomputed matrix, then resize to the model input."""
    return resize(cv2.warpPerspective(frame, matrix, (w, h)), out_size)


def _capture_mean(cap, frames=5, discard=4):
    """Average a few frames to suppress noise, discarding stale buffered ones first."""
    for _ in range(discard):
        cap.read()
    acc = None
    for _ in range(frames):
        ret, frame = cap.read()
        if not ret:
            return None
        f = frame.astype(np.float32)
        acc = f if acc is None else acc + f
    return (acc / frames).astype(np.uint8)


def _quad_from_mask(mask, min_area):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    for r in (0.01, 0.02, 0.03, 0.05):
        approx = cv2.approxPolyDP(cnt, r * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    return None


def auto_calibrate(cap, projector_shape, settle=0.8, frames=5, min_area_frac=0.01,
                   hold=1.5, debug=None):
    """
    Find the projected area's corners with no user input.

    Flashes black then white and diffs the two camera shots: only the projection
    changes, so ambient light cancels out and the largest 4-corner contour of the
    difference is the screen. Returns TL/TR/BR/BL, or None to fall back to manual
    clicking. `debug` collects intermediates, filled even on failure.
    """
    dbg = debug if debug is not None else {}
    h, w = projector_shape[:2]
    black = np.zeros((h, w, 3), dtype=np.uint8)
    white = np.full((h, w, 3), 255, dtype=np.uint8)

    print("calibrating: flashing black/white on the projector ...")
    shots = []
    for img in (black, white):
        cv2.imshow(WINDOW, img)
        cv2.waitKey(1)
        time.sleep(settle)          # projector latency + camera auto-exposure
        shot = _capture_mean(cap, frames=frames)
        if shot is None:
            print("warning: camera read failed during calibration.")
            return None
        shots.append(shot)
    dbg["shot_black"], dbg["shot_white"] = shots

    diff = cv2.absdiff(shots[1], shots[0])
    gray = cv2.GaussianBlur(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dbg["diff"] = diff
    dbg["mask"] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    quad = _quad_from_mask(mask, min_area_frac * gray.shape[0] * gray.shape[1])
    if quad is None:
        print("warning: no 4-corner projected region found.")
        return None

    points = order_corners(quad)
    try:
        warp_size(points)
    except ValueError as e:
        print(f"warning: calibration rejected: {e}")
        return None

    canvas = draw_quad(shots[1], points, "auto calibration (white flash)")
    dbg["overlay"] = canvas
    cv2.imshow("AutoCalib", canvas)
    cv2.waitKey(1)
    time.sleep(hold)
    cv2.destroyWindow("AutoCalib")
    print(f"calibrated: {points}")
    return points


def manual_calibrate(cap, hold=1.0):
    """Click the 4 corners in any order; they get sorted into TL/TR/BR/BL."""
    points = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    window = "CheckPoint"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    print("click the 4 corners of the projected area in any order ('q' aborts).")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("warning: camera read failed during calibration.")
                return None
            canvas = frame.copy()
            for i, p in enumerate(points):
                cv2.circle(canvas, p, 5, (0, 0, 255), -1)
                if i:
                    cv2.line(canvas, points[i - 1], p, (255, 0, 0), 2)
            if len(points) == 4:
                cv2.line(canvas, points[-1], points[0], (255, 0, 0), 2)
            cv2.imshow(window, canvas)
            if len(points) == 4:
                cv2.waitKey(1)
                time.sleep(hold)
                return order_corners(points)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return None
    finally:
        cv2.destroyWindow(window)


def open_webcam(index, backend="auto", width=None, height=None, fps=None):
    """
    Open a webcam and report what actually happened.

    On Windows the default MSMF backend regularly stalls for tens of seconds opening
    a camera (measured 20.8s vs 0.9s for DirectShow), so 'auto' tries DirectShow first.
    """
    if backend == "auto":
        order = ([("dshow", cv2.CAP_DSHOW), ("msmf", cv2.CAP_MSMF)]
                 if sys.platform == "win32" else [("default", None)])
    else:
        order = [(backend, CAM_BACKENDS[backend])]

    cap = None
    for name, flag in order:
        _step(f"opening webcam {index} via {name} ...")
        t0 = time.time()
        cap = cv2.VideoCapture(index) if flag is None else cv2.VideoCapture(index, flag)
        if cap.isOpened():
            _step(f"opened via {name} in {time.time() - t0:.1f}s")
            break
        cap.release()
        print(f"warning: {name} could not open webcam {index}; trying next backend.")
        cap = None
    if cap is None:
        return None

    for prop, value in ((cv2.CAP_PROP_FRAME_WIDTH, width),
                        (cv2.CAP_PROP_FRAME_HEIGHT, height),
                        (cv2.CAP_PROP_FPS, fps)):
        if value:
            cap.set(prop, value)

    ret, frame = cap.read()       # drivers silently ignore requests; read back reality
    if not ret or frame is None:
        print("warning: webcam opened but the first read failed.")
        cap.release()
        return None
    h, w = frame.shape[:2]
    print(f"webcam {index}: {w}x{h} @ {cap.get(cv2.CAP_PROP_FPS):.0f}fps "
          f"(requested {width}x{height} @ {fps})")
    return cap


def place_window(screen_index, image=None, announce=True) -> Optional[Monitor]:
    """
    The projector window, fullscreen on `screen_index`.

    Thin wrapper over utils.display so the window title lives in one place and
    collect.py / record.py keep importing `place_window` from here.
    """
    return place_fullscreen(WINDOW, screen_index, image, announce)


@dataclass
class LiveResult:
    """Field names match RunRecorder / offline.FrameResult."""

    frame_id: int
    name_id: str
    light: np.ndarray
    distorted: np.ndarray
    distorted_det: np.ndarray
    restored: np.ndarray
    restored_det: np.ndarray
    residual: np.ndarray
    residual_mean: float = 0.0
    det_distorted: List = field(default_factory=list)
    det_restored: List = field(default_factory=list)
    t_restore: float = 0.0
    t_detect: float = 0.0
    t_worker: float = 0.0
    write_dropped: bool = False
    surface: Optional[np.ndarray] = None
    gt_boxes: List = field(default_factory=list)


def _writer(write_queue, recorder, dropped):
    """
    Everything that touches the disk, on its own thread.

    The jpegs, the mp4 encode and the csv row all wait on I/O, and none of it is
    needed to analyse the next frame. Left in the worker they were charged to
    `t_worker`, which is what the stride is computed from, so ~9 ms of writing per
    frame quietly cost analysed frames. Here they overlap with the next restore.

    Only this thread writes frame images, so the recorder needs no lock. `None` ends
    it, after the queue it has already been given is drained.
    """
    while True:
        item = write_queue.get()
        if item is None:
            return
        result, panel = item
        try:
            recorder.write_video(panel)
            if recorder.should_save(result.frame_id):
                recorder.save_frame_images(result, panel=panel)
            recorder.log_detections(result)
        except Exception as e:                      # a bad frame must not kill the run
            dropped.append(e)
            print(f"warning: writing frame {result.frame_id} failed: {e}")


def _worker(frame_queue, result_queue, write_queue, restorer, detector, stop_event,
            lost=None, detector_name="detector", min_area=500,
            min_width=20, min_height=20):
    """
    Restore, detect and build the panel - all off the main thread.

    Doing this between two projector frames is what used to pull playback below the
    clip's fps. The main loop is left with `cv2.imshow` and the counters; the disk
    goes to `_writer`. The panel is built here rather than there because the main
    loop displays it.

    `lost` collects the frame ids whose result could not be handed back, so the main
    loop can stop waiting for them. list.append is atomic, so it needs no lock.
    """
    from ..utils.visualize import draw_detections

    while not stop_event.is_set():
        try:
            frame_id, distorted, light = frame_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        started = time.perf_counter()

        t0 = time.perf_counter()
        restored_plain, residual, residual_mean = restorer.restore_full(distorted, light)
        t_restore = time.perf_counter() - t0

        distorted_plain = resize(distorted, tuple(restorer.input_size))
        t1 = time.perf_counter()
        raw_c = detector(distorted_plain)
        raw_r = detector(restored_plain)
        t_detect = time.perf_counter() - t1

        gate = dict(min_area=min_area, min_width=min_width, min_height=min_height)
        det_c = filter_detections(raw_c, **gate)
        det_r = filter_detections(raw_r, **gate)

        result = LiveResult(
            frame_id=frame_id, name_id=f"frame_{frame_id:06d}",
            light=light,
            distorted=distorted_plain,
            distorted_det=draw_detections(distorted_plain.copy(), det_c),
            restored=restored_plain,
            restored_det=draw_detections(restored_plain.copy(), det_r),
            residual=residual, residual_mean=residual_mean,
            det_distorted=det_c, det_restored=det_r,
            t_restore=t_restore, t_detect=t_detect,
        )
        panel = build_panel(result, detector_name)
        result.t_worker = time.perf_counter() - started
        try:
            write_queue.put_nowait((result, panel))
        except queue.Full:
            # Never stall the models on the disk: the run keeps its numbers, the
            # frame just does not reach result.mp4 or the jpgs.
            result.write_dropped = True

        try:
            result_queue.put((result, panel), timeout=1.0)
        except queue.Full:
            # The main loop is not draining. Dropping the result silently left it
            # counted as in flight forever, so the final drain waited out its whole
            # timeout and blamed every straggler on a missed deadline. Report it.
            if lost is not None:
                lost.append(result.frame_id)


def build_panel(result, detector_name="detector") -> np.ndarray:
    return grid_2x2(
        result.light, result.distorted_det, result.restored_det, result.residual,
        labels=["light (projected source)",
                f"distorted + {detector_name} ({len(result.det_distorted)})",
                f"restored + {detector_name} ({len(result.det_restored)})",
                f"residual (mean {result.residual_mean:.3f})"],
    )


def run_live(video, restorer, detector, recorder, background=None, camera=0,
             screen=1, offset=6, manual_calib=False, debug_view=False,
             max_frames=0, cam_size=(1280, 960), cam_fps=30, cam_backend="auto",
             calib_settle=0.8, min_area=500, min_width=20,
             min_height=20, detector_name="detector", analyse_every=0) -> Dict:
    """
    Drive the rig until the clip ends or 'q' is pressed. Returns a summary dict.

    Calibration happens ONCE before the loop and the homography is reused for every
    frame. It is never re-estimated, so if the camera or projector moves mid-run the
    warp stays wrong - use debug_view to watch for that.
    """
    out_size = tuple(restorer.input_size)

    if background is None:
        background = np.full((720, 1280, 3), 32, np.uint8)
    background_small = resize(background, out_size)

    _step(f"placing the projector window on monitor {screen} ...")
    monitor = place_window(screen, background)

    _step(f"opening the projector clip: {video}")
    clip = cv2.VideoCapture(str(video))
    if not clip.isOpened():
        raise RuntimeError(f"cannot open the projector clip: {video}")
    clip_frames = int(clip.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_fps = clip.get(cv2.CAP_PROP_FPS) or 30
    clip_size = (int(clip.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(clip.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(f"clip: {clip_frames} frames @ {clip_fps:.0f}fps, "
          f"{clip_size[0]}x{clip_size[1]} (projected as-is; "
          f"the model sees {out_size[0]}x{out_size[1]})")

    cam = open_webcam(camera, cam_backend, cam_size[0], cam_size[1], cam_fps)
    if cam is None:
        clip.release()
        cv2.destroyAllWindows()
        raise RuntimeError(
            f"cannot open webcam index {camera}\n"
            f"    try another --camera index, another --cam-backend, and close any "
            f"other app holding the camera."
        )

    calib_debug, mode, points = {}, "manual", None
    if not manual_calib:
        points = auto_calibrate(cam, background.shape, settle=calib_settle,
                                debug=calib_debug)
        cv2.imshow(WINDOW, background)
        cv2.waitKey(1)
        if points is None:
            print("falling back to manual calibration.")
        else:
            mode = "auto"
    if points is None:
        points = manual_calibrate(cam)
    if points is None:
        cam.release()
        clip.release()
        cv2.destroyAllWindows()
        raise RuntimeError("calibration aborted")

    w_cal, h_cal = warp_size(points)
    matrix = homography(points, w_cal, h_cal)
    print(f"warp target: {w_cal}x{h_cal} from {points}")

    ok, raw = cam.read()
    raw = raw if ok else None
    warped_preview = (warp(raw, matrix, w_cal, h_cal, out_size)
                      if raw is not None else None)
    written = recorder.save_calibration(points, raw, warped_preview, calib_debug)
    print(f"calibration debug: {len(written)} image(s) -> {recorder.calib_dir}")

    recorder.set(calibration={
        "mode": mode,
        "points_tl_tr_br_bl": [list(map(int, p)) for p in points],
        "warp_target": [w_cal, h_cal],
        "homography": matrix.tolist(),
        "output_size": list(out_size),
        "note": "estimated once before the loop; not re-estimated per frame",
    }, capture={
        "camera_index": camera, "backend": cam_backend,
        "requested": [cam_size[0], cam_size[1], cam_fps],
        "actual": [int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)),
                   int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                   round(cam.get(cv2.CAP_PROP_FPS), 1)],
    }, projector={
        "clip": str(video), "frames": clip_frames, "fps": round(clip_fps, 2),
        "projected_size": list(clip_size), "model_input_size": list(out_size),
        "screen_index": screen, "offset_frames": offset,
        "monitor": dict(zip(monitor._fields, monitor)) if monitor else None,
    })

    _step("starting the restore/detect worker and the writer ...")
    frame_queue = queue.Queue(maxsize=MAX_IN_FLIGHT)
    result_queue = queue.Queue(maxsize=MAX_IN_FLIGHT + 2)
    write_queue = queue.Queue(maxsize=MAX_PENDING_WRITES)
    write_errors, lost_results = [], []
    stop_event = threading.Event()
    writer = threading.Thread(target=_writer, daemon=True,
                              args=(write_queue, recorder, write_errors))
    writer.start()
    worker = threading.Thread(
        target=_worker, daemon=True,
        args=(frame_queue, result_queue, write_queue, restorer, detector, stop_event,
              lost_results),
        kwargs={"detector_name": detector_name, "min_area": min_area,
                "min_width": min_width, "min_height": min_height})
    worker.start()

    place_window(screen, announce=False)
    # Pre-roll so `light` lags `distorted` by `offset` frames, compensating rig latency.
    light_buffer = [background_small.copy() for _ in range(max(0, offset))]

    frame_count = processed = dropped = skipped = writes_dropped = 0
    totals = {"distorted": 0, "restored": 0}
    sums = {"residual": 0.0, "t_restore": 0.0, "t_detect": 0.0}
    # Analysis runs behind the projector, never in front of it: the loop hands a frame
    # to the worker only when it has room and moves on regardless, so restore+detect
    # never holds up the clip. MAX_IN_FLIGHT frames may be under analysis at once.
    in_flight = 0
    stop = False
    # Analyse every Nth frame rather than "whenever the worker is free": an even
    # spacing is what makes result.mp4 play like the clip, only at a lower rate.
    # Auto measures the machine for STRIDE_WARMUP frames, then holds that stride.
    stride, auto_stride, samples = max(1, analyse_every), analyse_every <= 0, []

    def consume(result, panel):
        """Show one finished frame and fold it into the counters; the writer stores it."""
        nonlocal processed, stride, auto_stride, writes_dropped
        writes_dropped += bool(result.write_dropped)
        totals["distorted"] += len(result.det_distorted)
        totals["restored"] += len(result.det_restored)
        sums["residual"] += result.residual_mean
        sums["t_restore"] += result.t_restore
        sums["t_detect"] += result.t_detect
        processed += 1

        if auto_stride and result.t_worker:
            samples.append(result.t_worker)
            if len(samples) >= STRIDE_WARMUP:
                # 20% headroom: a stride the worker only just sustains misses its
                # slot whenever a frame runs long, and a miss is a visible hitch.
                typical = statistics.median(samples[STRIDE_DISCARD:])
                stride = max(1, math.ceil(clip_fps * typical * 1.2))
                auto_stride = False
                print(f"analysing every {stride} frame(s) = "
                      f"{clip_fps / stride:.1f} of {clip_fps:.0f} fps "
                      f"({typical * 1000:.0f} ms per frame).", flush=True)
        if processed == 1:
            print(f"first frame through the pipeline in {time.time() - t0:.1f}s "
                  f"(includes warmup).", flush=True)
        cv2.imshow(PANEL, panel)

    def collect(block=False):
        """Take whatever the worker has finished. Only the final drain blocks."""
        nonlocal in_flight, dropped
        while True:
            # Results the worker finished but could not hand back are gone; they must
            # still leave in_flight, or the final drain waits for something no one
            # will ever send.
            while lost_results:
                lost_results.pop()
                in_flight -= 1
                dropped += 1
            if not in_flight or not (block or not result_queue.empty()):
                return
            try:
                result = result_queue.get(timeout=10)
            except queue.Empty:
                print(f"warning: {in_flight} frame(s) produced no result in 10s.")
                dropped += in_flight
                in_flight = 0
                return
            in_flight -= 1
            consume(*result)

    print("running - press 'q' in the Combined_View window to stop.", flush=True)
    t0 = time.time()
    # The projector must keep the clip's own cadence; analysis throughput is separate.
    frame_interval = 1.0 / max(clip_fps, 1e-6)
    next_frame_at = time.perf_counter()

    try:
        while not stop:
            if max_frames and frame_count >= max_frames:
                print(f"reached max_frames={max_frames}.")
                break

            ok_c, cam_frame = cam.read()
            if not ok_c:
                print("webcam ended.")
                break
            distorted = warp(cam_frame, matrix, w_cal, h_cal, out_size)

            if frame_count == 0:
                # Once per run: what the camera saw and what the warp made of it.
                written, figure = recorder.save_warp_pair(cam_frame, distorted, points)
                print(f"first-frame warp: {len(written)} image(s) -> "
                      f"{recorder.calib_dir}")
                cv2.imshow(WARP_WINDOW, figure)
                cv2.waitKey(1)

            ok_p, light_frame = clip.read()
            if not ok_p:
                print("projector clip ended.")
                break

            # Project the clip at its own resolution and keep the model's copy
            # separate. Shrinking before projecting would put the network's input
            # size on the screen, only for the fullscreen window to blow it back up.
            light_buffer.append(resize(light_frame, out_size))
            cv2.imshow(WINDOW, light_frame)
            if debug_view:
                cv2.imshow(DEBUG_WINDOW,
                           draw_quad(cam_frame, points,
                                     f"pre-warp camera  frame {frame_count}"))

            light = light_buffer.pop(0)
            if frame_count % stride == 0:
                try:
                    # Hand over every on-stride frame, letting the queue absorb a
                    # worker that ran long. Gating on "is the worker idle" instead
                    # would drop whole slots and put a hitch in the analysed video.
                    # The raw camera frame stays here: the worker only ever needed
                    # the rectified `distorted`, and queueing both pinned a
                    # full-resolution frame per slot for nothing.
                    frame_queue.put_nowait((frame_count, distorted, light))
                    in_flight += 1
                except queue.Full:
                    skipped += 1        # stride is still catching up; let it go by
            else:
                skipped += 1
            frame_count += 1

            collect()
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                print("stopped by user.")
                break

            # Absolute deadlines, so a slow iteration is absorbed rather than added
            # to the next one. A camera that already paces the loop never sleeps here.
            _hold_until(next_frame_at)
            next_frame_at = max(next_frame_at + frame_interval, time.perf_counter())

        collect(block=True)     # the clip ended; finish what the worker still holds
    except KeyboardInterrupt:
        print("\ninterrupted.")
    finally:
        stop_event.set()
        worker.join(timeout=3)
        # The writer holds the only copy of frames that are already analysed, so it
        # is drained rather than cut off - the recorder is closed right after this.
        write_queue.put(None)
        writer.join(timeout=30)
        clip.release()
        cam.release()
        cv2.destroyAllWindows()

    n = max(processed, 1)
    elapsed = max(time.time() - t0, 1e-6)
    summary = {
        "frames_projected": frame_count,
        "frames_processed": processed,
        "frames_skipped": skipped,
        "frames_dropped": dropped,
        "writes_dropped": writes_dropped,
        "write_errors": len(write_errors),
        "analyse_every": stride,
        "fps_projector": round(frame_count / elapsed, 2),
        "fps_end_to_end": round(processed / elapsed, 2),
        "detections_total": totals,
        "detections_per_frame": {k: round(v / n, 3) for k, v in totals.items()},
        "residual_mean": round(sums["residual"] / n, 5),
        "avg_restore_ms": round(sums["t_restore"] / n * 1000, 2),
        "avg_detect_ms": round(sums["t_detect"] / n * 1000, 2),
    }
    if totals["distorted"]:
        summary["detection_delta_pct"] = round(
            (totals["restored"] - totals["distorted"]) / totals["distorted"] * 100, 1)
    return summary
