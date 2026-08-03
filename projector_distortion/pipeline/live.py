"""
Live pipeline: webcam + projector rig. Needs real hardware.

    projector shows BeamVideo frame N
    webcam captures it                  -> 4-point homography -> `pro`
    the frame shown --offset frames ago -> `beam`
    restore(pro, beam) -> detect both -> 2x2 panel + recorder

On Windows, cv2.setWindowProperty(WND_PROP_FULLSCREEN) snaps the window back to the
primary display and silently undoes cv2.moveWindow(), which is why a naive --screen
has no effect. Geometry goes through the Win32 API instead.
"""

import ctypes
import queue
import sys
import threading
import time
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

from ..models import filter_detections
from ..utils.image import resize
from ..utils.visualize import draw_quad, grid_2x2

WINDOW = "Projector_Display"
PANEL = "Combined_View"
DEBUG_WINDOW = "PreWarp_Debug"

CLICK_ORDER = "TL -> TR -> BR -> BL"

CAM_BACKENDS = {"auto": None, "any": cv2.CAP_ANY, "dshow": cv2.CAP_DSHOW,
                "msmf": cv2.CAP_MSMF, "v4l2": cv2.CAP_V4L2}


def _step(msg):
    print(f"   . {msg}", flush=True)


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


Monitor = namedtuple("Monitor", "x y width height primary name")


class _RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


class _MONITORINFOEXW(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_ulong), ("rcMonitor", _RECT), ("rcWork", _RECT),
                ("dwFlags", ctypes.c_ulong), ("szDevice", ctypes.c_wchar * 32)]


_MONITORINFOF_PRIMARY = 1


def _set_dpi_aware():
    """Report physical pixels; without this, display scaling corrupts coordinates."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def _monitors_win32():
    user32 = ctypes.windll.user32
    proc = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                              ctypes.POINTER(_RECT), ctypes.c_ssize_t)
    found = []

    def _cb(hmon, hdc, lprc, lparam):
        info = _MONITORINFOEXW()
        info.cbSize = ctypes.sizeof(_MONITORINFOEXW)
        if user32.GetMonitorInfoW(hmon, ctypes.byref(info)):
            r = info.rcMonitor
            found.append(Monitor(r.left, r.top, r.right - r.left, r.bottom - r.top,
                                 bool(info.dwFlags & _MONITORINFOF_PRIMARY),
                                 info.szDevice))
        return 1

    user32.EnumDisplayMonitors(None, None, proc(_cb), 0)
    return found


def list_monitors() -> List[Monitor]:
    """All displays, primary first then left to right, so index 0 is always primary."""
    monitors = []
    if sys.platform == "win32":
        _set_dpi_aware()
        try:
            monitors = _monitors_win32()
        except Exception as e:
            print(f"warning: Win32 monitor enumeration failed ({e}); trying screeninfo.")
    if not monitors:
        try:
            from screeninfo import get_monitors
            monitors = [Monitor(m.x, m.y, m.width, m.height,
                                bool(getattr(m, "is_primary", False)), m.name or "")
                        for m in get_monitors()]
        except Exception as e:
            print(f"warning: screeninfo unavailable ({e}).")
    monitors.sort(key=lambda m: (not m.primary, m.x, m.y))
    return monitors


def _borderless_win32(title, mon):
    """Strip the frame and pin the window to `mon`. True when it took effect."""
    user32 = ctypes.windll.user32
    user32.FindWindowW.restype = ctypes.c_void_p
    hwnd = user32.FindWindowW(None, title)
    if not hwnd:
        return False

    GWL_STYLE, GWL_EXSTYLE = -16, -20
    WS_POPUP, WS_VISIBLE = 0x80000000, 0x10000000
    FLAGS = 0x0020 | 0x0040 | 0x0010    # FRAMECHANGED | SHOWWINDOW | NOACTIVATE

    def signed(v):
        return v - (1 << 32) if v >= (1 << 31) else v

    user32.SetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_long]
    user32.SetWindowPos.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_uint]
    user32.SetWindowLongW(hwnd, GWL_STYLE, signed(WS_POPUP | WS_VISIBLE))
    user32.SetWindowLongW(hwnd, GWL_EXSTYLE, 0)
    user32.SetWindowPos(hwnd, None, mon.x, mon.y, mon.width, mon.height, FLAGS)
    return True


def place_window(screen_index, image=None, announce=True) -> Optional[Monitor]:
    """Put the projector window fullscreen on `screen_index`. Returns that Monitor."""
    monitors = list_monitors()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    if not monitors:
        print("warning: no monitors detected; plain fullscreen on the primary display.")
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        if image is not None:
            cv2.imshow(WINDOW, image)
            cv2.waitKey(1)
        return None

    if announce:
        print(f"{len(monitors)} monitor(s) detected:")
        for i, m in enumerate(monitors):
            star = " (primary)" if m.primary else ""
            print(f"      --screen {i} -> {m.width}x{m.height} at ({m.x},{m.y})"
                  f"{star}  {m.name}")

    if not 0 <= screen_index < len(monitors):
        print(f"warning: monitor {screen_index} unavailable; using 0 (primary).")
        screen_index = 0
    target = monitors[screen_index]

    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_AUTOSIZE, 0)
    cv2.moveWindow(WINDOW, target.x, target.y)
    cv2.resizeWindow(WINDOW, target.width, target.height)
    if image is not None:
        cv2.imshow(WINDOW, image)     # the window must exist before Win32 can find it
        cv2.waitKey(1)

    if sys.platform == "win32":
        if not _borderless_win32(WINDOW, target):
            print("warning: could not restyle the window; using OpenCV fullscreen.")
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.waitKey(1)
    return target


@dataclass
class LiveResult:
    """Field names match RunRecorder / offline.FrameResult."""

    frame_id: int
    name_id: str
    beam: np.ndarray
    captured: np.ndarray
    captured_det: np.ndarray
    restored: np.ndarray
    restored_det: np.ndarray
    residual: np.ndarray
    residual_mean: float = 0.0
    det_captured: List = field(default_factory=list)
    det_restored: List = field(default_factory=list)
    t_restore: float = 0.0
    t_detect: float = 0.0
    clean: Optional[np.ndarray] = None
    gt_boxes: List = field(default_factory=list)


def _worker(frame_queue, result_queue, restorer, detector, stop_event,
            best_per_class=False, min_area=500, min_width=20, min_height=20):
    from ..utils.visualize import draw_detections

    while not stop_event.is_set():
        try:
            frame_id, captured, beam = frame_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        t0 = time.perf_counter()
        restored_clean, residual, residual_mean = restorer.restore_full(captured, beam)
        t_restore = time.perf_counter() - t0

        captured_clean = resize(captured, tuple(restorer.input_size))
        t1 = time.perf_counter()
        raw_c = detector(captured_clean)
        raw_r = detector(restored_clean)
        t_detect = time.perf_counter() - t1

        gate = dict(min_area=min_area, min_width=min_width, min_height=min_height,
                    best_per_class=best_per_class)
        det_c = filter_detections(raw_c, **gate)
        det_r = filter_detections(raw_r, **gate)

        result = LiveResult(
            frame_id=frame_id, name_id=f"frame_{frame_id:06d}",
            beam=beam,
            captured=captured_clean,
            captured_det=draw_detections(captured_clean.copy(), det_c),
            restored=restored_clean,
            restored_det=draw_detections(restored_clean.copy(), det_r),
            residual=residual, residual_mean=residual_mean,
            det_captured=det_c, det_restored=det_r,
            t_restore=t_restore, t_detect=t_detect,
        )
        try:
            result_queue.put(result, timeout=1.0)
        except queue.Full:
            continue


def build_panel(result, detector_name="detector") -> np.ndarray:
    return grid_2x2(
        result.beam, result.captured_det, result.restored_det, result.residual,
        labels=["beam (projected source)",
                f"captured + {detector_name} ({len(result.det_captured)})",
                f"restored + {detector_name} ({len(result.det_restored)})",
                f"|residual| mean {result.residual_mean:.3f}"],
    )


def run_live(video, restorer, detector, recorder, background=None, camera=0,
             screen=1, offset=6, manual_calib=False, debug_view=False,
             max_frames=0, cam_size=(1280, 960), cam_fps=30, cam_backend="auto",
             calib_settle=0.8, best_per_class=False, min_area=500, min_width=20,
             min_height=20, detector_name="detector") -> Dict:
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
    print(f"clip: {clip_frames} frames @ {clip_fps:.0f}fps")

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
        "screen_index": screen, "offset_frames": offset,
        "monitor": dict(zip(monitor._fields, monitor)) if monitor else None,
    })

    _step("starting the restore/detect worker ...")
    frame_queue, result_queue = queue.Queue(maxsize=10), queue.Queue(maxsize=10)
    stop_event = threading.Event()
    worker = threading.Thread(
        target=_worker, daemon=True,
        args=(frame_queue, result_queue, restorer, detector, stop_event),
        kwargs={"best_per_class": best_per_class, "min_area": min_area,
                "min_width": min_width, "min_height": min_height})
    worker.start()

    place_window(screen, announce=False)
    # Pre-roll so `beam` lags `pro` by `offset` frames, compensating rig latency.
    pro_buffer = [background_small.copy() for _ in range(max(0, offset))]

    frame_count = processed = dropped = 0
    totals = {"captured": 0, "restored": 0}
    sums = {"residual": 0.0, "t_restore": 0.0, "t_detect": 0.0}
    print("running - press 'q' in the Combined_View window to stop.", flush=True)
    t0 = time.time()

    try:
        while True:
            if max_frames and frame_count >= max_frames:
                print(f"reached max_frames={max_frames}.")
                break

            ok_c, cam_frame = cam.read()
            if not ok_c:
                print("webcam ended.")
                break
            captured = warp(cam_frame, matrix, w_cal, h_cal, out_size)

            ok_p, pro_frame = clip.read()
            if not ok_p:
                print("projector clip ended.")
                break
            pro_frame = resize(pro_frame, out_size)

            pro_buffer.append(pro_frame)
            cv2.imshow(WINDOW, pro_frame)
            if debug_view:
                cv2.imshow(DEBUG_WINDOW,
                           draw_quad(cam_frame, points,
                                     f"pre-warp camera  frame {frame_count}"))

            beam = pro_buffer.pop(0)
            try:
                frame_queue.put((frame_count, captured, beam), timeout=1.0)
            except queue.Full:
                dropped += 1
                frame_count += 1
                continue
            try:
                result = result_queue.get(timeout=10)
            except queue.Empty:
                dropped += 1
                print(f"warning: frame {frame_count} produced no result in 10s.")
                frame_count += 1
                continue

            panel = build_panel(result, detector_name)
            recorder.write_video(panel)
            saved = recorder.should_save(result.frame_id)
            if saved:
                recorder.save_frame_images(result, panel=panel, raw_frame=cam_frame)
            recorder.log_frame(result, saved=saved)

            totals["captured"] += len(result.det_captured)
            totals["restored"] += len(result.det_restored)
            sums["residual"] += result.residual_mean
            sums["t_restore"] += result.t_restore
            sums["t_detect"] += result.t_detect
            processed += 1
            if processed == 1:
                print(f"first frame through the pipeline in {time.time() - t0:.1f}s "
                      f"(includes warmup).", flush=True)

            cv2.imshow(PANEL, panel)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("stopped by user.")
                break
            frame_count += 1
    except KeyboardInterrupt:
        print("\ninterrupted.")
    finally:
        stop_event.set()
        worker.join(timeout=3)
        clip.release()
        cam.release()
        cv2.destroyAllWindows()

    n = max(processed, 1)
    elapsed = max(time.time() - t0, 1e-6)
    summary = {
        "frames_processed": processed,
        "frames_dropped": dropped,
        "fps_end_to_end": round(processed / elapsed, 2),
        "detections_total": totals,
        "detections_per_frame": {k: round(v / n, 3) for k, v in totals.items()},
        "residual_mean": round(sums["residual"] / n, 5),
        "avg_restore_ms": round(sums["t_restore"] / n * 1000, 2),
        "avg_detect_ms": round(sums["t_detect"] / n * 1000, 2),
    }
    if totals["captured"]:
        summary["detection_delta_pct"] = round(
            (totals["restored"] - totals["captured"]) / totals["captured"] * 100, 1)
    return summary
