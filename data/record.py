#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project a clip on the rig's screen and record the webcam's view of it as one mp4.

The projection half of `demo.py --live` with the restore/detect half removed: the
clip goes out fullscreen on --screen at its own fps, the camera feed goes into a
single mp4, and nothing else happens. Useful for capturing raw distorted footage
before there are weights to run, or for proving a rig works end to end.

    python data/record.py                        # BeamVideo -> data/recordings/rec_<t>.mp4
    python data/record.py --screen 2 --seconds 30
    python data/record.py --clip data/SampleData/sample_video/mIni_Video_1.mp4 --loop
    python data/record.py --warp                 # record rectified frames, not raw ones

Recording starts as soon as the window is up - no keypress. 'q' in the preview or
the projector window stops early, as do the end of the clip, --seconds and
--max-frames.

The mp4 lands next to a .json holding the clip, camera and monitor settings the run
actually used, plus the measured capture rate.
"""

import argparse
import json
import os
import queue
import sys
import threading
import time
from datetime import datetime

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_CLIP = os.path.join(DATA_DIR, "live", "BeamVideo.mp4")
DEFAULT_BACKGROUND = os.path.join(DATA_DIR, "live", "BaseBackGround.jpg")
DEFAULT_OUT_DIR = os.path.join(DATA_DIR, "recordings")

PREVIEW = "Recording_Preview"

# Camera frames allowed to wait on the encoder. Deep enough to absorb a slow write,
# shallow enough that a stalled encoder is reported as drops instead of eating RAM.
MAX_QUEUED = 32


def shrink(img, size):
    """`size` is (width, height); None or a match keeps the image as it is."""
    if size is None or (img.shape[1], img.shape[0]) == tuple(size):
        return img
    shrinking = size[0] * size[1] < img.shape[1] * img.shape[0]
    return cv2.resize(img, tuple(size),
                      interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR)


def hold_until(deadline):
    """Block until `deadline` (a perf_counter value). Plain sleep, no busy-waiting."""
    remaining = deadline - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)


def under_root(path):
    """Relative inputs resolve against the repo, so this runs from any directory."""
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


class Recorder:
    """
    Encode camera frames into one mp4, off the capture thread.

    The header fps is measured rather than requested. Webcams routinely deliver
    something other than what they were asked for - and report a third number - and a
    wrong header is exactly what makes a recording play back in fast or slow motion.
    The probe frames are held in memory until the rate is known, so measuring costs
    no footage.
    """

    def __init__(self, path, codec="mp4v", fps=0.0, probe=24, transform=None):
        self.path = path
        self.codec = codec
        self.fps = float(fps)
        self.probe = max(2, int(probe))
        self.transform = transform
        self.writer = None
        self.size = None
        self.frames_written = 0
        self.failed = None
        self._buffer = []          # (timestamp, frame) until the fps is known
        self._first_t = None
        self._last_t = None

    def _open(self, frame):
        self.size = (frame.shape[1], frame.shape[0])
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*self.codec),
                                 self.fps, self.size)
        if not writer.isOpened():
            self.failed = (f"cannot open the video writer for {self.path} "
                           f"with codec '{self.codec}' at {self.size[0]}x{self.size[1]}")
            return False
        self.writer = writer
        print(f"recording {self.size[0]}x{self.size[1]} @ {self.fps:.2f}fps "
              f"({self.codec}) -> {self.path}", flush=True)
        return True

    def add(self, timestamp, frame):
        """Take one camera frame. False once the writer has given up."""
        if self.failed:
            return False
        if self.transform is not None:
            frame = self.transform(frame)
        if self._first_t is None:
            self._first_t = timestamp
        self._last_t = timestamp

        if self.writer is None:
            self._buffer.append((timestamp, frame))
            if self.fps <= 0 and len(self._buffer) < self.probe:
                return True
            if self.fps <= 0:
                span = self._buffer[-1][0] - self._buffer[0][0]
                measured = (len(self._buffer) - 1) / span if span > 1e-6 else 0.0
                # A camera that stalled during warmup can measure absurdly low; the
                # clamp keeps a bad probe from producing an unplayable header.
                self.fps = min(max(measured, 1.0), 240.0)
                print(f"measured capture rate: {measured:.2f} fps "
                      f"over {len(self._buffer)} frame(s)", flush=True)
            if not self._open(self._buffer[0][1]):
                self._buffer.clear()
                return False
            for _, buffered in self._buffer:
                self._write(buffered)
            self._buffer.clear()
            return True

        self._write(frame)
        return True

    def _write(self, frame):
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = shrink(frame, self.size)
        self.writer.write(frame)
        self.frames_written += 1

    def close(self):
        """Flush whatever is still buffered, then release. Returns a summary dict."""
        if self.writer is None and self._buffer:
            # Stopped before the probe filled: fall back to the elapsed-time rate.
            span = self._buffer[-1][0] - self._buffer[0][0]
            if self.fps <= 0:
                self.fps = min(max((len(self._buffer) - 1) / span
                                   if span > 1e-6 else 1.0, 1.0), 240.0)
            if self._open(self._buffer[0][1]):
                for _, buffered in self._buffer:
                    self._write(buffered)
            self._buffer.clear()
        if self.writer is not None:
            self.writer.release()
            self.writer = None

        elapsed = (self._last_t - self._first_t) if self._first_t is not None else 0.0
        mb = (round(os.path.getsize(self.path) / 1e6, 2)
              if os.path.exists(self.path) else 0.0)
        return {"path": self.path, "codec": self.codec,
                "fps_header": round(self.fps, 3),
                "size": list(self.size) if self.size else None,
                "frames": self.frames_written,
                "capture_span_sec": round(elapsed, 2),
                "measured_fps": round(self.frames_written / elapsed, 2)
                if elapsed > 1e-6 else None,
                "file_mb": mb, "error": self.failed}


def capture_loop(cam, frames, stop_event, latest, stats):
    """Read the camera as fast as it delivers and hand every frame to the encoder."""
    while not stop_event.is_set():
        ok, frame = cam.read()
        if not ok:
            stats["read_failed"] = True
            break
        stats["read"] += 1
        latest[0] = frame
        try:
            frames.put_nowait((time.perf_counter(), frame))
        except queue.Full:
            stats["dropped"] += 1      # the encoder is behind; the rig keeps running
    try:
        frames.put(None, timeout=2.0)  # sentinel: no more frames are coming
    except queue.Full:
        pass


def encode_loop(recorder, frames, stop_event, stats):
    """Drain the queue into the mp4 until the sentinel arrives or the writer dies."""
    while True:
        try:
            item = frames.get(timeout=0.5)
        except queue.Empty:
            if stop_event.is_set():
                break
            continue
        if item is None:
            break
        if not recorder.add(*item):
            stats["encoder_failed"] = True
            stop_event.set()
            break


def build_transform(matrix, warp_wh, out_size):
    """Rectify (optional) then resize (optional), as one callable for the encoder."""
    if matrix is None and out_size is None:
        return None

    def transform(frame):
        if matrix is not None:
            frame = cv2.warpPerspective(frame, matrix, warp_wh)
        return shrink(frame, out_size)

    return transform


def calibrate(cam, background, manual, settle):
    """Four corners of the projected area, exactly as the live pipeline finds them."""
    from projector_distortion.pipeline.live import (
        WINDOW, auto_calibrate, homography, manual_calibrate, warp_size,
    )

    points = None
    if not manual:
        points = auto_calibrate(cam, background.shape, settle=settle)
        cv2.imshow(WINDOW, background)
        cv2.waitKey(1)
        if points is None:
            print("falling back to manual calibration.")
    if points is None:
        points = manual_calibrate(cam)
    if points is None:
        raise SystemExit("calibration aborted")

    w_cal, h_cal = warp_size(points)
    print(f"warp target: {w_cal}x{h_cal} from {points}")
    return points, homography(points, w_cal, h_cal), (w_cal, h_cal)


def run(args):
    from projector_distortion.pipeline.live import WINDOW, open_webcam, place_window

    clip_path = under_root(args.clip)
    if not os.path.exists(clip_path):
        raise SystemExit(f"projector clip not found: {args.clip}")

    bg_path = under_root(args.background)
    background = cv2.imread(bg_path) if os.path.exists(bg_path) else None
    if background is None:
        print(f"warning: background image not found ({args.background}); "
              f"using a flat grey frame.")
        background = np.full((720, 1280, 3), 32, np.uint8)

    clip = cv2.VideoCapture(clip_path)
    if not clip.isOpened():
        raise SystemExit(f"cannot open the projector clip: {clip_path}")
    clip_frames = int(clip.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_fps = clip.get(cv2.CAP_PROP_FPS) or 30.0
    clip_size = (int(clip.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(clip.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(f"clip: {clip_frames} frames @ {clip_fps:.0f}fps, "
          f"{clip_size[0]}x{clip_size[1]} (projected as-is)")

    monitor = place_window(args.screen, background)

    cam = open_webcam(args.camera, args.cam_backend, args.cam_width, args.cam_height,
                      args.cam_fps)
    if cam is None:
        clip.release()
        cv2.destroyAllWindows()
        raise SystemExit(
            f"cannot open webcam index {args.camera}\n"
            f"    try another --camera index or --cam-backend, and close any other "
            f"app holding the camera.")
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    points = matrix = warp_wh = None
    if args.warp:
        points, matrix, warp_wh = calibrate(cam, background, args.manual_calib,
                                            args.calib_settle)
    out_size = None if 0 in args.rec_size else tuple(args.rec_size)

    if args.start_delay > 0:
        print(f"starting in {args.start_delay:.1f}s ...", flush=True)
        cv2.imshow(WINDOW, background)
        cv2.waitKey(1)
        time.sleep(args.start_delay)

    recorder = Recorder(args.out, codec=args.codec, fps=args.fps, probe=args.fps_probe,
                        transform=build_transform(matrix, warp_wh, out_size))
    frames = queue.Queue(maxsize=MAX_QUEUED)
    stop_event = threading.Event()
    latest = [None]
    stats = {"read": 0, "dropped": 0, "read_failed": False, "encoder_failed": False}

    grabber = threading.Thread(target=capture_loop, daemon=True,
                               args=(cam, frames, stop_event, latest, stats))
    encoder = threading.Thread(target=encode_loop, daemon=True,
                               args=(recorder, frames, stop_event, stats))

    if args.preview:
        cv2.namedWindow(PREVIEW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(PREVIEW, 480, 360)
        # The new window can steal focus and pull the projector window off its
        # monitor on Windows; put it back before anything is projected.
        place_window(args.screen, background, announce=False)

    print("recording - press 'q' to stop.", flush=True)
    grabber.start()
    encoder.start()

    projected = loops = 0
    t0 = time.time()
    frame_interval = 1.0 / max(clip_fps, 1e-6)
    next_frame_at = time.perf_counter()
    reason = "clip ended"
    try:
        while not stop_event.is_set():
            if args.max_frames and projected >= args.max_frames:
                reason = f"reached --max-frames {args.max_frames}"
                break
            if args.seconds and (time.time() - t0) >= args.seconds:
                reason = f"reached --seconds {args.seconds}"
                break
            if stats["read_failed"]:
                reason = "webcam read failed"
                break
            if stats["encoder_failed"]:
                reason = "encoder failed"
                break

            ok, pro_frame = clip.read()
            if not ok:
                if not args.loop:
                    break
                clip.set(cv2.CAP_PROP_POS_FRAMES, 0)
                loops += 1
                ok, pro_frame = clip.read()
                if not ok:
                    reason = "clip could not be rewound"
                    break

            cv2.imshow(WINDOW, pro_frame)
            projected += 1

            if args.preview and latest[0] is not None \
                    and projected % max(1, args.preview_every) == 0:
                cv2.imshow(PREVIEW, shrink(latest[0], (480, 360)))
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                reason = "stopped by user"
                break

            # Absolute deadlines, so a slow iteration is absorbed rather than added
            # to the next one - that is what keeps the projection at the clip's fps.
            hold_until(next_frame_at)
            next_frame_at = max(next_frame_at + frame_interval, time.perf_counter())
    except KeyboardInterrupt:
        reason = "interrupted"
        print("\ninterrupted.")
    finally:
        stop_event.set()
        grabber.join(timeout=3)
        encoder.join(timeout=10)
        video = recorder.close()
        cam.release()
        clip.release()
        cv2.destroyAllWindows()

    elapsed = max(time.time() - t0, 1e-6)
    meta = {
        "started_at": datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
        "elapsed_sec": round(elapsed, 2),
        "stop_reason": reason,
        "video": video,
        "clip": {"path": clip_path, "frames": clip_frames, "fps": round(clip_fps, 2),
                 "size": list(clip_size), "loops": loops},
        "projector": {"screen_index": args.screen,
                      "frames_projected": projected,
                      "fps_projector": round(projected / elapsed, 2),
                      "monitor": dict(zip(monitor._fields, monitor)) if monitor else None},
        "camera": {"index": args.camera, "backend": args.cam_backend,
                   "requested": [args.cam_width, args.cam_height, args.cam_fps],
                   "frames_read": stats["read"], "frames_dropped": stats["dropped"]},
        "warp": {"enabled": bool(args.warp),
                 "points_tl_tr_br_bl": [list(map(int, p)) for p in points]
                 if points else None,
                 "target": list(warp_wh) if warp_wh else None,
                 "homography": matrix.tolist() if matrix is not None else None},
        "rec_size": list(out_size) if out_size else None,
    }
    meta_path = os.path.splitext(args.out)[0] + ".json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

    print("\n" + "=" * 70)
    print(f"done: {reason} after {elapsed:.1f}s")
    print(f"  projector {projected} frame(s) at "
          f"{projected / elapsed:.1f} fps (clip is {clip_fps:.0f} fps)"
          + (f", {loops} loop(s)" if loops else ""))
    print(f"  camera {stats['read']} read, {stats['dropped']} dropped, "
          f"{video['frames']} written")
    if video["error"]:
        print(f"  ERROR: {video['error']}")
    elif not video["size"]:
        print("  ERROR: no camera frame was ever captured, so nothing was written")
    else:
        print(f"  -> {video['path']}  "
              f"{video['size'][0]}x{video['size'][1]} @ {video['fps_header']}fps, "
              f"{video['file_mb']} MB")
    print(f"  -> {meta_path}")
    return 1 if (video["error"] or not video["size"]) else 0


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--clip", default=DEFAULT_CLIP, help="video to project")
    p.add_argument("--background", default=DEFAULT_BACKGROUND,
                   help="image shown before the clip starts")
    p.add_argument("--out", default=None,
                   help=f"output mp4 (default: {DEFAULT_OUT_DIR}/rec_<MMDDHHMMSS>.mp4)")
    p.add_argument("--screen", type=int, default=1,
                   help="monitor index for the fullscreen window (0 = primary; the "
                        "table printed at startup lists them)")
    p.add_argument("--loop", action="store_true",
                   help="restart the clip instead of stopping at its end")
    p.add_argument("--seconds", type=float, default=0.0,
                   help="stop after N seconds (0 = no limit)")
    p.add_argument("--max-frames", type=int, default=0,
                   help="stop after N projected frames (0 = no limit)")
    p.add_argument("--start-delay", type=float, default=0.0,
                   help="seconds to hold the background before recording starts")

    rec = p.add_argument_group("recording")
    rec.add_argument("--fps", type=float, default=0.0,
                     help="mp4 header fps (0 = measure the camera's real rate)")
    rec.add_argument("--fps-probe", type=int, default=24,
                     help="frames used to measure the rate; they are buffered, "
                          "not dropped")
    rec.add_argument("--codec", default="mp4v", help="fourcc, e.g. mp4v, avc1, XVID")
    rec.add_argument("--rec-size", type=int, nargs=2, default=[0, 0], metavar=("W", "H"),
                     help="resize before encoding (0 0 keeps the camera resolution)")
    rec.add_argument("--warp", action="store_true",
                     help="calibrate once and record the rectified screen instead of "
                          "the raw camera frame")
    rec.add_argument("--manual-calib", action="store_true",
                     help="with --warp: click the 4 corners instead of auto-detecting")
    rec.add_argument("--calib-settle", type=float, default=0.8,
                     help="seconds to wait after each calibration flash")
    rec.add_argument("--no-preview", dest="preview", action="store_false",
                     help="skip the small camera preview window")
    rec.add_argument("--preview-every", type=int, default=5,
                     help="refresh the preview every N projected frames")

    cam = p.add_argument_group("camera")
    cam.add_argument("--camera", type=int, default=0, help="webcam index")
    cam.add_argument("--cam-width", type=int, default=1280)
    cam.add_argument("--cam-height", type=int, default=960)
    cam.add_argument("--cam-fps", type=int, default=30)
    cam.add_argument("--cam-backend", default="auto",
                     choices=["auto", "any", "dshow", "msmf", "v4l2"])

    p.set_defaults(preview=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not args.out:
        args.out = os.path.join(DEFAULT_OUT_DIR,
                                f"rec_{datetime.now().strftime('%m%d%H%M%S')}.mp4")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    if len(args.codec) != 4:
        raise SystemExit(f"--codec must be a 4-character fourcc, got '{args.codec}'")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
