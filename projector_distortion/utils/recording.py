"""
RunRecorder - owns everything a run writes to disk.

    output/<run_name>/
      run_meta.json      config, environment, calibration, summary
      detections.csv     one row per detected box, tagged distorted | restored
      calib/             calibration evidence (live runs only)
      warp/              first live frame, before and after the warp, + the figure
      captures/          the footage itself, before any box is drawn on it
      frames/            annotated views + residual, `save_every` apart
      frames_all/        the 2x2 comparison panels, kept apart so they can be
                         flipped through on their own
      result.mp4         2x2 panel

Keeping captures/ un-annotated is what lets evaluate.py score the same run later, or
another detector re-run over the identical restoration.

Restoration quality (PSNR/SSIM against a clean reference) is not measured here -
that needs ground truth and belongs to evaluate.py.
"""

import csv
import json
import os
import platform
import sys
import time
from datetime import datetime

import cv2

FRAME_KINDS = ("beam", "distorted", "distorted_det", "restored", "restored_det",
               "residual", "panel", "raw")

# `beam` stays off by default - the panel already shows it. The un-annotated
# distorted/restored pair is kept, since only that survives a later re-analysis.
DEFAULT_FRAME_KINDS = ("distorted", "restored", "distorted_det", "restored_det",
                       "residual", "panel", "raw")

DETECTION_FIELDS = ["frame_id", "name_id", "source", "cls_id", "name", "conf",
                    "x1", "y1", "x2", "y2"]
# Each kind lands in the directory named here; anything unlisted goes to frames/.
KIND_DIRS = {"distorted": "captures", "restored": "captures", "panel": "frames_all"}

# Measured at jpeg quality 92: 640x360 tiles, 1280x720 panel, full-frame raw.
_KB_PER_KIND = {"beam": 53, "distorted": 53, "distorted_det": 53, "restored": 53,
                "restored_det": 53, "residual": 68, "panel": 210, "raw": 136}


def parse_kinds(text):
    """'restored,panel' -> ('restored', 'panel'); raises on an unknown name."""
    kinds = [k.strip() for k in str(text).split(",") if k.strip()]
    unknown = [k for k in kinds if k not in FRAME_KINDS]
    if unknown:
        raise ValueError(f"unknown image kinds {unknown}; choose from {list(FRAME_KINDS)}")
    return tuple(kinds)


def estimate_footprint_mb(kinds, save_every, total_frames):
    """Rough MB of jpgs for a whole run, per kind rather than a flat average."""
    if not save_every or not kinds or not total_frames:
        return 0.0
    kb = sum(_KB_PER_KIND.get(k, 64) for k in kinds)
    return (total_frames / save_every) * kb / 1024


class RunRecorder:
    """Use as a context manager so csv handles and the video writer always close."""

    def __init__(self, output_dir, save_every=1, frame_kinds=DEFAULT_FRAME_KINDS,
                 jpeg_quality=92, max_saved_frames=0, video_size=None, video_fps=30):
        self.dir = str(output_dir)
        self.save_every = max(0, int(save_every))
        self.frame_kinds = tuple(k for k in frame_kinds if k in FRAME_KINDS)
        self.jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
        self.max_saved_frames = max(0, int(max_saved_frames))

        self.frames_dir = os.path.join(self.dir, "frames")
        self.captures_dir = os.path.join(self.dir, "captures")
        self.panels_dir = os.path.join(self.dir, "frames_all")
        self.calib_dir = os.path.join(self.dir, "calib")
        self.warp_dir = os.path.join(self.dir, "warp")
        os.makedirs(self.dir, exist_ok=True)
        if self.save_every:
            for kind in self.frame_kinds:
                os.makedirs(self._dir_for(kind), exist_ok=True)

        self.meta = {
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "argv": sys.argv[1:],
        }
        self.saved_frames = 0
        self.bytes_written = 0
        self._t0 = time.time()

        self.video_path = None
        self.writer = None
        if video_size:
            self.video_path = os.path.join(self.dir, "result.mp4")
            self.writer = cv2.VideoWriter(self.video_path,
                                          cv2.VideoWriter_fourcc(*"mp4v"),
                                          video_fps, tuple(video_size))
            if not self.writer.isOpened():
                print(f"warning: could not open the video writer at {self.video_path}")
                self.writer = None

        self._det_fh = open(os.path.join(self.dir, "detections.csv"), "w", newline="",
                            encoding="utf-8")
        self._det = csv.writer(self._det_fh)
        self._det.writerow(DETECTION_FIELDS)

    def set(self, **kwargs):
        self.meta.update(kwargs)
        self._flush_meta()

    def _flush_meta(self):
        with open(os.path.join(self.dir, "run_meta.json"), "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2, ensure_ascii=False, default=str)

    def save_calibration(self, points, raw_frame=None, warped=None, debug=None):
        """
        Write the pre-warp view with the quad, the rectified result, and the
        auto-calibration intermediates. Written even when detection failed, which
        is when they matter most.
        """
        from .visualize import draw_quad

        os.makedirs(self.calib_dir, exist_ok=True)
        written = []
        if raw_frame is not None:
            if points:
                written.append(self._imwrite(
                    os.path.join(self.calib_dir, "raw_points.jpg"),
                    draw_quad(raw_frame, points, "pre-warp camera + calibration quad")))
            written.append(self._imwrite(os.path.join(self.calib_dir, "raw_frame.jpg"),
                                         raw_frame))
        if warped is not None:
            written.append(self._imwrite(os.path.join(self.calib_dir, "warped.jpg"),
                                         warped))
        for key in ("shot_black", "shot_white", "diff", "mask", "overlay"):
            img = (debug or {}).get(key)
            if img is not None:
                written.append(self._imwrite(
                    os.path.join(self.calib_dir, f"{key}.jpg"), img))
        return [w for w in written if w]

    def save_warp_pair(self, pre, post, points=None, name="first_frame"):
        """
        One frame's warp input and output, plus the side-by-side figure.

        Live runs call this once, on the first frame off the camera. The calib/
        shots come from the black/white flashes before the loop; these come from
        the run itself, so they show the warp the frames were actually rectified
        with. Returns (written paths, figure) - the caller usually also displays
        the figure.
        """
        from .visualize import draw_quad, warp_before_after

        os.makedirs(self.warp_dir, exist_ok=True)
        figure = warp_before_after(pre, post, points)
        images = {
            "pre_warp": pre,
            "pre_warp_quad": (draw_quad(pre, points, "pre-warp camera + calibration "
                                                     "quad") if points else None),
            "post_warp": post,
            "compare": figure,
        }
        written = [self._imwrite(os.path.join(self.warp_dir, f"{name}_{kind}.jpg"), img)
                   for kind, img in images.items() if img is not None]
        return [w for w in written if w], figure

    def should_save(self, frame_id):
        if not self.save_every or not self.frame_kinds:
            return False
        if self.max_saved_frames and self.saved_frames >= self.max_saved_frames:
            return False
        return frame_id % self.save_every == 0

    def log_detections(self, result):
        """One csv row per detected box. Call for every frame, saved or not."""
        for source, dets in (("distorted", result.det_distorted),
                             ("restored", result.det_restored)):
            for d in dets:
                x1, y1, x2, y2 = (int(v) for v in d.box)
                self._det.writerow([result.frame_id, result.name_id, source, d.cls_id,
                                    d.name, f"{d.conf:.4f}", x1, y1, x2, y2])

    def _dir_for(self, kind):
        return os.path.join(self.dir, KIND_DIRS.get(kind, "frames"))

    def save_frame_images(self, result, panel=None, raw_frame=None):
        """Write the configured kinds for this frame; returns how many were written."""
        images = {
            "beam": result.beam,
            "distorted": result.distorted,
            "distorted_det": result.distorted_det,
            "restored": result.restored,
            "restored_det": result.restored_det,
            "residual": result.residual,
            "panel": panel,
            "raw": raw_frame,
        }
        n = 0
        for kind in self.frame_kinds:
            img = images.get(kind)
            if img is None:
                continue
            path = os.path.join(self._dir_for(kind), f"{result.name_id}_{kind}.jpg")
            if self._imwrite(path, img):
                n += 1
        if n:
            self.saved_frames += 1
        return n

    def write_video(self, panel):
        if self.writer is not None:
            self.writer.write(panel)

    def _imwrite(self, path, img):
        if not cv2.imwrite(path, img, self.jpeg_params):
            print(f"warning: failed to write {path}")
            return None
        self.bytes_written += os.path.getsize(path)
        return path

    def finish(self, **summary):
        self.meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
        self.meta["elapsed_sec"] = round(time.time() - self._t0, 2)
        self.meta["saved_frames"] = self.saved_frames
        self.meta["image_mb"] = round(self.bytes_written / 1048576, 2)
        if self.video_path and os.path.exists(self.video_path):
            self.meta["video_mb"] = round(os.path.getsize(self.video_path) / 1048576, 2)
        self.meta.update(summary)
        self._flush_meta()
        return self.meta

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        fh = getattr(self, "_det_fh", None)
        if fh and not fh.closed:
            fh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
