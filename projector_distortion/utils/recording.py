# projector_distortion/utils/recording.py
"""
RunRecorder - owns everything a run writes to disk.

    output/<run_name>/
      run_meta.json      config, environment, calibration, summary
      detections.csv     one row per box, tagged by source (captured | restored)
      frames.csv         one row per frame / image
      calib/             calibration evidence (live runs only)
      frames/            sampled images, `save_every` apart
      result.mp4         2x2 panel (live runs only)

Images are split into clean and annotated variants. Keeping the clean copy is what
makes PSNR/SSIM and re-running a different detector possible after the fact.
"""

import csv
import json
import os
import platform
import sys
import time
from datetime import datetime

import cv2

#: Every image kind a frame can produce. All are written by default.
FRAME_KINDS = ("beam", "captured", "captured_det", "restored", "restored_det",
               "residual", "panel", "raw")
DEFAULT_FRAME_KINDS = FRAME_KINDS

DETECTION_FIELDS = ["frame_id", "name_id", "source", "cls_id", "name", "conf",
                    "x1", "y1", "x2", "y2"]
FRAME_FIELDS = ["frame_id", "name_id", "t_wall", "n_captured", "n_restored",
                "residual_mean", "t_restore_ms", "t_detect_ms", "psnr", "ssim", "saved"]


def parse_kinds(text):
    """'restored,panel' -> ('restored', 'panel'); raises on an unknown name."""
    kinds = [k.strip() for k in str(text).split(",") if k.strip()]
    unknown = [k for k in kinds if k not in FRAME_KINDS]
    if unknown:
        raise ValueError(f"unknown image kinds {unknown}; choose from {list(FRAME_KINDS)}")
    return tuple(kinds)


#: Measured at jpeg quality 92 on 640x360 tiles (panel is 1280x720, raw is full frame).
_KB_PER_KIND = {"beam": 53, "captured": 53, "captured_det": 53, "restored": 53,
                "restored_det": 53, "residual": 68, "panel": 210, "raw": 136}


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
        self.calib_dir = os.path.join(self.dir, "calib")
        os.makedirs(self.dir, exist_ok=True)
        if self.save_every and self.frame_kinds:
            os.makedirs(self.frames_dir, exist_ok=True)

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

        self._frm_fh = open(os.path.join(self.dir, "frames.csv"), "w", newline="",
                            encoding="utf-8")
        self._frm = csv.writer(self._frm_fh)
        self._frm.writerow(FRAME_FIELDS)

    # ---- metadata --------------------------------------------------------

    def set(self, **kwargs):
        self.meta.update(kwargs)
        self._flush_meta()

    def _flush_meta(self):
        with open(os.path.join(self.dir, "run_meta.json"), "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2, ensure_ascii=False, default=str)

    # ---- calibration (live runs) -----------------------------------------

    def save_calibration(self, points, raw_frame=None, warped=None, debug=None):
        """
        Write the pre-warp camera view with the quad on it, the rectified result, and
        any auto-calibration intermediates. Populated even when detection failed,
        which is when they matter most.
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

    # ---- per frame -------------------------------------------------------

    def should_save(self, frame_id):
        if not self.save_every or not self.frame_kinds:
            return False
        if self.max_saved_frames and self.saved_frames >= self.max_saved_frames:
            return False
        return frame_id % self.save_every == 0

    def log_frame(self, result, saved=False, psnr=None, ssim=None):
        """CSV rows for one frame. Call for every frame, saved or not."""
        for source, dets in (("captured", result.det_captured),
                             ("restored", result.det_restored)):
            for d in dets:
                x1, y1, x2, y2 = (int(v) for v in d.box)
                self._det.writerow([result.frame_id, result.name_id, source, d.cls_id,
                                    d.name, f"{d.conf:.4f}", x1, y1, x2, y2])
        self._frm.writerow([
            result.frame_id, result.name_id, f"{time.time() - self._t0:.3f}",
            len(result.det_captured), len(result.det_restored),
            f"{result.residual_mean:.5f}",
            f"{result.t_restore * 1000:.1f}", f"{result.t_detect * 1000:.1f}",
            "" if psnr is None else f"{psnr:.3f}",
            "" if ssim is None else f"{ssim:.5f}",
            int(saved),
        ])

    def save_frame_images(self, result, panel=None, raw_frame=None):
        """Write the configured kinds for this frame; returns how many were written."""
        images = {
            "beam": result.beam,
            "captured": result.captured,
            "captured_det": result.captured_det,
            "restored": result.restored,
            "restored_det": result.restored_det,
            "residual": result.residual,
            "panel": panel,
            "raw": raw_frame,
        }
        base = os.path.join(self.frames_dir, str(result.name_id))
        n = 0
        for kind in self.frame_kinds:
            img = images.get(kind)
            if img is not None and self._imwrite(f"{base}_{kind}.jpg", img):
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

    # ---- lifecycle -------------------------------------------------------

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
        for attr in ("_det_fh", "_frm_fh"):
            fh = getattr(self, attr, None)
            if fh and not fh.closed:
                fh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
