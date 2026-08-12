"""
Offline pipeline: restore + detect over image pairs on disk. No hardware needed.

For each (distorted, light) pair:

    restored = restorer(distorted, light)
    detector(distorted)  -> "distorted" detections  (before restoration)
    detector(restored)   -> "restored" detections   (after)

Un-annotated and annotated images are kept separate, so PSNR/SSIM against the surface
ground truth and re-detection with another backend remain possible after the run.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..data import Sample, find_samples, load_labels
from ..models import filter_detections
from ..utils.image import psnr, read_bgr, resize, ssim
from ..utils.visualize import draw_detections, grid_2x2


@dataclass
class FrameResult:
    """One processed pair. Field names match what RunRecorder expects."""

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
    surface: Optional[np.ndarray] = None
    gt_boxes: List = field(default_factory=list)

    def metrics(self) -> Dict[str, Optional[float]]:
        """PSNR/SSIM of distorted vs surface and restored vs surface; None without GT."""
        if self.surface is None:
            return {"psnr_distorted": None, "psnr_restored": None,
                    "ssim_distorted": None, "ssim_restored": None}
        ref = resize(self.surface, (self.restored.shape[1], self.restored.shape[0]))
        return {
            "psnr_distorted": psnr(ref, self.distorted),
            "psnr_restored": psnr(ref, self.restored),
            "ssim_distorted": ssim(ref, self.distorted),
            "ssim_restored": ssim(ref, self.restored),
        }


def process_sample(sample: Sample, restorer, detector, frame_id=0,
                   min_area=500, min_width=20, min_height=20,
                   with_gt=True) -> FrameResult:
    """Run one (distorted, light) pair end to end."""
    distorted = read_bgr(sample.distorted)
    light = read_bgr(sample.light)
    size = tuple(restorer.input_size)

    t0 = time.perf_counter()
    restored_plain, residual, residual_mean = restorer.restore_full(distorted, light)
    t_restore = time.perf_counter() - t0

    distorted_plain = resize(distorted, size)

    t1 = time.perf_counter()
    raw_distorted = detector(distorted_plain)
    raw_restored = detector(restored_plain)
    t_detect = time.perf_counter() - t1

    gate = dict(min_area=min_area, min_width=min_width, min_height=min_height)
    det_c = filter_detections(raw_distorted, **gate)
    det_r = filter_detections(raw_restored, **gate)

    distorted_det = draw_detections(distorted_plain.copy(), det_c)
    restored_det = draw_detections(restored_plain.copy(), det_r)

    surface_img, gt_boxes = None, []
    if with_gt and sample.surface:
        surface_img = resize(read_bgr(sample.surface), size)
        # LabelMe labels name their classes, so they need the detector's own id order.
        gt_boxes = load_labels(sample.label, size[0], size[1],
                               class_names=getattr(detector, "class_names", None))

    return FrameResult(
        frame_id=frame_id, name_id=sample.name_id,
        light=resize(light, size),
        distorted=distorted_plain, distorted_det=distorted_det,
        restored=restored_plain, restored_det=restored_det,
        residual=residual, residual_mean=residual_mean,
        det_distorted=det_c, det_restored=det_r,
        t_restore=t_restore, t_detect=t_detect,
        surface=surface_img, gt_boxes=gt_boxes,
    )


def build_panel(result: FrameResult, detector_name="detector") -> np.ndarray:
    """2x2: light | distorted+det / restored+det | residual."""
    return grid_2x2(
        result.light, result.distorted_det, result.restored_det, result.residual,
        labels=[
            "light (projected source)",
            f"distorted + {detector_name} ({len(result.det_distorted)})",
            f"restored + {detector_name} ({len(result.det_restored)})",
            "residual",
        ],
    )


def run_offline(input_root, restorer, detector, recorder, limit=0,
                min_area=500, min_width=20, min_height=20,
                detector_name="detector", progress=True) -> Dict:
    """
    Restore and detect over every discovered pair, writing through `recorder`.

    Restoration quality is deliberately not scored here - that needs surface ground
    truth and is evaluate.py's job.
    """
    samples = find_samples(input_root, limit=limit)
    print(f"found {len(samples)} distorted/light pair(s)")

    totals = {"distorted": 0, "restored": 0}
    sums = {"residual": 0.0, "t_restore": 0.0, "t_detect": 0.0}
    t0 = time.time()

    iterator = enumerate(samples)
    if progress:
        try:
            from tqdm import tqdm
            iterator = enumerate(tqdm(samples, desc="restore+detect", unit="img"))
        except ImportError:
            pass

    for i, sample in iterator:
        result = process_sample(sample, restorer, detector, frame_id=i,
                                min_area=min_area, min_width=min_width,
                                min_height=min_height, with_gt=False)
        panel = build_panel(result, detector_name)

        if recorder.should_save(i):
            recorder.save_frame_images(result, panel=panel)
        recorder.log_detections(result)
        recorder.write_video(panel)

        totals["distorted"] += len(result.det_distorted)
        totals["restored"] += len(result.det_restored)
        sums["residual"] += result.residual_mean
        sums["t_restore"] += result.t_restore
        sums["t_detect"] += result.t_detect

    n = max(len(samples), 1)
    summary = {
        "images": len(samples),
        "elapsed_sec": round(time.time() - t0, 2),
        "detections_total": totals,
        "detections_per_image": {k: round(v / n, 3) for k, v in totals.items()},
        "residual_mean": round(sums["residual"] / n, 5),
        "avg_restore_ms": round(sums["t_restore"] / n * 1000, 2),
        "avg_detect_ms": round(sums["t_detect"] / n * 1000, 2),
    }
    if totals["distorted"]:
        delta = (totals["restored"] - totals["distorted"]) / totals["distorted"] * 100
        summary["detection_delta_pct"] = round(delta, 1)
    return summary
