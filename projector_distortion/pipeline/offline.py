"""
Offline pipeline: restore + detect over image pairs on disk. No hardware needed.

For each (pro, beam) pair:

    restored = restorer(pro, beam)
    detector(pro)        -> "captured" detections   (before restoration)
    detector(restored)   -> "restored" detections   (after)

Clean and annotated images are kept separate, so PSNR/SSIM against the clean ground
truth and re-detection with another backend remain possible after the run.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..data import Sample, find_samples, load_yolo_labels
from ..models import filter_detections
from ..utils.image import psnr, read_bgr, resize, ssim
from ..utils.visualize import draw_detections, grid_2x2


@dataclass
class FrameResult:
    """One processed pair. Field names match what RunRecorder expects."""

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

    def metrics(self) -> Dict[str, Optional[float]]:
        """PSNR/SSIM of captured vs clean and restored vs clean; None without GT."""
        if self.clean is None:
            return {"psnr_captured": None, "psnr_restored": None,
                    "ssim_captured": None, "ssim_restored": None}
        ref = resize(self.clean, (self.restored.shape[1], self.restored.shape[0]))
        return {
            "psnr_captured": psnr(ref, self.captured),
            "psnr_restored": psnr(ref, self.restored),
            "ssim_captured": ssim(ref, self.captured),
            "ssim_restored": ssim(ref, self.restored),
        }


def process_sample(sample: Sample, restorer, detector, frame_id=0,
                   best_per_class=False, min_area=500, min_width=20, min_height=20,
                   with_gt=True) -> FrameResult:
    """Run one (pro, beam) pair end to end."""
    pro = read_bgr(sample.pro)
    beam = read_bgr(sample.beam)
    size = tuple(restorer.input_size)

    t0 = time.perf_counter()
    restored_clean, residual, residual_mean = restorer.restore_full(pro, beam)
    t_restore = time.perf_counter() - t0

    captured_clean = resize(pro, size)

    t1 = time.perf_counter()
    raw_captured = detector(captured_clean)
    raw_restored = detector(restored_clean)
    t_detect = time.perf_counter() - t1

    gate = dict(min_area=min_area, min_width=min_width, min_height=min_height,
                best_per_class=best_per_class)
    det_c = filter_detections(raw_captured, **gate)
    det_r = filter_detections(raw_restored, **gate)

    captured_det = draw_detections(captured_clean.copy(), det_c)
    restored_det = draw_detections(restored_clean.copy(), det_r)

    clean_img, gt_boxes = None, []
    if with_gt and sample.clean:
        clean_img = resize(read_bgr(sample.clean), size)
        gt_boxes = load_yolo_labels(sample.label, size[0], size[1])

    return FrameResult(
        frame_id=frame_id, name_id=sample.name_id,
        beam=resize(beam, size),
        captured=captured_clean, captured_det=captured_det,
        restored=restored_clean, restored_det=restored_det,
        residual=residual, residual_mean=residual_mean,
        det_captured=det_c, det_restored=det_r,
        t_restore=t_restore, t_detect=t_detect,
        clean=clean_img, gt_boxes=gt_boxes,
    )


def build_panel(result: FrameResult, detector_name="detector") -> np.ndarray:
    """2x2: beam | captured+det / restored+det | residual."""
    return grid_2x2(
        result.beam, result.captured_det, result.restored_det, result.residual,
        labels=[
            "beam (projected source)",
            f"captured + {detector_name} ({len(result.det_captured)})",
            f"restored + {detector_name} ({len(result.det_restored)})",
            "|residual| (removed light)",
        ],
    )


def run_offline(input_root, restorer, detector, recorder, gt_root=None, limit=0,
                best_per_class=False, min_area=500, min_width=20, min_height=20,
                detector_name="detector", progress=True) -> Dict:
    """Process every discovered pair through `recorder`; returns a summary dict."""
    samples = find_samples(input_root, gt_root=gt_root, limit=limit)
    n_gt = sum(1 for s in samples if s.clean)
    print(f"found {len(samples)} pro/beam pair(s); {n_gt} have clean ground truth")

    totals = {"captured": 0, "restored": 0}
    sums = {"residual": 0.0, "t_restore": 0.0, "t_detect": 0.0}
    metric_rows = []
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
                                best_per_class=best_per_class, min_area=min_area,
                                min_width=min_width, min_height=min_height)
        m = result.metrics()
        panel = build_panel(result, detector_name)

        saved = recorder.should_save(i)
        if saved:
            recorder.save_frame_images(result, panel=panel)
        recorder.log_frame(result, saved=saved,
                           psnr=m["psnr_restored"], ssim=m["ssim_restored"])
        recorder.write_video(panel)

        totals["captured"] += len(result.det_captured)
        totals["restored"] += len(result.det_restored)
        sums["residual"] += result.residual_mean
        sums["t_restore"] += result.t_restore
        sums["t_detect"] += result.t_detect
        if m["psnr_restored"] is not None:
            metric_rows.append(m)

    n = max(len(samples), 1)
    summary = {
        "images": len(samples),
        "with_ground_truth": n_gt,
        "elapsed_sec": round(time.time() - t0, 2),
        "detections_total": totals,
        "detections_per_image": {k: round(v / n, 3) for k, v in totals.items()},
        "residual_mean": round(sums["residual"] / n, 5),
        "avg_restore_ms": round(sums["t_restore"] / n * 1000, 2),
        "avg_detect_ms": round(sums["t_detect"] / n * 1000, 2),
    }
    if totals["captured"]:
        delta = (totals["restored"] - totals["captured"]) / totals["captured"] * 100
        summary["detection_delta_pct"] = round(delta, 1)
    if metric_rows:
        quality = {
            key: round(float(np.mean([r[key] for r in metric_rows])), 4)
            for key in ("psnr_captured", "psnr_restored",
                        "ssim_captured", "ssim_restored")
        }
        quality["psnr_gain_db"] = round(
            quality["psnr_restored"] - quality["psnr_captured"], 4)
        quality["ssim_gain"] = round(
            quality["ssim_restored"] - quality["ssim_captured"], 5)
        summary["quality"] = quality
    return summary
