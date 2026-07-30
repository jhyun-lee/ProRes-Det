#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Score detection and restoration before vs after, against data/sample_gt.

    python evaluate.py
    python evaluate.py --detector ssd --conf 0.3
    python evaluate.py --detectors yolo,ssd            # one row per backend
    python evaluate.py --iou 0.75 --output output/eval

Reports, for the captured (distorted) image and the restored one:

    detection   precision / recall / F1 / mAP@IoU, per class and overall
    restoration PSNR and SSIM against the clean ground truth

mAP here is the single-IoU-threshold average precision over classes (VOC-style area
under the interpolated PR curve), computed from this run's detections only - it is not
COCO's averaged-over-IoU metric.
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from projector_distortion.cli import (  # noqa: E402
    DEFAULT_GT, DEFAULT_INPUT, DEFAULT_OUTPUT, add_common_args, box_filter_kwargs,
    build_models, run_dir,
)
from projector_distortion.config import resolve_path  # noqa: E402
from projector_distortion.data import find_samples  # noqa: E402
from projector_distortion.pipeline import process_sample  # noqa: E402
from projector_distortion.utils.image import iou as box_iou  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=DEFAULT_INPUT, help="folder of pro/beam pairs")
    p.add_argument("--gt", default=DEFAULT_GT,
                   help="ground truth folder (clean/ + labels/)")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="where the report goes")
    p.add_argument("--name", default=None, help="run folder name")
    p.add_argument("--limit", type=int, default=0, help="score at most N pairs")
    p.add_argument("--iou", type=float, default=0.5, help="IoU for a true positive")
    p.add_argument("--detectors", default=None,
                   help="comma separated backends to compare, e.g. yolo,ssd")
    p.add_argument("--best-per-class", action="store_true")
    add_common_args(p)
    return p


# --- metrics ------------------------------------------------------------------

def match_detections(dets, gt, iou_thr):
    """
    Greedy highest-confidence-first matching within a class.

    Returns (tp_flags aligned with the confidence-sorted dets, n_gt).
    """
    order = sorted(range(len(dets)), key=lambda i: -dets[i].conf)
    used = [False] * len(gt)
    tp = [0] * len(dets)
    for rank, i in enumerate(order):
        d = dets[i]
        best_j, best_iou = -1, 0.0
        for j, (gcls, gbox) in enumerate(gt):
            if used[j] or gcls != d.cls_id:
                continue
            v = box_iou(d.box, gbox)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0 and best_iou >= iou_thr:
            used[best_j] = True
            tp[rank] = 1
    return tp, len(gt), [dets[i].conf for i in order], [dets[i].cls_id for i in order]


def average_precision(tp_flags, confs, n_gt):
    """VOC-style AP: area under the interpolated precision/recall curve."""
    if n_gt == 0:
        return float("nan")
    if not tp_flags:
        return 0.0
    order = np.argsort(-np.asarray(confs, dtype=float))
    tp = np.asarray(tp_flags, dtype=float)[order]
    fp = 1.0 - tp
    tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
    recall = tp_c / n_gt
    precision = tp_c / np.maximum(tp_c + fp_c, 1e-12)
    # make precision monotonically decreasing, then integrate over recall
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    r = np.concatenate([[0.0], recall])
    p = np.concatenate([[precision[0] if len(precision) else 0.0], precision])
    return float(np.sum(np.diff(r) * p[1:]))


class Scorer:
    """Accumulates per-class TP/FP/GT and confidences for one source."""

    def __init__(self, class_names):
        self.class_names = list(class_names)
        self.per_class = defaultdict(lambda: {"tp": [], "conf": [], "n_gt": 0})
        self.images = 0

    def add(self, dets, gt, iou_thr):
        self.images += 1
        by_cls_gt = defaultdict(list)
        for cls_id, box in gt:
            by_cls_gt[cls_id].append((cls_id, box))
        by_cls_det = defaultdict(list)
        for d in dets:
            by_cls_det[d.cls_id].append(d)

        for cls_id in set(by_cls_gt) | set(by_cls_det):
            tp, n_gt, confs, _ = match_detections(
                by_cls_det.get(cls_id, []), by_cls_gt.get(cls_id, []), iou_thr)
            slot = self.per_class[cls_id]
            slot["tp"].extend(tp)
            slot["conf"].extend(confs)
            slot["n_gt"] += n_gt

    def report(self):
        rows, aps = [], []
        tot_tp = tot_fp = tot_gt = 0
        for cls_id in sorted(self.per_class):
            s = self.per_class[cls_id]
            tp = int(sum(s["tp"]))
            fp = len(s["tp"]) - tp
            n_gt = s["n_gt"]
            fn = max(0, n_gt - tp)
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / n_gt if n_gt else float("nan")
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
            ap = average_precision(s["tp"], s["conf"], n_gt)
            if not np.isnan(ap):
                aps.append(ap)
            tot_tp, tot_fp, tot_gt = tot_tp + tp, tot_fp + fp, tot_gt + n_gt
            name = (self.class_names[cls_id] if cls_id < len(self.class_names)
                    else str(cls_id))
            rows.append({"cls_id": cls_id, "name": name, "n_gt": n_gt, "tp": tp,
                         "fp": fp, "fn": fn, "precision": round(prec, 4),
                         "recall": None if np.isnan(rec) else round(rec, 4),
                         "f1": round(f1, 4),
                         "ap": None if np.isnan(ap) else round(ap, 4)})

        prec = tot_tp / (tot_tp + tot_fp) if tot_tp + tot_fp else 0.0
        rec = tot_tp / tot_gt if tot_gt else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        overall = {"images": self.images, "n_gt": tot_gt, "tp": tot_tp, "fp": tot_fp,
                   "fn": max(0, tot_gt - tot_tp), "precision": round(prec, 4),
                   "recall": round(rec, 4), "f1": round(f1, 4),
                   "mAP": round(float(np.mean(aps)), 4) if aps else None}
        return overall, rows


# --- main ---------------------------------------------------------------------

def evaluate_one(backend, args, out_dir):
    """Score a single detector backend; returns the result dict."""
    args.detector = backend
    args.det_weights = None if backend != args.detector else args.det_weights
    restorer, detector, info = build_models(args, need_detector=True)
    filt = box_filter_kwargs(args, info["detection_config"])
    class_names = detector.class_names

    samples = find_samples(resolve_path(args.input), gt_root=resolve_path(args.gt),
                           limit=args.limit)
    scored = [s for s in samples if s.label and s.clean]
    if not scored:
        raise SystemExit(
            f"no sample has both a clean image and a label under {args.gt}\n"
            f"    expected {args.gt}/clean/Ori<id>.jpg and {args.gt}/labels/Ori<id>.txt")
    print(f"scoring {len(scored)} of {len(samples)} pair(s) that have ground truth")

    sc_cap, sc_res = Scorer(class_names), Scorer(class_names)
    quality, rows = [], []

    iterator = enumerate(scored)
    try:
        from tqdm import tqdm
        iterator = enumerate(tqdm(scored, desc=f"evaluate [{backend}]", unit="img"))
    except ImportError:
        pass

    for i, sample in iterator:
        r = process_sample(sample, restorer, detector, frame_id=i, **filt)
        sc_cap.add(r.det_captured, r.gt_boxes, args.iou)
        sc_res.add(r.det_restored, r.gt_boxes, args.iou)
        m = r.metrics()
        quality.append(m)
        rows.append({"name_id": r.name_id, "n_gt": len(r.gt_boxes),
                     "n_captured": len(r.det_captured),
                     "n_restored": len(r.det_restored),
                     **{k: (None if v is None else round(v, 4)) for k, v in m.items()}})

    cap_overall, cap_rows = sc_cap.report()
    res_overall, res_rows = sc_res.report()
    q = {k: round(float(np.mean([x[k] for x in quality])), 4) for k in quality[0]} \
        if quality else {}

    result = {
        "detector": detector.info(),
        "restorer": restorer.info(),
        "iou_threshold": args.iou,
        "box_filter": filt,
        "images_scored": len(scored),
        "detection": {"captured": cap_overall, "restored": res_overall},
        "detection_per_class": {"captured": cap_rows, "restored": res_rows},
        "restoration": q,
    }
    if cap_overall["mAP"] is not None and res_overall["mAP"] is not None:
        result["detection"]["mAP_gain"] = round(
            res_overall["mAP"] - cap_overall["mAP"], 4)
    if q:
        result["restoration"]["psnr_gain_db"] = round(
            q["psnr_restored"] - q["psnr_captured"], 4)
        result["restoration"]["ssim_gain"] = round(
            q["ssim_restored"] - q["ssim_captured"], 5)

    with open(os.path.join(out_dir, f"per_image_{backend}.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir, f"per_class_{backend}.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source"] + list(cap_rows[0]))
        w.writeheader()
        for src, rs in (("captured", cap_rows), ("restored", res_rows)):
            for r in rs:
                w.writerow({"source": src, **r})
    return result


def main(argv=None):
    args = build_parser().parse_args(argv)
    out_dir = run_dir(args.output, args.name or "eval")
    backends = [b.strip() for b in (args.detectors or args.detector or "yolo").split(",")
                if b.strip()]

    results = {}
    for backend in backends:
        print(f"\n{'=' * 70}\n{backend}\n{'=' * 70}")
        results[backend] = evaluate_one(backend, args, out_dir)

    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    _print_table(results)
    print(f"\n-> {out_dir}\n   report.json | per_image_*.csv | per_class_*.csv")
    return 0


def _print_table(results):
    print("\n" + "=" * 78)
    print("detection (captured = before restoration, restored = after)")
    print("=" * 78)
    hdr = f"{'backend':10s} {'source':10s} {'P':>7s} {'R':>7s} {'F1':>7s} {'mAP':>7s} " \
          f"{'TP':>5s} {'FP':>5s} {'FN':>5s}"
    print(hdr)
    print("-" * len(hdr))
    for backend, r in results.items():
        for src in ("captured", "restored"):
            d = r["detection"][src]
            m = "  n/a" if d["mAP"] is None else f"{d['mAP']:.4f}"
            print(f"{backend:10s} {src:10s} {d['precision']:7.4f} {d['recall']:7.4f} "
                  f"{d['f1']:7.4f} {m:>7s} {d['tp']:5d} {d['fp']:5d} {d['fn']:5d}")
        gain = r["detection"].get("mAP_gain")
        if gain is not None:
            print(f"{'':10s} {'gain':10s} {'':7s} {'':7s} {'':7s} {gain:+7.4f}")

    q = next((r["restoration"] for r in results.values() if r["restoration"]), None)
    if q:
        print("\n" + "=" * 78)
        print("restoration quality vs clean ground truth")
        print("=" * 78)
        print(f"  PSNR  captured {q['psnr_captured']:7.3f} dB -> "
              f"restored {q['psnr_restored']:7.3f} dB   "
              f"gain {q.get('psnr_gain_db', 0):+.3f}")
        print(f"  SSIM  captured {q['ssim_captured']:7.4f}    -> "
              f"restored {q['ssim_restored']:7.4f}      "
              f"gain {q.get('ssim_gain', 0):+.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
