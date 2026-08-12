#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Score detection and restoration before vs after, against a labelled split.

    python evaluate.py                                 # data/SampleData/sample_eval
    python evaluate.py --detector ssd --conf 0.3
    python evaluate.py --detector yolo,ssd             # one row per backend
    python evaluate.py --iou 0.75 --output output/eval
    python evaluate.py --input data/SampleData/sample_test \
                       --gt    data/SampleData/sample_test   # the held-out split

Both labelled splits live under data/SampleData: sample_eval (22 pairs, YOLO txt
labels) is the default, sample_test (200 pairs, LabelMe json labels) is held out.
Either format is read directly, so --gt only ever needs the split root.

Reports, for the distorted image and the restored one:

    detection   precision / recall / F1 / mAP@IoU, per class and overall
    restoration PSNR and SSIM against the surface ground truth

mAP here is the single-IoU-threshold average precision over classes (VOC-style area
under the interpolated PR curve), computed from this run's detections only - it is
not COCO's averaged-over-IoU metric.
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
    DEFAULT_GT, DEFAULT_INPUT, DEFAULT_OUTPUT, DETECTOR_HELP, add_common_args,
    box_filter_kwargs, build_models, run_dir,
)
from projector_distortion.config import resolve_path  # noqa: E402
from projector_distortion.data import find_samples  # noqa: E402
from projector_distortion.pipeline import process_sample  # noqa: E402
from projector_distortion.utils.image import iou as box_iou  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=DEFAULT_INPUT,
                   help="folder of distorted/light pairs")
    p.add_argument("--gt", default=DEFAULT_GT,
                   help="ground truth folder (surface/ + labels/)")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="where the report goes")
    p.add_argument("--name", default=None, help="run folder name")
    p.add_argument("--limit", type=int, default=0, help="score at most N pairs")
    p.add_argument("--iou", type=float, default=0.5, help="IoU for a true positive")
    add_common_args(p, detector_help=DETECTOR_HELP + ". Takes a comma separated list "
                                     "here (yolo,ssd) for one row per backend")
    return p


def match_detections(dets, gt_boxes, iou_thr):
    """
    Greedy highest-confidence-first matching, one class at a time.

    Both arguments are already a single class's worth - `Scorer.add` splits them -
    so nothing here compares class ids. Returns (tp_flags, n_gt, confs), all aligned
    with the confidence-sorted order of `dets`.
    """
    order = sorted(range(len(dets)), key=lambda i: -dets[i].conf)
    used = [False] * len(gt_boxes)
    tp = [0] * len(dets)
    for rank, i in enumerate(order):
        best_j, best_iou = -1, 0.0
        for j, gbox in enumerate(gt_boxes):
            if used[j]:
                continue
            v = box_iou(dets[i].box, gbox)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0 and best_iou >= iou_thr:
            used[best_j] = True
            tp[rank] = 1
    return tp, len(gt_boxes), [dets[i].conf for i in order]


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
    for i in range(len(precision) - 2, -1, -1):     # make precision monotonic
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
        """`gt` is [(cls_id, box), ...]. Both sides are split by class, then matched."""
        self.images += 1
        by_cls_gt, by_cls_det = defaultdict(list), defaultdict(list)
        for cls_id, box in gt:
            by_cls_gt[cls_id].append(box)
        for d in dets:
            by_cls_det[d.cls_id].append(d)

        for cls_id in set(by_cls_gt) | set(by_cls_det):
            tp, n_gt, confs = match_detections(
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


def evaluate_one(backend, args, out_dir, det_weights):
    """Score a single detector backend; returns the result dict."""
    args.detector = backend
    args.det_weights = det_weights

    restorer, detector, info = build_models(args, need_detector=True)
    box_filter = box_filter_kwargs(args, info["detection_config"])
    class_names = detector.class_names

    samples = find_samples(resolve_path(args.input), gt_root=resolve_path(args.gt),
                           limit=args.limit)
    scored = [s for s in samples if s.label and s.surface]
    if not scored:
        raise SystemExit(
            f"no sample has both a surface image and a label under {args.gt}\n"
            f"    expected {args.gt}/surface/surface_<id>.jpg and "
            f"{args.gt}/labels/surface_<id>.txt (or .json)")
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
        r = process_sample(sample, restorer, detector, frame_id=i, **box_filter)
        sc_cap.add(r.det_distorted, r.gt_boxes, args.iou)
        sc_res.add(r.det_restored, r.gt_boxes, args.iou)
        m = r.metrics()
        quality.append(m)
        rows.append({"name_id": r.name_id, "n_gt": len(r.gt_boxes),
                     "n_distorted": len(r.det_distorted),
                     "n_restored": len(r.det_restored),
                     **{k: (None if v is None else round(v, 4)) for k, v in m.items()}})

    cap_overall, cap_rows = sc_cap.report()
    res_overall, res_rows = sc_res.report()
    q = ({k: round(float(np.mean([x[k] for x in quality])), 4) for k in quality[0]}
         if quality else {})

    result = {
        "detector": detector.info(),
        "restorer": restorer.info(),
        "iou_threshold": args.iou,
        "box_filter": box_filter,
        "images_scored": len(scored),
        "detection": {"distorted": cap_overall, "restored": res_overall},
        "detection_per_class": {"distorted": cap_rows, "restored": res_rows},
        "restoration": q,
    }
    if cap_overall["mAP"] is not None and res_overall["mAP"] is not None:
        result["detection"]["mAP_gain"] = round(
            res_overall["mAP"] - cap_overall["mAP"], 4)
    if q:
        result["restoration"]["psnr_gain_db"] = round(
            q["psnr_restored"] - q["psnr_distorted"], 4)
        result["restoration"]["ssim_gain"] = round(
            q["ssim_restored"] - q["ssim_distorted"], 5)

    _write_csv(os.path.join(out_dir, f"per_image_{backend}.csv"),
               list(rows[0]), rows)
    _write_csv(os.path.join(out_dir, f"per_class_{backend}.csv"),
               ["source"] + list(cap_rows[0]),
               [{"source": src, **r} for src, rs in (("distorted", cap_rows),
                                                     ("restored", res_rows))
                for r in rs])
    return result


def _write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _weights_per_backend(backends, explicit):
    """
    --det-weights names one checkpoint, so it can apply to at most one backend.

    Across several it is ignored and every backend falls back to
    configs/detection.yaml; otherwise comparing yolo against ssd would load one
    backend's checkpoint into the other.
    """
    if explicit and len(backends) == 1:
        return {backends[0]: explicit}
    if explicit:
        print(f"warning: --det-weights names one checkpoint but {len(backends)} "
              f"backends are being compared; ignoring it. Per-backend paths come "
              f"from configs/detection.yaml.")
    return {b: None for b in backends}


def _eval_dir_name(input_root):
    """`.../sample_eval` -> `Eval_sample_eval`, so a report is named by its dataset."""
    return "Eval_" + os.path.basename(os.path.normpath(str(input_root)))


def _clear_stale_reports(out_dir):
    """
    Drop per-backend csvs left by an earlier run of the same dataset.

    Without this, scoring yolo and later ssd into the same folder leaves
    `per_class_yolo.csv` behind while report.json only mentions ssd.
    """
    for name in os.listdir(out_dir):
        if name.startswith(("per_class_", "per_image_")) and name.endswith(".csv"):
            os.remove(os.path.join(out_dir, name))


def main(argv=None):
    args = build_parser().parse_args(argv)
    input_root = resolve_path(args.input) or args.input
    out_dir = run_dir(args.output, args.name or _eval_dir_name(input_root))
    _clear_stale_reports(out_dir)
    # --detector takes a comma separated list here, so one run can compare backends.
    backends = [b.strip() for b in (args.detector or "yolo").split(",") if b.strip()]
    weights = _weights_per_backend(backends, args.det_weights)

    results = {}
    for backend in backends:
        print(f"\n{'=' * 70}\n{backend}\n{'=' * 70}")
        results[backend] = evaluate_one(backend, args, out_dir, weights[backend])

    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    _print_table(results)
    print(f"\n-> {out_dir}\n   report.json | per_image_*.csv | per_class_*.csv")
    return 0


def _print_table(results):
    print("\n" + "=" * 78)
    print("detection (distorted = before restoration, restored = after)")
    print("=" * 78)
    hdr = f"{'backend':10s} {'source':10s} {'P':>7s} {'R':>7s} {'F1':>7s} {'mAP':>7s} " \
          f"{'TP':>5s} {'FP':>5s} {'FN':>5s}"
    print(hdr)
    print("-" * len(hdr))
    for backend, r in results.items():
        for src in ("distorted", "restored"):
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
        print("restoration quality vs surface ground truth")
        print("=" * 78)
        print(f"  PSNR  distorted {q['psnr_distorted']:7.3f} dB -> "
              f"restored {q['psnr_restored']:7.3f} dB   "
              f"gain {q.get('psnr_gain_db', 0):+.3f}")
        print(f"  SSIM  distorted {q['ssim_distorted']:7.4f}    -> "
              f"restored {q['ssim_restored']:7.4f}      "
              f"gain {q.get('ssim_gain', 0):+.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
