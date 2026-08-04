#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Restore projector distortion, then detect objects - in one command.

Offline (default, no hardware):
    python demo.py
    python demo.py --input data/sample_input --output output --save-every 1
    python demo.py --detector ssd --conf 0.4
    python demo.py --detector none                  # restoration only

Live rig (webcam + projector):
    python demo.py --live --screen 2
    python demo.py --live --screen 2 --save-every 30 --debug-view

Restore and detect only. Scoring the result against ground truth - detection mAP and
restoration PSNR/SSIM - is evaluate.py's job.

Each run writes output/<run>/ with run_meta.json, detections.csv, the un-annotated
captures under captures/ and the 2x2 comparison panels under frames_all/ - the panel
already carries the annotated views and the residual. Live runs add calib/ - the
quad that was found and the first frame before and after rectification.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from projector_distortion.cli import (  # noqa: E402
    DEFAULT_INPUT, DEFAULT_LIVE_BG, DEFAULT_LIVE_VIDEO, DEFAULT_OUTPUT,
    add_common_args, box_filter_kwargs, build_models, run_dir,
)
from projector_distortion.config import resolve_path  # noqa: E402
from projector_distortion.models.restoration import add_ablation_args  # noqa: E402
from projector_distortion.utils.recording import (  # noqa: E402
    DEFAULT_FRAME_KINDS, FRAME_KINDS, RunRecorder, estimate_footprint_mb, parse_kinds,
)
from projector_distortion.utils.visualize import panel_size  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--input", default=DEFAULT_INPUT,
                   help="folder of pro/beam pairs (offline mode)")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="where run folders go")
    p.add_argument("--name", default=None, help="run folder name (default: timestamp)")
    p.add_argument("--limit", type=int, default=0, help="process at most N pairs")
    p.add_argument("--best-per-class", action="store_true",
                   help="keep only the top box per class (original demo behaviour)")

    r = p.add_argument_group("recording")
    r.add_argument("--save-every", type=int, default=1,
                   help="write images every N frames (0 = none; csv is always complete)")
    r.add_argument("--save-kinds", default=",".join(DEFAULT_FRAME_KINDS),
                   help=f"subset of {','.join(FRAME_KINDS)} "
                        f"(default: {','.join(DEFAULT_FRAME_KINDS)})")
    r.add_argument("--max-saved-frames", type=int, default=0, help="cap on saved frames")
    r.add_argument("--jpeg-quality", type=int, default=92)
    r.add_argument("--video", action="store_true",
                   help="also write result.mp4 of the 2x2 panels (offline mode)")

    live = p.add_argument_group("live mode")
    live.add_argument("--live", action="store_true", help="drive a webcam + projector")
    live.add_argument("--clip", default=DEFAULT_LIVE_VIDEO, help="clip to project")
    live.add_argument("--background", default=DEFAULT_LIVE_BG,
                      help="image shown during calibration")
    live.add_argument("--camera", type=int, default=0, help="webcam index")
    live.add_argument("--screen", type=int, default=1,
                      help="monitor index for the fullscreen window (0 = primary; the "
                           "table printed at startup lists them)")
    live.add_argument("--offset", type=int, default=6,
                      help="projector->camera latency, in frames")
    live.add_argument("--manual-calib", action="store_true",
                      help="click the 4 corners instead of auto-detecting them")
    live.add_argument("--debug-view", action="store_true",
                      help="live window with the pre-warp camera feed + the quad")
    live.add_argument("--cam-width", type=int, default=1280)
    live.add_argument("--cam-height", type=int, default=960)
    live.add_argument("--cam-fps", type=int, default=30)
    live.add_argument("--cam-backend", default="auto",
                      choices=["auto", "any", "dshow", "msmf", "v4l2"])
    live.add_argument("--calib-settle", type=float, default=0.8,
                      help="seconds to wait after each calibration flash")
    live.add_argument("--max-frames", type=int, default=0,
                      help="0 = until the clip ends")
    live.add_argument("--analyse-every", type=int, default=0, metavar="N",
                      help="restore+detect every Nth projected frame (0 = the densest "
                           "N that stays perfectly evenly spaced). 1 analyses the most "
                           "frames but cannot keep every slot, so the spacing gets "
                           "slightly uneven. The projector plays at the clip's own fps "
                           "either way.")

    add_common_args(p)
    # Only needed when the checkpoint is a bare state_dict with no embedded config.
    add_ablation_args(p, capacity=False)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    kinds = parse_kinds(args.save_kinds)

    restorer, detector, info = build_models(args, need_detector=True)
    box_filter = box_filter_kwargs(args, info["detection_config"])
    out_dir = run_dir(args.output, args.name)

    want_video = args.live or args.video
    video_size = panel_size(*info["input_size"])

    with RunRecorder(out_dir, save_every=args.save_every, frame_kinds=kinds,
                     jpeg_quality=args.jpeg_quality,
                     max_saved_frames=args.max_saved_frames,
                     video_size=video_size if want_video else None) as rec:
        rec.set(mode="live" if args.live else "offline",
                restorer=restorer.info(),
                detector=detector.info(),
                box_filter=box_filter, device=info["device"],
                recording={"save_every": args.save_every, "frame_kinds": list(kinds),
                           "jpeg_quality": args.jpeg_quality,
                           "max_saved_frames": args.max_saved_frames,
                           "video": bool(want_video)})
        print(f"output: {out_dir}")

        if args.live:
            summary = _run_live(args, restorer, detector, rec, box_filter,
                                info["input_size"])
        else:
            summary = _run_offline(args, restorer, detector, rec, box_filter, kinds)

        meta = rec.finish(**summary)

    _report(meta, summary, detector, out_dir)
    return 0


def _run_live(args, restorer, detector, rec, box_filter, input_size):
    import cv2

    from projector_distortion.pipeline.live import run_live  # noqa: F401

    bg_path = resolve_path(args.background)
    background = cv2.imread(bg_path) if bg_path else None
    if background is None:
        print(f"warning: background image not found ({args.background}); "
              f"using a flat grey frame.")
    clip = resolve_path(args.clip)
    if not clip or not os.path.exists(clip):
        raise SystemExit(f"projector clip not found: {args.clip}")

    common = dict(background=background, camera=args.camera, screen=args.screen,
                  max_frames=args.max_frames, cam_fps=args.cam_fps,
                  cam_size=(args.cam_width, args.cam_height),
                  cam_backend=args.cam_backend, detector_name=detector.name)

    return run_live(
        clip, restorer, detector, rec, offset=args.offset,
        manual_calib=args.manual_calib, debug_view=args.debug_view,
        calib_settle=args.calib_settle, analyse_every=args.analyse_every,
        **common, **box_filter)


def _run_offline(args, restorer, detector, rec, box_filter, kinds):
    from projector_distortion.pipeline import run_offline

    input_root = resolve_path(args.input)
    if not input_root or not os.path.isdir(input_root):
        raise SystemExit(
            f"input folder not found: {args.input}\n"
            f"    expected pro/ and beam/ subfolders; see data/README_data.md")

    pro_dir = os.path.join(input_root, "pro")
    n_pairs = len(os.listdir(pro_dir)) if os.path.isdir(pro_dir) else 0
    budget = estimate_footprint_mb(kinds, args.save_every, n_pairs)
    print(f"    images every {args.save_every or '-'} frames x {len(kinds)} "
          f"kinds ~ {budget:.0f} MB")

    return run_offline(input_root, restorer, detector, rec,
                       limit=args.limit, detector_name=detector.name, **box_filter)


def _report(meta, summary, detector, out_dir):
    print("\n" + "=" * 70)
    n = summary.get("images") or summary.get("frames_processed") or 0
    print(f"done: {n} frame(s) in {meta.get('elapsed_sec')}s")
    if "fps_projector" in summary:
        print(f"  projector {summary['fps_projector']:.1f} fps "
              f"({summary['frames_projected']} frames) | "
              f"analysis {summary['fps_end_to_end']:.1f} fps "
              f"(every {summary.get('analyse_every', 1)} frame(s): "
              f"{summary['frames_processed']} analysed, "
              f"{summary.get('frames_skipped', 0)} skipped, "
              f"{summary.get('frames_dropped', 0)} dropped)")
    print(f"  restore {summary.get('avg_restore_ms')} ms/frame | "
          f"detect {summary.get('avg_detect_ms')} ms/frame | "
          f"mean |residual| {summary.get('residual_mean')}")

    t = summary.get("detections_total", {})
    if detector.name != "none" and t:
        per = (summary.get("detections_per_image")
               or summary.get("detections_per_frame", {}))
        delta = summary.get("detection_delta_pct")
        print(f"  {detector.name} boxes: distorted {t.get('distorted')} "
              f"({per.get('distorted')}/frame) -> restored {t.get('restored')} "
              f"({per.get('restored')}/frame)"
              + (f"   {delta:+.1f}%" if delta is not None else ""))

    print(f"  saved {meta.get('saved_frames')} frame set(s), {meta.get('image_mb')} MB"
          + (f" + {meta['video_mb']} MB video" if "video_mb" in meta else "")
          + (f"   ({summary['writes_dropped']} frame(s) outran the writer)"
             if summary.get("writes_dropped") else ""))
    print(f"  -> {out_dir}")
    print("     run_meta.json | detections.csv | captures/ | frames_all/"
          + (" | calib/" if meta.get("mode") == "live" else ""))
    print("     score it with:  python evaluate.py")


if __name__ == "__main__":
    raise SystemExit(main())
