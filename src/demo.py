#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Restore projector distortion, then detect objects - in one command.

Offline (default, no hardware):
    python demo.py
    python demo.py --input data/SampleData/sample_eval --output output --save-every 1
    python demo.py --input data/SampleData/sample_test # the held-out split

Replay a recording (two mp4s in, one out, no hardware):
    python demo.py --video-pair data/recordings/rec_0828144808.mp4
    python demo.py --video-pair rec.mp4 --no-detect --video-out restored panel
    python demo.py --video-pair rec.mp4 --clip data/live/train_light_1.mp4

Live rig (webcam + projector):
    python demo.py --live
    python demo.py --live --screen 2
    python demo.py --live --save-every 30 --debug-view

Which models run is a property of the install, not of the command: the restoration
backend and checkpoint come from configs/restoration.yaml, and the detector, its
checkpoint and its confidence floor from configs/detection.yaml. Set
`detector.backend: none` there to run restoration only for every run, or pass
--no-detect for one.

The rig itself - which monitor the projector is on, which webcam watches it, and the
projector->camera latency - is configs/live.yaml. --screen and --camera override the
first two for one run, the same way they do on `Data.py capture`; the latency does
not, because it is measured once per rig and does not differ between two runs on it.

Restore and detect only. Scoring the result against ground truth - detection mAP and
restoration PSNR/SSIM - is evaluate.py's job.

Each run writes output/<run>/ with run_meta.json, detections.csv, the un-annotated
captures under captures/ and the 2x2 comparison panels under frames_all/ - the panel
already carries the annotated views and the residual. Live and replay runs add
calib/ - the quad that was found and the first frame before and after rectification -
and replay always writes the video, which is the point of it: result.mp4, or one
result_<kind>.mp4 per view when --video-out names more than one.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from projector_distortion.cli import (  # noqa: E402
    DEFAULT_INPUT, DEFAULT_LIVE_BG, DEFAULT_LIVE_VIDEO, DEFAULT_OUTPUT,
    box_filter_kwargs, build_models, run_dir,
)
from projector_distortion.config import load_config, pick, resolve_path  # noqa: E402
from projector_distortion.pipeline.replay import (  # noqa: E402
    VIDEO_OUT_KINDS, clip_for, replay_fps, video_targets,
)
from projector_distortion.utils.recording import (  # noqa: E402
    FRAME_KINDS, KIND_DIRS, RunRecorder,
)
from projector_distortion.utils.visualize import panel_size  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--input", default=DEFAULT_INPUT,
                   help="folder of distorted/light pairs (offline mode)")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="where run folders go")
    p.add_argument("--limit", type=int, default=0, help="process at most N pairs")

    p.add_argument("--clip", default=None,
                   help=f"the projected clip. Live mode plays it (default: "
                        f"{DEFAULT_LIVE_VIDEO}); replay mode pairs it against the "
                        f"recording, and defaults to whatever the recording's .json "
                        f"says was projected")
    p.add_argument("--no-detect", action="store_true",
                   help="restoration only; same as detector.backend: none in "
                        "configs/detection.yaml, without editing it")

    r = p.add_argument_group("recording")
    r.add_argument("--save-every", type=int, default=1,
                   help="write images every N frames (0 = none; csv is always complete)")
    r.add_argument("--video", action="store_true",
                   help="also write result.mp4 of the 2x2 panels (offline mode)")

    rep = p.add_argument_group("replay mode")
    rep.add_argument("--video-pair", default=None,
                     help="a `Data.py record` mp4 to replay against --clip; writes "
                          "result.mp4 and needs no hardware")
    rep.add_argument("--video-out", default=["panel"], nargs="+",
                     choices=VIDEO_OUT_KINDS, metavar="KIND",
                     help="what result.mp4 holds: panel (2x2) or restored (the "
                          "restored frame alone). Name both to write one video each, "
                          "off the same pass: "
                          "--video-out restored panel -> result_restored.mp4 + "
                          "result_panel.mp4 (default: panel)")
    rep.add_argument("--no-review", action="store_true",
                     help="skip the calibration verdict window (default: ask, the "
                          "same as rig.review_calibration does live)")

    live = p.add_argument_group("live mode")
    live.add_argument("--live", action="store_true", help="drive a webcam + projector")
    live.add_argument("--screen", type=int, default=None,
                      help="monitor index for the projector, 0 = primary "
                           "(default: rig.screen in configs/live.yaml; "
                           "`python Data.py check` prints the table)")
    live.add_argument("--camera", type=int, default=None,
                      help="webcam index (default: rig.camera in configs/live.yaml)")
    live.add_argument("--debug-view", action="store_true",
                      help="live window with the pre-warp camera feed + the quad")

    p.add_argument("--device", default=None, help="cuda | cpu (default: cuda if present)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.live and args.video_pair:
        raise SystemExit("--live drives the rig and --video-pair replays a file; "
                         "pick one")

    mode = "live" if args.live else ("replay" if args.video_pair else "offline")
    restorer, detector, info = build_models(
        args.device, need_detector=True,
        detector_backend="none" if args.no_detect else None)
    box_filter = box_filter_kwargs(info["detection_config"])
    out_dir = run_dir(args.output)

    # The writer is opened before the first frame exists, so the canvas size has to
    # be derived the same way the loop will build it.
    want_video = args.live or args.video or mode == "replay"
    video_size = (video_targets(args.video_out, info["input_size"])
                  if mode == "replay" else panel_size(*info["input_size"]))
    video_fps = replay_fps(args.video_pair) if mode == "replay" else 30

    with RunRecorder(out_dir, save_every=args.save_every, frame_kinds=FRAME_KINDS,
                     video_size=video_size if want_video else None,
                     video_fps=video_fps) as rec:
        rec.set(mode=mode,
                restorer=restorer.info(),
                detector=detector.info(),
                box_filter=box_filter, device=info["device"],
                recording={"save_every": args.save_every,
                           "frame_kinds": list(FRAME_KINDS),
                           "video": bool(want_video),
                           # What the writers actually opened, so _artefacts names
                           # the files that exist rather than a fixed result.mp4.
                           "videos": sorted(os.path.basename(v)
                                            for v in rec.video_paths.values())})
        print(f"output: {out_dir}")

        if mode == "live":
            summary = _run_live(args, restorer, detector, rec, box_filter)
        elif mode == "replay":
            summary = _run_replay(args, restorer, detector, rec, box_filter)
        else:
            summary = _run_offline(args, restorer, detector, rec, box_filter)

        meta = rec.finish(**summary)

    _report(meta, summary, detector, out_dir)
    return 0


def _run_live(args, restorer, detector, rec, box_filter):
    import cv2

    from projector_distortion.pipeline.live import run_live  # noqa: F401

    # The rig belongs to the machine, so it is configured rather than flagged. The two
    # exceptions are which monitor and which webcam: those are the settings someone
    # gets wrong on the first run of a session, and `Data.py capture` already lets them
    # be corrected without editing a file. `offset` deliberately stays config-only - it
    # is measured once per rig and does not differ between two runs on it.
    try:
        rig = load_config("live").get("rig") or {}
    except (FileNotFoundError, ImportError) as e:
        raise SystemExit(f"could not read configs/live.yaml\n    {e}") from e

    bg_path = resolve_path(DEFAULT_LIVE_BG)
    background = cv2.imread(bg_path) if bg_path else None
    if background is None:
        print(f"warning: background image not found ({DEFAULT_LIVE_BG}); "
              f"using a flat grey frame.")
    clip = resolve_path(args.clip or DEFAULT_LIVE_VIDEO)
    if not clip or not os.path.exists(clip):
        raise SystemExit(f"projector clip not found: {args.clip}")

    # max_frames and analyse_every keep run_live's own defaults: run to the end of the
    # clip, and pick the analysis stride by measuring it.
    return run_live(
        clip, restorer, detector, rec, background=background,
        camera=int(pick(args.camera, rig, "camera", default=0)),
        screen=int(pick(args.screen, rig, "screen", default=1)),
        cam_backend=str(rig.get("cam_backend", "auto")),
        offset=int(rig.get("offset", 6)), detector_name=detector.name,
        review_calib=bool(rig.get("review_calibration", True)),
        debug_view=args.debug_view, **box_filter)


def _run_replay(args, restorer, detector, rec, box_filter):
    from projector_distortion.pipeline import run_replay

    recording = resolve_path(args.video_pair)
    if not recording or not os.path.isfile(recording):
        raise SystemExit(f"recording not found: {args.video_pair}\n"
                         f"    shoot one with:  python Data.py record --seconds 30")
    clip, source = clip_for(recording, explicit=args.clip,
                            fallback=DEFAULT_LIVE_VIDEO)
    clip = resolve_path(clip) if clip else None
    if not clip or not os.path.isfile(clip):
        raise SystemExit(
            f"projected clip not found: {args.clip or clip}\n"
            f"    pass --clip with the clip this recording was shot against")
    print(f"clip: {os.path.basename(clip)} (from {source})")

    # A replay warps with a single homography too, so a quad off by a corner is just
    # as quietly fatal here as it is live - hence the same rig settings.
    try:
        rig = load_config("live").get("rig") or {}
    except (FileNotFoundError, ImportError) as e:
        raise SystemExit(f"could not read configs/live.yaml\n    {e}") from e

    return run_replay(
        recording, clip, restorer, detector, rec, limit=args.limit,
        offset_frames=int(rig.get("offset", 6)),
        video_out=args.video_out,
        review_calib=(not args.no_review) and bool(rig.get("review_calibration", True)),
        detector_name=detector.name, **box_filter)


def _run_offline(args, restorer, detector, rec, box_filter):
    from projector_distortion.pipeline import run_offline

    input_root = resolve_path(args.input)
    if not input_root or not os.path.isdir(input_root):
        raise SystemExit(
            f"input folder not found: {args.input}\n"
            f"    expected distorted/ and light/ subfolders; see data/README_data.md")

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
    print(f"     {_artefacts(meta)}")
    # Only the offline run scores: evaluate.py needs a surface reference, and neither
    # a live session nor a replay of one has a shot of the unlit scene.
    if meta.get("mode") == "offline":
        print("     score it with:  python evaluate.py")


def _artefacts(meta) -> str:
    """
    What this run actually wrote.

    The image directories only exist when --save-every asked for them, so a fixed
    list sent people looking for folders that were never created.
    """
    names = ["run_meta.json", "detections.csv"]
    rec = meta.get("recording") or {}
    if rec.get("save_every"):
        names += sorted({f"{KIND_DIRS[k]}/" for k in rec.get("frame_kinds") or ()
                         if k in KIND_DIRS})
    if meta.get("mode") in ("live", "replay"):
        names.append("calib/")
    names += sorted(rec.get("videos") or [])
    return " | ".join(names)


if __name__ == "__main__":
    raise SystemExit(main())
