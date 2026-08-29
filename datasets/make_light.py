#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video -> the `light` half of every training pair. No hardware, runs anywhere.

    python Data.py make_light                    # the three train_light_*.mp4 clips
    python Data.py make_light --step 6           # denser sampling
    python Data.py make_light --augment none     # no colour variants
    python Data.py make_light --src /path/clip.mp4 /path/folder

Writes into the one shared folder every capture day draws from:

    data/Create_Data/projected/

Deliberately not dated. The frames a clip yields are the same on any day, so a
per-day copy would cost disk and buy nothing; capture.py samples this pool afresh
each round instead. Running this again adds to the pool rather than replacing it.

`--src` defaults to light.src in configs/collect.yaml - the three train clips.
data/live/test_light.mp4 is the held-out projection source and is deliberately not
among them, so a set collected from these shares no projected frame with the test
split.

Each kept frame is written six ways by default - untouched, inverted, and four hue
rotations. The projector throws arbitrary colour and `light` is the model's input
channels 3:6, so widening that distribution costs disk, not footage or rig time.

    projected/light_<tag>_<videoIdx>_<frameIdx>_<variant>.jpg
                                                        variant: 0 | invert | <hue>

The folder is `projected/` but the files stay `light_*`: `projected_` is already the
pre-rename prefix for *distorted* files, so a light frame named that way would parse
as a distorted one.

There is no cache and no "already done" check: every run stamps a fresh tag into the
filenames, so running this twice over one clip writes a second copy of every frame
rather than skipping it.
"""

import argparse
import os
import random
import sys
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # for `common`
from common import (  # noqa: E402
    LIGHT_PREFIX, VIDEO_EXT, cfg, jpeg_params, pick, projected_dir, shrink,
    train_clips, write_meta,
)

AUGMENT_MODES = ("full", "invert", "none")
PER_FRAME = {"full": 6, "invert": 2, "none": 1}


def augment(frame, mode, rng, bands):
    """
    (suffix, image) for every copy one source frame turns into.

    Suffixes match what the bundled splits already carry (`_0`, `_invert`, `_<hue>`).
    """
    yield "0", frame
    if mode == "none":
        return
    yield "invert", cv2.bitwise_not(frame)
    if mode == "invert":
        return

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0].astype(np.int16)      # uint8 would wrap at 256 before the % 180
    for lo, hi in bands:
        # lo or 1: a 0 degree shift would write the same suffix as the untouched copy
        shift = rng.randrange(lo or 1, hi)
        rotated = hsv.copy()
        rotated[..., 0] = ((hue + shift) % 180).astype(np.uint8)
        yield str(shift), cv2.cvtColor(rotated, cv2.COLOR_HSV2BGR)


def extract_frames(video, out_dir, tag, video_idx, step, limit, mode, rng, size, bands):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"warning: cannot open {video}; skipped.")
        return 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  {os.path.basename(video)}: {total} frames, keeping every {step}")

    params = jpeg_params()
    saved = kept = frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            small = shrink(frame, size)
            for suffix, image in augment(small, mode, rng, bands):
                name = f"{LIGHT_PREFIX}{tag}_{video_idx}_{frame_idx}_{suffix}.jpg"
                cv2.imwrite(os.path.join(out_dir, name), image, params)
                saved += 1
            kept += 1
            # --limit counts source frames, so the augmentation multiplier stays a
            # property of --augment and does not silently change what --limit means.
            if limit and kept >= limit:
                break
        frame_idx += 1
    cap.release()
    return saved


def collect_videos(sources):
    """Any mix of video files and folders of them, flattened."""
    videos = []
    for src in sources:
        if os.path.isdir(src):
            videos += [os.path.join(src, f) for f in sorted(os.listdir(src))
                       if f.lower().endswith(VIDEO_EXT)]
        elif os.path.exists(src):
            videos.append(src)
        else:
            print(f"warning: no such video or folder, skipped: {src}")
    return videos


def cmd_light(args):
    out_dir = args.out or projected_dir(create=True)
    os.makedirs(out_dir, exist_ok=True)
    print(f"projected: {out_dir}")

    videos = collect_videos(args.src or train_clips())
    if not videos:
        raise SystemExit("no video found. Pass --src, or check light.src in "
                         "configs/collect.yaml.")

    step = pick(args.step, "light", "step", default=30)
    mode = pick(args.augment, "light", "augment", default="full")
    size = tuple(cfg("light", "size", default=[1280, 720]))
    bands = [tuple(b) for b in cfg("light", "hue_bands",
                                   default=[[0, 45], [45, 90], [90, 140], [140, 180]])]
    first_idx = int(cfg("light", "first_video_index", default=1000))
    seed = cfg("light", "seed")

    tag = datetime.now().strftime("%m%d%H%M%S")   # no '_': it has to survive id parsing
    rng = random.Random(seed)
    print(f"{len(videos)} video(s) -> {out_dir}")
    print(f"  augment: {mode} ({PER_FRAME[mode]} image(s) per source frame)")

    total = 0
    for i, video in enumerate(videos, start=first_idx):
        total += extract_frames(video, out_dir, tag, i, step, args.limit, mode, rng,
                                size, bands)

    write_meta(out_dir, "light", {
        "videos": [os.path.abspath(v) for v in videos], "out": out_dir, "tag": tag,
        "step": step, "size": list(size), "augment": mode, "seed": seed,
        "per_source_frame": PER_FRAME[mode], "frames": total,
    })
    print(f"\n{total} light frame(s) -> {out_dir}")
    print(f"    {len(os.listdir(out_dir)) - 1} file(s) in the folder in total")
    print("next: python Data.py capture --screen 2")
    return 0


def build_parser():
    p = argparse.ArgumentParser(
        prog="Data.py make_light", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=None,
                   help="where the light frames go (default: the shared "
                        "<session.dir>/projected)")
    p.add_argument("--src", nargs="+", default=None,
                   help="video file(s) and/or folder(s) of them "
                        "(default: light.src in collect.yaml)")
    p.add_argument("--step", type=int, default=None,
                   help="keep every Nth frame (default: light.step, 30 = one per "
                        "second at 30fps)")
    p.add_argument("--augment", choices=AUGMENT_MODES, default=None,
                   help="copies per source frame: full = original + inverted + 4 hue "
                        "rotations (6x), invert = 2x, none = 1x "
                        "(default: light.augment)")
    p.add_argument("--limit", type=int, default=0,
                   help="at most N source frames per video, before augmentation")
    return p


def main(argv=None):
    return cmd_light(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
