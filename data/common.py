#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared plumbing for the collection scripts in this folder.

    check.py       list the displays, preview the webcam       (no session)
    make_light.py  video      -> light frames                  [light]
    capture.py     projector  -> raw surface/distorted shots    [capture]
    warp.py        raw shots  -> aligned pairs                  [warp]
    record.py      project a clip and record it, no models

They are separate programs because they need separate things: `check` and `capture`
drive hardware, the rest are plain file work and run on any machine. `Data.py` at the
repo root dispatches to all of them.

Settings come from projector_distortion/configs/collect.yaml. Only what genuinely
changes between runs stayed a CLI flag; the rest is a property of the rig or of the
naming convention, so it is set once in the YAML.
"""

import json
import os
import sys
from datetime import datetime

import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# config.py imports nothing heavier than PyYAML, so make_light.py and warp.py stay
# runnable on a machine with no torch.
from projector_distortion.config import load_config, resolve_path  # noqa: E402

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp")
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".wmv")

# Kept as literals rather than imported from projector_distortion.data: that module
# pulls in torch, and two of these stages are meant to run without it.
DISTORTED_PREFIX = "distorted_"
LIGHT_PREFIX = "light_"
SURFACE_PREFIX = "surface_"

PREVIEW = "Webcam_Preview"
CAPTURE_VIEW = "Capture_Feed"

_CFG = None


def cfg(*keys, default=None):
    """
    One setting out of collect.yaml, by path: cfg("capture", "settle_ms").

    Loaded once. A missing key returns `default` rather than raising, so a config
    trimmed down to the few lines someone cares about still works.
    """
    global _CFG
    if _CFG is None:
        try:
            _CFG = load_config("collect")
        except (FileNotFoundError, ImportError) as e:
            raise SystemExit(f"could not read configs/collect.yaml\n    {e}") from e
    node = _CFG
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return default if node is None else node


def cfg_path(*keys, default=None):
    """A path setting, made absolute against the project root."""
    return resolve_path(cfg(*keys, default=default))


def pick(cli_value, *keys, default=None):
    """CLI wins when it was given, otherwise collect.yaml, otherwise `default`."""
    return cli_value if cli_value is not None else cfg(*keys, default=default)


def train_clips():
    """The clips make_light.py reads by default, absolute."""
    return [resolve_path(p) for p in cfg("light", "src", default=[])]


# ------------------------------------------------------------- the data directory
#
#   <dir>/projected/      light frames, built once and shared by every day
#   <dir>/raw_<MMDD>/     that day's captures, unrectified
#   <dir>/warp_<MMDD>/    the rectified pairs for those captures, plus debug/
#
# A warp folder is named after the raw it came from, not after the day warp ran, so
# raw_0812 and warp_0812 describe the same captures however long the gap between them.

# The pre-rename name of the projected/ folder. Only the folder was renamed: the
# files inside stay light_<lightId>.jpg, because `projected_` is already the
# pre-rename prefix for *distorted* files and reusing it would make a light frame
# parse as a distorted one.
LEGACY_PROJECTED_DIR = "light"


def today():
    return datetime.now().strftime("%m%d")


def data_dir():
    # normpath: the YAML writes forward slashes, and these paths get printed and
    # pasted back as arguments often enough that mixed separators are worth avoiding.
    return os.path.normpath(resolve_path(cfg("session", "dir",
                                             default="data/Create_Data")))


def projected_dir(create=False):
    """
    The shared light-frame folder. `light/` is still read if that is what exists.

    Not dated: the frames a clip yields do not change from one day to the next, so
    re-extracting them per day would only cost disk.
    """
    base = data_dir()
    current = os.path.join(base, cfg("session", "projected", default="projected"))
    if create or os.path.isdir(current):
        return current
    legacy = os.path.join(base, LEGACY_PROJECTED_DIR)
    return legacy if os.path.isdir(legacy) else current


def _dated(prefix_key, default_prefix, date=None):
    prefix = cfg("session", prefix_key, default=default_prefix)
    return os.path.join(data_dir(), f"{prefix}{date or today()}")


def raw_dir(date=None):
    return _dated("raw_prefix", "raw_", date)


def warp_dir(date=None):
    return _dated("warp_prefix", "warp_", date)


def dated_suffix(path):
    """The MMDD out of a raw_<MMDD> / warp_<MMDD> folder name, or None."""
    name = os.path.basename(os.path.normpath(str(path)))
    _, _, tail = name.partition("_")
    return tail or None


def latest_dated(prefix_key, default_prefix):
    """
    The newest raw_*/warp_* folder, or None.

    Naming is zero-padded MMDD, so lexical order is chronological within a year.
    """
    base = data_dir()
    prefix = cfg("session", prefix_key, default=default_prefix)
    if not os.path.isdir(base):
        return None
    hits = sorted(d for d in os.listdir(base)
                  if d.startswith(prefix) and os.path.isdir(os.path.join(base, d)))
    return os.path.join(base, hits[-1]) if hits else None


def resolve_raw(explicit, create=False):
    """
    Which raw_<MMDD> a stage works in.

    capture creates today's. warp takes the newest that exists rather than
    re-deriving today's, so rectifying the morning after a shoot still finds the
    captures instead of an empty folder.
    """
    if explicit:
        root = os.path.abspath(explicit)
    elif create:
        root = raw_dir()
    else:
        root = latest_dated("raw_prefix", "raw_")
        if root is None:
            raise SystemExit(
                f"no {cfg('session', 'raw_prefix', default='raw_')}<MMDD> folder "
                f"under {data_dir()}\n"
                f"    run: python Data.py capture     (creates today's)\n"
                f"    or pass --raw <folder> to work in a specific one.")
        if os.path.basename(root) != os.path.basename(raw_dir()):
            print(f"using the most recent captures ({os.path.basename(root)}), "
                  f"not today's - --raw <folder> picks another.")
    os.makedirs(root, exist_ok=True)
    return root


# ------------------------------------------------------------------ small helpers


def images_in(directory):
    if not directory or not os.path.isdir(directory):
        return {}
    return {f: os.path.join(directory, f) for f in sorted(os.listdir(directory))
            if f.lower().endswith(IMAGE_EXT)}


def write_meta(root, stage, payload):
    """Merge one stage's settings into the session's collect_meta.json."""
    path = os.path.join(root, "collect_meta.json")
    meta = {}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, ValueError):
            print(f"warning: could not read {path}; it is being rewritten.")
    meta.setdefault("root", root)
    meta[stage] = dict(payload, finished_at=datetime.now().isoformat(timespec="seconds"))
    os.makedirs(root, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
    return path


def shrink(img, size):
    """`size` is (width, height); None keeps the image as it is."""
    if size is None or (img.shape[1], img.shape[0]) == tuple(size):
        return img
    shrinking = size[0] * size[1] < img.shape[1] * img.shape[0]
    return cv2.resize(img, tuple(size),
                      interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR)


def jpeg_params():
    return [int(cv2.IMWRITE_JPEG_QUALITY), int(cfg("session", "jpeg_quality",
                                                   default=95))]


