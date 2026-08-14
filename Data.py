#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One entry point for everything under data/.

    python Data.py check                     monitors + webcam, run once
    python Data.py make_light                video   -> light frames   [starts a session]
    python Data.py capture --screen 2        project -> raw captures   [needs the rig]
    python Data.py capture_warp --screen 2   the same, rectifying as it shoots
    python Data.py warp                      rectify -> aligned pairs
    python Data.py record --seconds 30       project a clip and record it, no models

`capture` writes only the raw captures, so the warp can be redone later without the
rig. `capture_warp` writes the raw *and* the rectified pairs in one pass - same
collection, one less step, and `Data.py warp` can still redo any of it.

Everything lands under data/Create_Data/:

    projected/      light frames - built once, shared by every capture day
    raw_<MMDD>/     that day's captures, unrectified   surface/ distorted/
    warp_<MMDD>/    the rectified pairs                surface/ distorted/ debug/

projected/ is not dated because the frames a clip yields do not change from one day
to the next. A warp folder is named after the raw it came from, so raw_0812 and
warp_0812 describe the same captures however long the gap. `capture` fills today's
raw; `warp` takes the newest raw unless --raw names another.

Every stage keeps its own --help:

    python Data.py capture --help

Settings that do not change between runs - camera backend, resolutions, the hue bands,
the warp geometry - live in projector_distortion/configs/collect.yaml, not on the
command line.
"""

import argparse
import importlib.util
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# stage -> (module file, extra argv, one-line help)
STAGES = {
    "check": ("check.py", [], "list the displays and preview the webcam"),
    "make_light": ("make_light.py", [],
                   "video -> light frames (starts a session)"),
    "capture": ("capture.py", [], "projector + webcam -> raw captures"),
    "capture_warp": ("capture.py", ["--warp"],
                     "capture, and rectify each shot as it is taken"),
    "warp": ("warp.py", [], "rectify raw captures into aligned pairs"),
    "record": ("record.py", [], "project a clip and record the camera, no models"),
}

ORDER = ["check", "make_light", "capture", "capture_warp", "warp", "record"]


def _load(filename):
    """
    Import one of the data/ scripts by path.

    They are standalone programs, not a package - there is no data/__init__.py and
    deliberately so, since each one is meant to be runnable on its own. Loading by
    path is what lets this dispatcher reuse their main() unchanged.
    """
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise SystemExit(f"missing stage script: {path}")
    if DATA_DIR not in sys.path:
        sys.path.insert(0, DATA_DIR)        # the scripts import `common` from there
    spec = importlib.util.spec_from_file_location(os.path.splitext(filename)[0], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_parser():
    p = argparse.ArgumentParser(
        prog="Data.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=ORDER, metavar="stage",
                   help="one of: " + " | ".join(ORDER))
    p.add_argument("args", nargs=argparse.REMAINDER,
                   help="passed through to that stage; try `Data.py <stage> --help`")
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    # Bare `Data.py`, or `-h` before a stage: list the stages rather than making the
    # reader guess at REMAINDER semantics.
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        print("\nstages:")
        for name in ORDER:
            print(f"  {name:<13s} {STAGES[name][2]}")
        return 0

    args = build_parser().parse_args(argv)
    module, extra, _ = STAGES[args.stage]
    return _load(module).main(extra + args.args)


if __name__ == "__main__":
    raise SystemExit(main())
