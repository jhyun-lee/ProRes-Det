#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
List the displays and preview the webcam. Run this before anything drives the rig.

    python Data.py check
    python Data.py check --camera 1

Its whole job is to make capture.py not be the first time a wrong monitor index or a
camera held by another app shows up. Prints the --screen table, then opens a preview
window; 'q' closes it.

The camera backend lives in configs/collect.yaml (capture.cam_backend) - it is a
property of the machine, not of the run.
"""

import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # for `common`
from common import PREVIEW, cfg, pick  # noqa: E402


def cmd_check(args):
    from projector_distortion.pipeline.live import open_webcam
    from projector_distortion.utils.display import list_monitors

    monitors = list_monitors()
    if not monitors:
        print("no monitors detected; --screen will fall back to the primary display.")
    for i, m in enumerate(monitors):
        star = " (primary)" if m.primary else ""
        print(f"  --screen {i} -> {m.width}x{m.height} at ({m.x},{m.y}){star}  {m.name}")

    camera = pick(args.camera, "capture", "camera", default=0)
    backend = cfg("capture", "cam_backend", default="auto")
    cam = open_webcam(camera, backend)
    if cam is None:
        raise SystemExit(
            f"cannot open webcam index {camera}\n"
            f"    try another --camera index, or set capture.cam_backend in "
            f"configs/collect.yaml (auto | any | dshow | msmf | v4l2).")
    print(f"camera {camera} ({backend}) open - 'q' closes the preview.")
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                print("warning: camera read failed.")
                break
            cv2.imshow(PREVIEW, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()

    print("\nnext: python Data.py make_light")
    return 0


def build_parser():
    p = argparse.ArgumentParser(
        prog="Data.py check", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", type=int, default=None,
                   help="webcam index (default: capture.camera in collect.yaml)")
    return p


def main(argv=None):
    return cmd_check(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
