#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rectify raw captures into aligned surface/distorted pairs. No hardware, runs anywhere.

    python Data.py warp
    python Data.py warp --review              # check every scene, fix the bad ones
    python Data.py warp --manual              # click the corners on every scene
    python Data.py warp --mode homography
    python Data.py warp --raw data/Create_Data/raw_0813

Boundary detection is a contour heuristic on a photograph: right most of the time,
and wrong in ways the summary line cannot show. `--review` puts each scene on screen -
the detected quad next to the rectified result - and waits:

    enter / a   accept this scene
    m           click the 4 corners by hand instead
    r           re-run the detector
    s           skip this scene
    A           accept this and every scene after it, no more prompts
    q           stop; the scenes already written are kept

Without --review a scene whose boundary cannot be found is reported and skipped, the
same as before.

Reads raw_<MMDD>/ (capture.py wrote it) and writes warp_<MMDD>/surface/ +
warp_<MMDD>/distorted/ at the model's input size, plus warp_<MMDD>/debug/ overlays
showing where the screen boundary was found. Defaults to the most recent raw folder,
and the warp folder is named after it.

Resolutions, the boundary sampling and the debug toggle live under `warp:` in
configs/collect.yaml.

This stage is not optional. The screen is a trapezoid in the camera and the objects
sit at a different scale in every capture, so nothing before it is trainable - and a
session that only has raw/ is not a layout the loaders recognise at all. After it,
`surface` and `distorted` share a pixel grid and the residual the model learns is only
the projected light.

Three modes:

    boundary     4 corners + the measured edge bow  (default)
    homography   corners only, the exact model for a flat screen
    tps          the legacy warp; needs OpenCV < 5
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # for `common`
from common import (  # noqa: E402
    cfg, dated_suffix, images_in, pick, resolve_raw, shrink, warp_dir, write_meta,
)

# Fallbacks only; the live values come from warp.* in configs/collect.yaml.
WORK_SIZE = (1280, 720)
WARP_POINTS = 20
SCREEN_INSET = 2


def resample(path, n):
    """`n` points spread evenly along a polyline by arc length, both ends included."""
    path = np.asarray(path, np.float32).reshape(-1, 2)
    if len(path) < 2:
        return np.repeat(path[:1] if len(path) else np.zeros((1, 2), np.float32), n, 0)
    seg = np.hypot(*np.diff(path, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] <= 1e-9:
        return np.repeat(path[:1], n, axis=0)
    targets = np.linspace(0.0, cum[-1], n)
    idx = np.clip(np.searchsorted(cum, targets, side="right") - 1, 0, len(seg) - 1)
    alpha = ((targets - cum[idx]) / np.maximum(seg[idx], 1e-9)).astype(np.float32)
    return (path[idx] * (1 - alpha[:, None]) + path[idx + 1] * alpha[:, None])


def inset_contour(contour, shape, pixels):
    """
    Pull the boundary a few pixels inward, off the bright rim of the projection.

    The rim is the blurriest part of the capture; sampling the warp from it drags
    smeared edge pixels into the middle of the rectified image.
    """
    if pixels <= 0:
        return contour
    mask = np.zeros(shape[:2], np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    k = np.ones((2 * pixels + 1, 2 * pixels + 1), np.uint8)
    cnts, _ = cv2.findContours(cv2.erode(mask, k), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)
    # A contour too thin to survive the erosion just keeps its original outline. Not
    # worth a warning: capture.py runs this on every preview frame, so anything
    # printed here floods the console while the operator is setting up.
    return max(cnts, key=cv2.contourArea) if cnts else contour


def screen_contour(img, inset=SCREEN_INSET):
    """Largest closed edge contour - on one of these captures, the screen."""
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(gray, 50, 150),
                       cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    return inset_contour(max(cnts, key=cv2.contourArea), img.shape, inset)


def quad_corners(contour):
    """TL, TR, BR, BL of the contour as float32, or None when it is not quad-like."""
    # Imported here so the module stays importable without the package on sys.path
    # until it is actually needed.
    from projector_distortion.pipeline.live import order_corners

    peri = cv2.arcLength(contour, True)
    for r in np.linspace(0.01, 0.05, 5):
        approx = cv2.approxPolyDP(contour, r * peri, True)
        if len(approx) == 4:
            return np.array(order_corners(approx.reshape(4, 2)), np.float32)
    return None


def edge_paths(contour, corners):
    """The four contour arcs between consecutive corners, clockwise from TL."""
    cnt = contour.reshape(-1, 2).astype(np.float32)
    if cv2.contourArea(cnt, True) < 0:      # force clockwise, so TL -> TR runs forward
        cnt = cnt[::-1]
    idx = [int(np.argmin(np.sum((cnt - c) ** 2, axis=1))) for c in corners]
    paths = []
    for i in range(4):
        a, b = idx[i], idx[(i + 1) % 4]
        paths.append(cnt[a:b + 1] if a <= b else np.vstack([cnt[a:], cnt[:b + 1]]))
    return paths


def boundary_points(paths, n_total, size):
    """`n_total` matched points around the screen boundary and the target rectangle."""
    w, h = size
    counts = [n_total // 4 + (1 if i < n_total % 4 else 0) for i in range(4)]
    rect = [np.float32([0, 0]), np.float32([w, 0]), np.float32([w, h]),
            np.float32([0, h])]
    src, dst = [], []
    for i, n in enumerate(counts):
        if n < 1:
            continue
        # End corner dropped: it is the next edge's start, and TPS rejects duplicates.
        src.append(resample(paths[i], n + 1)[:-1])
        dst.append(resample(np.vstack([rect[i], rect[(i + 1) % 4]]), n + 1)[:-1])
    return np.vstack(src).astype(np.float32), np.vstack(dst).astype(np.float32)


def project(points, matrix):
    """Apply a homography to an (N, 2) or (H, W, 2) array of points."""
    pts = np.asarray(points, np.float32)
    out = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), matrix)
    return out.reshape(pts.shape)


def coons_grid(paths, corners, size):
    """
    Transfinite (Coons) surface through four boundary curves, as an (h, w, 2) grid.

    Each interior point is a blend of the four edges minus the bilinear corner term.
    Straight edges give back the identity grid, which is what makes it usable as a
    correction rather than a warp of its own.
    """
    w, h = size
    tl, tr, br, bl = corners
    top = resample(paths[0], w)                  # TL -> TR
    right = resample(paths[1], h)                # TR -> BR
    bottom = resample(paths[2][::-1], w)         # BL -> BR
    left = resample(paths[3][::-1], h)           # TL -> BL

    u = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :, None]
    v = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    return ((1 - v) * top[None] + v * bottom[None]
            + (1 - u) * left[:, None] + u * right[:, None]
            - ((1 - u) * (1 - v) * tl + u * (1 - v) * tr
               + (1 - u) * v * bl + u * v * br)).astype(np.float32)


def boundary_map(paths, corners, size):
    """
    Dense camera-pixel lookup for the screen rectangle: corner homography + edge bow.

    The four corners fix a homography, which is the exact model for a flat screen
    seen off-axis. Blending the boundary curves directly (a plain Coons patch in
    camera space) would throw that away - it makes the interior bilinear, and on an
    angled rig that misplaces the middle of the image by tens of pixels.

    So the measured edges are pushed through the homography first. A flat screen
    lands them on the rectangle itself and the Coons blend comes back as the
    identity; edges that bow - lens barrel, a curtain, a wall that is not flat -
    deviate, and that deviation is what gets interpolated inward. Needs nothing but
    numpy, which matters since OpenCV 5 dropped the shape module TPS lives in.
    """
    w, h = size
    rect = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    to_rect = cv2.getPerspectiveTransform(corners, rect)
    to_camera = cv2.getPerspectiveTransform(rect, corners)

    flat = [project(p, to_rect) for p in paths]
    grid = project(coons_grid(flat, rect, size), to_camera)
    return np.ascontiguousarray(grid[..., 0]), np.ascontiguousarray(grid[..., 1])


def build_tps(src, dst):
    tps_factory = getattr(cv2, "createThinPlateSplineShapeTransformer", None)
    if tps_factory is None:
        raise SystemExit(
            "--warp tps needs OpenCV's shape module, which OpenCV 5 dropped.\n"
            "    use --warp boundary (curved edges too) or --warp homography, or pin "
            "opencv-python<5.")
    tps = tps_factory()
    matches = [cv2.DMatch(i, i, 0) for i in range(len(src))]
    tps.estimateTransformation(dst.reshape(1, -1, 2), src.reshape(1, -1, 2), matches)
    return tps


def fit(img, w, h):
    """Crop or pad to exactly w x h; TPS returns the input canvas, not the rectangle."""
    out = np.zeros((h, w, 3), img.dtype)
    ih, iw = min(h, img.shape[0]), min(w, img.shape[1])
    out[:ih, :iw] = img[:ih, :iw]
    return out


class Rectifier:
    """
    One scene's camera -> flat screen mapping, built once and reused for every capture.

    The geometry comes from the surface shot, where the screen boundary is visible;
    every `distorted` capture of that scene is warped with the identical mapping,
    which is what makes the pair pixel-aligned enough to train on.
    """

    def __init__(self, contour, size, mode="boundary", points=WARP_POINTS):
        self.size = (int(size[0]), int(size[1]))
        self.mode = mode
        self.corners = quad_corners(contour)
        if self.corners is None:
            raise ValueError("no 4-corner screen boundary in this capture")
        self.paths = edge_paths(contour, self.corners)
        self.src, self.dst = boundary_points(self.paths, points, self.size)

        w, h = self.size
        if mode == "boundary":
            self._map = boundary_map(self.paths, self.corners, self.size)
        elif mode == "homography":
            rect = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
            self._matrix = cv2.getPerspectiveTransform(self.corners, rect)
        elif mode == "tps":
            self._tps = build_tps(self.src, self.dst)
        else:
            raise ValueError(f"unknown warp mode: {mode}")

    def __call__(self, img):
        w, h = self.size
        if self.mode == "boundary":
            return cv2.remap(img, self._map[0], self._map[1], cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
        if self.mode == "homography":
            return cv2.warpPerspective(img, self._matrix, (w, h))
        return fit(self._tps.warpImage(img), w, h)


def debug_overlay(img, contour, src_points):
    canvas = img.copy()
    cv2.drawContours(canvas, [contour.astype(np.int32)], -1, (0, 255, 0), 2)
    for x, y in src_points.astype(int):
        cv2.circle(canvas, (int(x), int(y)), 5, (0, 255, 255), -1)
    return canvas


# ------------------------------------------------------- picking corners by hand

REVIEW_WINDOW = "Warp_Review"
CLICK_WINDOW = "Click_4_Corners"


def contour_from_corners(corners, per_edge=200):
    """
    A dense quad outline from 4 points, shaped like what findContours returns.

    Hand-clicked corners carry no measured edge curve, so the four edges are drawn
    straight. `boundary` mode then reduces to the corner homography, which is the
    right answer for a flat screen anyway - the bow correction only has meaning when
    there was a real edge to measure.
    """
    corners = np.asarray(corners, np.float32).reshape(4, 2)
    edges = [resample(np.vstack([corners[i], corners[(i + 1) % 4]]), per_edge + 1)[:-1]
             for i in range(4)]
    return np.vstack(edges).reshape(-1, 1, 2).astype(np.int32)


def click_corners(img, title=CLICK_WINDOW, max_width=1280):
    """
    Click the 4 screen corners on a still frame. Any order; they get sorted.

    live.py's manual_calibrate does this against a running camera; here the frame is
    already on disk, so this is the still-image twin. Returns corners in the image's
    own pixels, or None if the operator pressed 'q'.
    """
    from projector_distortion.pipeline.live import order_corners

    scale = min(1.0, max_width / img.shape[1])
    view = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1.0 else img.copy()
    points = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_mouse)
    print("    click the 4 screen corners, any order. "
          "right-click undoes, 'u' undoes, 'q' aborts.")
    try:
        while True:
            canvas = view.copy()
            for i, p in enumerate(points):
                cv2.circle(canvas, p, 6, (0, 0, 255), -1)
                if i:
                    cv2.line(canvas, points[i - 1], p, (255, 0, 0), 2)
            if len(points) == 4:
                cv2.line(canvas, points[-1], points[0], (255, 0, 0), 2)
            cv2.putText(canvas, f"{len(points)}/4", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.imshow(title, canvas)

            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                return None
            if key == ord("u") and points:
                points.pop()
            if len(points) == 4 and key in (13, 32, ord("a")):   # enter, space, 'a'
                ordered = order_corners(points)
                return np.array([(x / scale, y / scale) for x, y in ordered],
                                np.float32)
    finally:
        cv2.destroyWindow(title)


def describe_corners(corners, shape):
    """One line per corner plus the sanity numbers, for the review prompt."""
    h, w = shape[:2]
    quad = np.asarray(corners, np.float32).reshape(4, 2)
    area = abs(cv2.contourArea(quad))
    names = ("TL", "TR", "BR", "BL")
    lines = [f"      {n} ({x:6.0f}, {y:6.0f})" for n, (x, y) in zip(names, quad)]
    edges = [float(np.hypot(*(quad[(i + 1) % 4] - quad[i]))) for i in range(4)]
    lines.append(f"      covers {100 * area / (w * h):.0f}% of the {w}x{h} frame")
    lines.append(f"      edges  top {edges[0]:.0f}  right {edges[1]:.0f}  "
                 f"bottom {edges[2]:.0f}  left {edges[3]:.0f} px")
    return "\n".join(lines)


def _build(img, contour, work, mode, points):
    """(rectifier, preview) for one candidate boundary, or (None, None)."""
    try:
        rectifier = Rectifier(contour, work, mode=mode, points=points)
    except ValueError as e:
        print(f"    rejected: {e}")
        return None, None
    return rectifier, rectifier(img)


def recorded_corners(root):
    """
    {surfaceId: corners} that capture.py measured while the operator watched.

    Those were taken with the rectified preview on screen and the rig still standing,
    so they beat re-detecting here on every count. Missing or malformed entries just
    fall back to detection.
    """
    path = os.path.join(root, "collect_meta.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            raw = (json.load(f).get("capture") or {}).get("corners") or {}
    except (OSError, ValueError):
        return {}
    out = {}
    for sid, pts in raw.items():
        try:
            quad = np.asarray(pts, np.float32).reshape(4, 2)
        except (TypeError, ValueError):
            continue
        out[sid] = quad
    return out


def settle_scene(img, name, work, mode, points, inset, review, manual_first,
                 preset=None):
    """
    Decide one scene's warp: auto-detect, optionally show it, optionally redo by hand.

    Returns (rectifier, contour, verdict). `verdict` is one of accept | manual | skip
    | accept_all | quit, and is what tells the caller whether to keep asking.

    Auto-detection is a contour heuristic on a photograph, so it is right most of the
    time and wrong in ways nobody can predict from the summary line alone. Showing the
    rectified result next to the quad it came from is the only cheap check, and being
    able to click the corners on the spot is what stops one bad scene from costing a
    whole re-collection.
    """
    from projector_distortion.utils.visualize import warp_before_after

    source = "detected"
    if manual_first:
        contour = None
    elif preset is not None:
        contour = contour_from_corners(preset)
        source = "recorded at capture time"
    else:
        contour = screen_contour(img, inset)

    rectifier = preview = None
    used_manual = manual_first
    if contour is not None:
        rectifier, preview = _build(img, contour, work, mode, points)

    if rectifier is None and not review and not manual_first:
        return None, None, "auto_failed"      # unattended: report and move on

    while True:
        if rectifier is not None:
            print(f"    {mode} warp, corners {source}:")
            print(describe_corners(rectifier.corners, img.shape))
        else:
            print("    no usable screen boundary found automatically.")

        if not review and rectifier is not None:
            return rectifier, contour, "manual" if used_manual else "accept"

        if rectifier is not None:
            overlay = debug_overlay(img, contour, rectifier.src)
            corners = [tuple(int(v) for v in p) for p in rectifier.corners]
            cv2.imshow(REVIEW_WINDOW, warp_before_after(overlay, preview, corners))
            print(f"    [{name}] enter/a accept | m click corners | r re-detect | "
                  f"s skip | A accept all | q quit")
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyWindow(REVIEW_WINDOW)
        else:
            print(f"    [{name}] m click corners | s skip | q quit")
            key = ord("m")

        if key in (13, 32, ord("a")) and rectifier is not None:
            return rectifier, contour, "manual" if used_manual else "accept"
        if key == ord("A") and rectifier is not None:
            return rectifier, contour, "accept_all"
        if key == ord("s"):
            return None, None, "skip"
        if key == ord("q"):
            return None, None, "quit"
        if key == ord("r"):
            # Re-detect: an explicit request to ignore whatever preset there was.
            contour = screen_contour(img, inset)
            source = "detected"
            rectifier, preview = (_build(img, contour, work, mode, points)
                                  if contour is not None else (None, None))
            continue
        if key == ord("m"):
            clicked = click_corners(img)
            if clicked is None:
                # Aborting the click leaves whatever the auto pass had, if anything.
                if rectifier is None:
                    return None, None, "skip"
                continue
            contour = contour_from_corners(clicked)
            source = "clicked by hand"
            rectifier, preview = _build(img, contour, work, mode, points)
            used_manual = True
            if rectifier is not None and not review:
                return rectifier, contour, "manual"
            continue


def cmd_warp(args):
    from projector_distortion.data import surface_id
    from projector_distortion.utils.visualize import warp_before_after

    mode = pick(args.mode, "warp", "mode", default="boundary")
    work = tuple(cfg("warp", "work_size", default=list(WORK_SIZE)))
    final = tuple(cfg("warp", "final_size", default=[640, 360]))
    points = int(cfg("warp", "points", default=WARP_POINTS))
    inset = int(cfg("warp", "inset", default=SCREEN_INSET))
    debug = bool(cfg("warp", "debug", default=True))

    surface_dir = os.path.join(args.raw, "surface")
    distorted_dir = os.path.join(args.raw, "distorted")
    # Named after the captures, not after today: raw_0812 always pairs with warp_0812
    # however many days later it gets rectified.
    out_root = args.out or warp_dir(dated_suffix(args.raw))
    out_surface = os.path.join(out_root, "surface")
    out_distorted = os.path.join(out_root, "distorted")
    debug_dir = os.path.join(out_root, "debug")
    print(f"raw:  {args.raw}\nwarp: {out_root}")

    surface_files = images_in(surface_dir)
    if not surface_files:
        raise SystemExit(
            f"no surface captures in {surface_dir}\n"
            f"    run: python Data.py capture --raw {args.raw}")
    for d in (out_surface, out_distorted) + ((debug_dir,) if debug else ()):
        os.makedirs(d, exist_ok=True)

    distorted_files = images_in(distorted_dir)
    by_surface = {}
    for name, path in distorted_files.items():
        by_surface.setdefault(surface_id(name), []).append((name, path))

    names = list(surface_files)[:args.limit] if args.limit else list(surface_files)
    print(f"{len(names)} scene(s), {len(distorted_files)} capture(s) | "
          f"warp={mode} | {work[0]}x{work[1]} -> {final[0]}x{final[1]}")

    presets = {} if args.redetect else recorded_corners(args.raw)
    if presets:
        print(f"{len(presets)} scene(s) carry corners recorded at capture time; "
              f"--redetect ignores them.")

    done_surface = done_distorted = 0
    failed = []
    review = args.review
    manual_count = preset_count = 0
    for name in names:
        sid = surface_id(name)
        img = cv2.imread(surface_files[name])
        if img is None:
            failed.append((name, "unreadable"))
            continue

        preset = presets.get(sid)
        rectifier, contour, verdict = settle_scene(
            img, name, work, mode, points, inset,
            review=review, manual_first=args.manual, preset=preset)
        if verdict == "quit":
            print("  stopped by user; the scenes already written are kept.")
            break
        if verdict == "accept_all":
            review = False
        if rectifier is None:
            failed.append((name, "skipped" if verdict == "skip"
                           else "no screen boundary"))
            continue
        if verdict == "manual":
            manual_count += 1
        elif preset is not None:
            preset_count += 1

        surface = shrink(rectifier(img), final)
        cv2.imwrite(os.path.join(out_surface, name), surface)
        done_surface += 1

        if debug:
            overlay = debug_overlay(img, contour, rectifier.src)
            corners = [tuple(int(v) for v in p) for p in rectifier.corners]
            cv2.imwrite(os.path.join(debug_dir, f"{sid}_warp.jpg"),
                        warp_before_after(overlay, surface, corners))

        for distorted_name, distorted_path in by_surface.get(sid, []):
            distorted = cv2.imread(distorted_path)
            if distorted is None:
                failed.append((distorted_name, "unreadable"))
                continue
            cv2.imwrite(os.path.join(out_distorted, distorted_name),
                        shrink(rectifier(distorted), final))
            done_distorted += 1
        print(f"  {name}: {len(by_surface.get(sid, []))} capture(s) rectified",
              flush=True)

    # Against every surface shot, not just the ones --limit let through: a scene that
    # was merely skipped this run still has its surface image.
    orphans = sorted(set(by_surface) - {surface_id(n) for n in surface_files})
    if orphans:
        print(f"warning: {len(orphans)} surfaceId(s) have captures but no surface shot "
              f"(e.g. {orphans[0]}); those captures were skipped.")
    for name, why in failed:
        print(f"warning: {name}: {why}")

    write_meta(out_root, "warp", {
        "raw": args.raw,
        "mode": mode, "points": points, "inset": inset,
        "work_size": list(work), "final_size": list(final),
        "reviewed": bool(args.review), "manual_scenes": manual_count,
        "scenes_from_capture_corners": preset_count,
        "surface": done_surface, "distorted": done_distorted,
        "failed": [{"file": n, "reason": w} for n, w in failed],
        "orphan_surface_ids": orphans,
    })
    if preset_count:
        print(f"{preset_count} scene(s) used the corners recorded at capture time.")
    if manual_count:
        print(f"{manual_count} scene(s) had their corners clicked by hand.")
    print(f"\n{done_surface} surface + {done_distorted} distorted -> {out_root}")
    print(f"next: python demo.py --input {out_root}")
    return 0


def build_parser():
    p = argparse.ArgumentParser(
        prog="Data.py warp", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default=None,
                   help="captures to rectify (default: the newest "
                        "<session.dir>/raw_<MMDD>)")
    p.add_argument("--out", default=None,
                   help="where the pairs go (default: <session.dir>/warp_<MMDD>, "
                        "named after the raw folder)")
    p.add_argument("--mode", default=None, choices=["boundary", "homography", "tps"],
                   help="boundary: 4 corners + the measured edge bow | homography: "
                        "corners only | tps: the legacy warp, needs OpenCV < 5 "
                        "(default: warp.mode in collect.yaml)")
    p.add_argument("--limit", type=int, default=0, help="process at most N scenes")
    p.add_argument("--review", action="store_true",
                   help="show each scene's detected quad and its rectified result, "
                        "then choose: accept, click the corners by hand, re-detect, "
                        "skip, accept the rest, or quit")
    p.add_argument("--manual", action="store_true",
                   help="skip auto-detection and click the 4 corners on every scene")
    p.add_argument("--redetect", action="store_true",
                   help="ignore the corners capture.py recorded and detect again")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.raw = resolve_raw(args.raw)
    return cmd_warp(args)


if __name__ == "__main__":
    raise SystemExit(main())
