#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a dataset with a projector and a webcam - beam, distorted and original.

Three views of one moment are what a training sample is made of:

    beam    the frame the projector emitted            model input ch 3:6
    pro     the camera's view of it on the screen      model input ch 0:3  (distorted)
    clean   the same screen with nothing projected     restoration target  (original)

Four stages, in this order:

    python data/collect.py check                     # monitors + webcam, run once
    python data/collect.py beam --src videos/        # video   -> beam frames
    python data/collect.py capture --screen 2        # project -> raw ori/pro captures
    python data/collect.py warp                      # rectify -> aligned clean/pro

`check` and `capture` drive the rig; `beam` and `warp` are plain file work and run
on any machine.

Everything lands in one session folder, laid out so the rest of the repo reads it
with no conversion:

    data/collected_<MMDD>/
      beam/   output_video_<beamId>.jpg          [beam]
      raw/    ori/Ori<oriId>.jpg                 [capture] camera frames, unrectified
              pro/projected_<oriId>_<beamId>.jpg [capture]
      clean/  Ori<oriId>.jpg                     [warp]    rectified, 640x360
      pro/    projected_<oriId>_<beamId>.jpg     [warp]    rectified, 640x360
      debug/  <oriId>_warp.jpg                   [warp]    before/after evidence
      collect_meta.json

    python demo.py     --input data/collected_0803 --gt data/collected_0803
    python train.py --data-root data/collected_0803

Detection labels are not collected here - clean/ has to be annotated by hand into
labels/ before evaluate.py can score mAP. See data/README_data.md.
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_BACKGROUND = os.path.join(DATA_DIR, "live", "BaseBackGround.jpg")
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp")
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".wmv")

PREVIEW = "Webcam_Preview"
CAPTURE_VIEW = "Capture_Feed"


# --------------------------------------------------------------------------- shared

def default_root():
    """One session folder per day, so a second run of a stage extends the first."""
    return os.path.join(DATA_DIR, f"collected_{datetime.now().strftime('%m%d')}")


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


# ----------------------------------------------------------------------- 0. check

def cmd_check(args):
    """List the displays and open the webcam, so capture is not the first attempt."""
    from projector_distortion.pipeline.live import list_monitors, open_webcam

    monitors = list_monitors()
    if not monitors:
        print("no monitors detected; --screen will fall back to the primary display.")
    for i, m in enumerate(monitors):
        star = " (primary)" if m.primary else ""
        print(f"  --screen {i} -> {m.width}x{m.height} at ({m.x},{m.y}){star}  {m.name}")

    cam = open_webcam(args.camera, args.cam_backend, args.cam_width, args.cam_height,
                      args.cam_fps)
    if cam is None:
        raise SystemExit(f"cannot open webcam index {args.camera}; try another "
                         f"--camera index or --cam-backend.")
    print("preview running - 'q' closes it.")
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
    return 0


# ------------------------------------------------------------------------ 1. beam

def extract_frames(video, out_dir, tag, video_idx, step, size, quality, limit):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"warning: cannot open {video}; skipped.")
        return 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  {os.path.basename(video)}: {total} frames, keeping every {step}")

    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    saved = frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            name = f"output_video_{tag}_{video_idx}_{frame_idx}.jpg"
            cv2.imwrite(os.path.join(out_dir, name), shrink(frame, size), params)
            saved += 1
            if limit and saved >= limit:
                break
        frame_idx += 1
    cap.release()
    return saved


def cmd_beam(args):
    """
    Split the clip(s) the projector will play into the `beam` half of each pair.

    The filename carries the id the whole pipeline pairs on: `beamId` is everything
    after `output_video_`, and the capture stage copies it verbatim into the `pro`
    filename. Nothing downstream needs a manifest.
    """
    out_dir = args.out or os.path.join(args.root, "beam")
    os.makedirs(out_dir, exist_ok=True)

    if os.path.isdir(args.src):
        videos = [os.path.join(args.src, f) for f in sorted(os.listdir(args.src))
                  if f.lower().endswith(VIDEO_EXT)]
    else:
        videos = [args.src]
    if not videos:
        raise SystemExit(f"no video found at {args.src}")

    size = None if 0 in args.size else tuple(args.size)
    tag = datetime.now().strftime("%m%d%H%M%S")   # no '_': it has to survive id parsing
    print(f"{len(videos)} video(s) -> {out_dir}")

    total = 0
    for i, video in enumerate(videos, start=args.video_index):
        total += extract_frames(video, out_dir, tag, i, args.step, size, args.quality,
                                args.limit)

    write_meta(args.root, "beam", {
        "videos": [os.path.abspath(v) for v in videos], "out": out_dir, "tag": tag,
        "step": args.step, "size": list(size) if size else None,
        "quality": args.quality, "frames": total,
    })
    print(f"\n{total} beam frame(s) -> {out_dir}")
    print(f"next: python data/collect.py capture --root {args.root} --screen 1")
    return 0


# --------------------------------------------------------------------- 2. capture

def grab_fresh(cam, flush=3):
    """
    Drop the frames the driver already had queued, then take the live one.

    Webcams hand back a buffered frame first, which on this rig means the picture
    from *before* the projector changed. Flushing is what keeps `pro` matched to its
    `beam` - the live pipeline solves the same problem with --offset.
    """
    for _ in range(max(0, flush)):
        cam.grab()
    ok, frame = cam.read()
    return frame if ok else None


def shoot_clean(cam, flush):
    """
    Hold at the background frame until the operator presses 's'.

    That shot is the scene with no content projected on it - the restoration target -
    so it is taken by hand: the objects have to be placed and the operator out of
    frame first. Returns None if 'q' was pressed instead.
    """
    print("  set the scene, then press 's' in the preview to take the clean shot "
          "('q' aborts).")
    while True:
        frame = grab_fresh(cam, 1)
        if frame is None:
            print("warning: camera read failed.")
            return None
        cv2.imshow(PREVIEW, frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("s"):
            shot = grab_fresh(cam, flush)
            cv2.destroyWindow(PREVIEW)
            return shot
        if key == ord("q"):
            cv2.destroyWindow(PREVIEW)
            return None


def capture_round(cam, beam_files, ori_id, pro_dir, args, params):
    """Project every beam frame once and capture it. Returns (saved, aborted)."""
    from projector_distortion.pipeline.live import WINDOW

    saved = 0
    for i, path in enumerate(beam_files, start=1):
        beam = cv2.imread(path)
        if beam is None:
            print(f"warning: could not read {path}; skipped.")
            continue

        cv2.imshow(WINDOW, beam)
        key = cv2.waitKey(max(1, args.settle_ms)) & 0xFF   # projector + panel latency
        if key == ord("q"):
            print("  stopped by user.")
            return saved, True

        frame = grab_fresh(cam, args.flush)
        if frame is None:
            print("warning: camera read failed; ending the round.")
            return saved, True

        beam_id = os.path.splitext(os.path.basename(path))[0]
        beam_id = beam_id[len("output_video_"):] if beam_id.startswith("output_video_") \
            else beam_id
        out = os.path.join(pro_dir, f"projected_{ori_id}_{beam_id}.jpg")
        if cv2.imwrite(out, frame, params):
            saved += 1

        if i % max(1, args.preview_every) == 0 or i == 1:
            cv2.imshow(CAPTURE_VIEW, shrink(frame, (480, 360)))
            cv2.waitKey(1)
        if i % 25 == 0 or i == len(beam_files):
            print(f"    {i}/{len(beam_files)} projected, {saved} captured", flush=True)
    return saved, False


def cmd_capture(args):
    """
    Drive the projector and the webcam: one clean shot, then one capture per beam frame.

    A round is one scene setup. `oriId` is the timestamp of its clean shot and every
    capture in that round carries it, which is what ties many `pro` files back to the
    single `clean` they should be restored to.
    """
    from projector_distortion.pipeline.live import WINDOW, open_webcam, place_window

    beam_dir = args.beam or os.path.join(args.root, "beam")
    beam_files = list(images_in(beam_dir).values())
    if not beam_files:
        raise SystemExit(
            f"no beam frames in {beam_dir}\n"
            f"    run: python data/collect.py beam --root {args.root} --src <videos>")
    if args.shuffle:
        random.Random(args.seed).shuffle(beam_files)
    if args.limit:
        beam_files = beam_files[:args.limit]

    ori_dir = os.path.join(args.root, "raw", "ori")
    pro_dir = os.path.join(args.root, "raw", "pro")
    os.makedirs(ori_dir, exist_ok=True)
    os.makedirs(pro_dir, exist_ok=True)

    background = cv2.imread(args.background) if os.path.exists(args.background) else None
    if background is None:
        print(f"warning: background image not found ({args.background}); "
              f"using a flat grey frame.")
        background = np.full((720, 1280, 3), 32, np.uint8)

    monitor = place_window(args.screen, background)
    cam = open_webcam(args.camera, args.cam_backend, args.cam_width, args.cam_height,
                      args.cam_fps)
    if cam is None:
        cv2.destroyAllWindows()
        raise SystemExit(
            f"cannot open webcam index {args.camera}\n"
            f"    try another --camera index or --cam-backend, and close any other "
            f"app holding the camera.")
    # Many drivers ignore this; grab_fresh() is what actually drops the stale frames.
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]
    print(f"{len(beam_files)} beam frame(s) x {args.rounds} round(s) -> {args.root}")

    rounds, clean_shots, captured = 0, 0, 0
    t0 = time.time()
    try:
        for r in range(args.rounds):
            print(f"\nround {r + 1}/{args.rounds}")
            cv2.imshow(WINDOW, background)
            cv2.waitKey(1)

            clean = shoot_clean(cam, args.flush)
            if clean is None:
                print("aborted before the clean shot.")
                break
            ori_id = datetime.now().strftime("%m%d%H%M%S")   # no '_': parsed as the id
            cv2.imwrite(os.path.join(ori_dir, f"Ori{ori_id}.jpg"), clean, params)
            clean_shots += 1
            print(f"  clean shot: Ori{ori_id}.jpg")
            # The preview window took focus, which can pull the projector window off
            # its monitor on Windows; put it back before anything is projected.
            place_window(args.screen, background, announce=False)
            time.sleep(args.round_settle)

            saved, aborted = capture_round(cam, beam_files, ori_id, pro_dir, args, params)
            captured += saved
            rounds += 1
            if aborted:
                break
    except KeyboardInterrupt:
        print("\ninterrupted.")
    finally:
        cam.release()
        cv2.destroyAllWindows()

    write_meta(args.root, "capture", {
        "beam_dir": beam_dir, "beam_frames": len(beam_files),
        "rounds_requested": args.rounds, "rounds_done": rounds,
        "clean_shots": clean_shots, "captures": captured,
        "camera": {"index": args.camera, "backend": args.cam_backend,
                   "requested": [args.cam_width, args.cam_height, args.cam_fps]},
        "screen_index": args.screen,
        "monitor": dict(zip(monitor._fields, monitor)) if monitor else None,
        "settle_ms": args.settle_ms, "flush": args.flush,
        "background": args.background, "elapsed_sec": round(time.time() - t0, 1),
    })
    print(f"\n{clean_shots} clean shot(s), {captured} capture(s) in "
          f"{time.time() - t0:.0f}s -> {os.path.join(args.root, 'raw')}")
    print(f"next: python data/collect.py warp --root {args.root}")
    return 0


# ------------------------------------------------------------------------ 3. warp

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
    if not cnts:
        print("warning: inset erased the contour; using the original.")
        return contour
    return max(cnts, key=cv2.contourArea)


def screen_contour(img, inset=2):
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
    # Imported here so the beam stage never pays for torch.
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

    The geometry comes from the clean shot, where the screen boundary is visible;
    every `pro` of that scene is warped with the identical mapping, which is what
    makes the pair pixel-aligned enough to train on.
    """

    def __init__(self, contour, size, mode="boundary", points=20):
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


def cmd_warp(args):
    """
    Rectify the raw captures into aligned clean/pro pairs at the model's input size.

    The screen is a trapezoid in the camera and the objects sit at a different scale
    in every capture, so nothing before this stage is trainable. After it, `clean`
    and `pro` share a pixel grid and the residual the model learns is only the
    projected light.
    """
    from projector_distortion.data import ori_id
    from projector_distortion.utils.visualize import warp_before_after

    ori_dir = args.ori or os.path.join(args.root, "raw", "ori")
    pro_dir = args.pro or os.path.join(args.root, "raw", "pro")
    out_clean = os.path.join(args.root, "clean")
    out_pro = os.path.join(args.root, "pro")
    debug_dir = os.path.join(args.root, "debug")

    ori_files = images_in(ori_dir)
    if not ori_files:
        raise SystemExit(
            f"no clean captures in {ori_dir}\n"
            f"    run: python data/collect.py capture --root {args.root}")
    for d in (out_clean, out_pro) + ((debug_dir,) if args.debug else ()):
        os.makedirs(d, exist_ok=True)

    pro_files = images_in(pro_dir)
    by_ori = {}
    for name, path in pro_files.items():
        by_ori.setdefault(ori_id(name), []).append((name, path))

    work = tuple(args.work_size)
    final = tuple(args.final_size)
    names = list(ori_files)[:args.limit] if args.limit else list(ori_files)
    print(f"{len(names)} scene(s), {len(pro_files)} capture(s) | warp={args.warp} "
          f"| {work[0]}x{work[1]} -> {final[0]}x{final[1]}")

    done_clean = done_pro = 0
    failed = []
    for name in names:
        oid = ori_id(name)
        img = cv2.imread(ori_files[name])
        if img is None:
            failed.append((name, "unreadable"))
            continue

        contour = screen_contour(img, args.inset)
        if contour is None:
            failed.append((name, "no screen boundary"))
            continue
        try:
            rectifier = Rectifier(contour, work, mode=args.warp, points=args.points)
        except ValueError as e:
            failed.append((name, str(e)))
            continue

        clean = shrink(rectifier(img), final)
        cv2.imwrite(os.path.join(out_clean, name), clean)
        done_clean += 1

        if args.debug:
            overlay = debug_overlay(img, contour, rectifier.src)
            corners = [tuple(int(v) for v in p) for p in rectifier.corners]
            cv2.imwrite(os.path.join(debug_dir, f"{oid}_warp.jpg"),
                        warp_before_after(overlay, clean, corners))

        for pro_name, pro_path in by_ori.get(oid, []):
            pro = cv2.imread(pro_path)
            if pro is None:
                failed.append((pro_name, "unreadable"))
                continue
            cv2.imwrite(os.path.join(out_pro, pro_name), shrink(rectifier(pro), final))
            done_pro += 1
        print(f"  {name}: {len(by_ori.get(oid, []))} capture(s) rectified", flush=True)

    # Against every clean shot, not just the ones --limit let through: a scene that
    # was merely skipped this run still has its clean image.
    orphans = sorted(set(by_ori) - {ori_id(n) for n in ori_files})
    if orphans:
        print(f"warning: {len(orphans)} oriId(s) have captures but no clean shot "
              f"(e.g. {orphans[0]}); those captures were skipped.")
    for name, why in failed:
        print(f"warning: {name}: {why}")

    write_meta(args.root, "warp", {
        "mode": args.warp, "points": args.points, "inset": args.inset,
        "work_size": list(work), "final_size": list(final),
        "clean": done_clean, "pro": done_pro,
        "failed": [{"file": n, "reason": w} for n, w in failed],
        "orphan_ori_ids": orphans,
    })
    print(f"\n{done_clean} clean + {done_pro} pro -> {args.root}")
    print(f"next: python demo.py --input {args.root} --gt {args.root}")
    return 0


# ----------------------------------------------------------------------- argparse

def add_camera_args(p):
    p.add_argument("--camera", type=int, default=0, help="webcam index")
    p.add_argument("--cam-width", type=int, default=1280)
    p.add_argument("--cam-height", type=int, default=960)
    p.add_argument("--cam-fps", type=int, default=30)
    p.add_argument("--cam-backend", default="auto",
                   choices=["auto", "any", "dshow", "msmf", "v4l2"])
    return p


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)

    check = sub.add_parser("check", help="list the displays and preview the webcam")
    add_camera_args(check)
    check.set_defaults(func=cmd_check)

    beam = sub.add_parser("beam", help="video -> beam frames (no hardware)")
    beam.add_argument("--root", default=None, help="session folder")
    beam.add_argument("--src", default=os.path.join(DATA_DIR, "live", "BeamVideo.mp4"),
                      help="video file, or a folder of them")
    beam.add_argument("--out", default=None, help="default: <root>/beam")
    beam.add_argument("--step", type=int, default=30,
                      help="keep every Nth frame (30 = one per second at 30fps)")
    beam.add_argument("--size", type=int, nargs=2, default=[1280, 720],
                      metavar=("W", "H"), help="0 0 keeps the source resolution")
    beam.add_argument("--quality", type=int, default=95)
    beam.add_argument("--limit", type=int, default=0, help="at most N frames per video")
    beam.add_argument("--video-index", type=int, default=1000,
                      help="first video's index in the filename")
    beam.set_defaults(func=cmd_beam)

    cap = sub.add_parser("capture", help="projector + webcam -> raw ori/pro captures")
    cap.add_argument("--root", default=None, help="session folder")
    cap.add_argument("--beam", default=None, help="default: <root>/beam")
    cap.add_argument("--background", default=DEFAULT_BACKGROUND,
                     help="image projected while the clean shot is taken")
    cap.add_argument("--screen", type=int, default=1,
                     help="monitor index for the projector (0 = primary; `check` "
                          "prints the table)")
    cap.add_argument("--rounds", type=int, default=1,
                     help="scene setups; each one starts with a new clean shot")
    cap.add_argument("--limit", type=int, default=0, help="beam frames per round")
    cap.add_argument("--shuffle", action="store_true",
                     help="random beam order, so a short round still spans the clip")
    cap.add_argument("--seed", type=int, default=42)
    cap.add_argument("--settle-ms", type=int, default=150,
                     help="wait after showing a frame before capturing it")
    cap.add_argument("--flush", type=int, default=3,
                     help="buffered camera frames to drop before each capture")
    cap.add_argument("--round-settle", type=float, default=2.0,
                     help="seconds between the clean shot and the first projection")
    cap.add_argument("--preview-every", type=int, default=10,
                     help="refresh the capture preview every N frames")
    cap.add_argument("--jpeg-quality", type=int, default=95)
    add_camera_args(cap)
    cap.set_defaults(func=cmd_capture)

    warp = sub.add_parser("warp", help="rectify raw captures into aligned pairs")
    warp.add_argument("--root", default=None, help="session folder")
    warp.add_argument("--ori", default=None, help="default: <root>/raw/ori")
    warp.add_argument("--pro", default=None, help="default: <root>/raw/pro")
    warp.add_argument("--warp", default="boundary",
                      choices=["boundary", "homography", "tps"],
                      help="boundary: 4 corners + the measured edge bow | homography: "
                           "corners only | tps: the legacy warp, needs OpenCV < 5")
    warp.add_argument("--points", type=int, default=20,
                      help="boundary correspondences (tps and the debug overlay)")
    warp.add_argument("--inset", type=int, default=2,
                      help="pixels to pull the boundary in, off the projection rim")
    warp.add_argument("--work-size", type=int, nargs=2, default=[1280, 720],
                      metavar=("W", "H"), help="rectification resolution")
    warp.add_argument("--final-size", type=int, nargs=2, default=[640, 360],
                      metavar=("W", "H"), help="saved resolution (the model input)")
    warp.add_argument("--limit", type=int, default=0, help="process at most N scenes")
    warp.add_argument("--no-debug", dest="debug", action="store_false",
                      help="skip the before/after overlays in <root>/debug")
    warp.set_defaults(func=cmd_warp, debug=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if hasattr(args, "root"):     # every stage but `check` works inside a session
        args.root = args.root or default_root()
        os.makedirs(args.root, exist_ok=True)
        print(f"session: {args.root}")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
