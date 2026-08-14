#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project the light frames and shoot each one. Needs the rig: a projector and a webcam.

    python Data.py capture --screen 2
    python Data.py capture --screen 2 --rounds 20 --limit 25
    python Data.py capture --screen 2 --settle-ms 2000     # slower, safer pairing

Reads <session.dir>/projected (make_light.py wrote it) and fills today's
<session.dir>/raw_<MMDD>; --raw adds to a specific capture folder instead.

A **round** is one scene setup:

    background goes up  ->  you arrange the objects  ->  you press 's'
      -> one surface shot, the restoration target for this round
      -> --limit light frames, drawn at random from the whole light folder,
         projected and captured one at a time
      -> next round

While you are arranging, the screen boundary is detected live and the rectified view
is shown beside the camera one, so a warp that will not resolve gets fixed at the rig
instead of at the warp stage. Keys in that preview:

    s   take the surface shot and start the round
    c   click the 4 corners by hand (when detection cannot find them)
    r   drop the clicked corners, go back to auto-detection
    q   abort

Whatever corners are in force when 's' is pressed are written to collect_meta.json,
and `Data.py warp` uses them instead of detecting again.

So the objects on the screen change once per round, and every capture in that round
carries the round's surfaceId - which is what ties many `distorted` files back to the
single `surface` they should be restored to. Each round draws afresh, so ten rounds
are lit by ten different sets rather than one set ten times.

Nothing is on a clock: the surface shot waits for the keypress. 'q' aborts.

Output is unrectified: raw/surface and raw/distorted. `Data.py warp` is what turns
those into trainable pairs.

`Data.py capture_warp` (this script with --warp) does both in one pass - it builds
one rectifier per round from the corners on screen at the surface shot, and writes
the rectified surface/, distorted/ and debug/ into warp_<MMDD>/ alongside the raw.
The raw is kept either way, so a round that came out wrong can still be redone with
`Data.py warp --review` without going back to the rig.
"""

import argparse
import os
import random
import sys
import time
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # for `common`
from common import (  # noqa: E402
    CAPTURE_VIEW, DISTORTED_PREFIX, LIGHT_PREFIX, PREVIEW, SURFACE_PREFIX,
    cfg, cfg_path, dated_suffix, images_in, jpeg_params, pick, projected_dir,
    resolve_raw, shrink, warp_dir, write_meta,
)


def grab_fresh(cam, flush):
    """Drop the frames the driver already had queued, then take the live one."""
    for _ in range(max(0, flush)):
        cam.grab()
    ok, frame = cam.read()
    return frame if ok else None


def settle_and_shoot(cam, settle_ms):
    """
    Hold whatever is on the projector for `settle_ms`, then keep the last frame.

    The camera is read for the entire window rather than slept through. Two reasons,
    and both of them are what a mismatched pair looks like:

    - A webcam hands back whatever is already queued. Sleeping lets that queue fill
      with pictures of the *previous* light frame; reading continuously keeps it
      drained, so the frame kept at the end is genuinely the current one.
    - Auto-exposure and white balance only adjust on frames the driver actually
      delivers. Between an inverted frame and a hue-rotated one the whole image
      changes brightness, and an AE loop that is never run never converges.

    Returns (frame, aborted). `frame` is None if the camera stopped answering.
    """
    deadline = time.perf_counter() + max(0, settle_ms) / 1000.0
    frame = None
    while True:
        # 1ms, not 0: this is also what repaints the projector window.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return frame, True
        ok, latest = cam.read()
        if ok:
            frame = latest
        elif frame is None:
            return None, False
        if time.perf_counter() >= deadline:
            return frame, False


HINT = "s shoot    c click corners    r auto-detect    q abort"

# One tile's width in the framing preview. Camera and rectified views sit side by side
# in a single window - deliberately one window and not two: cv2.waitKey only receives
# keys from whichever HighGUI window has focus, and with the fullscreen projector
# window already competing, a second preview window is enough to lose the keystroke.
TILE_W = 640
HINT_H = 34


def _framing_canvas(frame, corners, rectified, pinned):
    """Camera view (with the quad drawn on it) beside the rectified one, plus hints."""
    left = frame.copy()
    if corners is not None:
        colour = (0, 200, 255) if pinned is not None else (0, 255, 0)
        cv2.polylines(left, [corners.astype(int).reshape(-1, 1, 2)], True, colour, 2)
        for i, (x, y) in enumerate(corners.astype(int)):
            cv2.circle(left, (x, y), 6, (0, 255, 255), -1)
            cv2.putText(left, "TL TR BR BL".split()[i], (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        label, label_colour = ("clicked" if pinned is not None else "auto"), colour
    else:
        label, label_colour = "no screen boundary - press 'c'", (0, 0, 255)

    def tile(img):
        scale = TILE_W / img.shape[1]
        return cv2.resize(img, (TILE_W, max(1, round(img.shape[0] * scale))))

    left_t = tile(left)
    right_t = (tile(rectified) if rectified is not None
               else np.full((left_t.shape[0], TILE_W, 3), 24, np.uint8))
    if rectified is None:
        cv2.putText(right_t, "no warp yet", (16, right_t.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (90, 90, 90), 2)

    h = max(left_t.shape[0], right_t.shape[0])
    canvas = np.zeros((h + HINT_H, TILE_W * 2, 3), np.uint8)
    canvas[:left_t.shape[0], :TILE_W] = left_t
    canvas[:right_t.shape[0], TILE_W:] = right_t

    cv2.putText(canvas, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                label_colour, 2)
    cv2.putText(canvas, "rectified", (TILE_W + 12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (200, 200, 200), 2)
    cv2.putText(canvas, HINT, (12, h + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2)
    return canvas


def _live_warp(frame, inset, size, corners=None):
    """
    (corners, rectified) for the screen in this frame, or (None, None).

    `corners` skips detection and uses the given quad - that is the hand-clicked
    path. Homography rather than the boundary blend either way: it is the exact model
    for a flat screen, it builds in half a millisecond where the blend needs eighty,
    and this is a live check rather than the warp that gets written.
    """
    from warp import Rectifier, contour_from_corners, screen_contour

    contour = (contour_from_corners(corners) if corners is not None
               else screen_contour(frame, inset))
    if contour is None:
        return None, None
    try:
        rectifier = Rectifier(contour, size, mode="homography")
    except ValueError:
        return None, None
    return rectifier.corners, rectifier(frame)


# A quad smaller than this fraction of the frame is a mis-click, not a screen. Four
# points that land almost on top of each other still make a geometrically valid
# homography, so without this the round would be rectified into garbage in silence.
MIN_QUAD_FRAC = 0.02


def write_debug(surface_img, rectifier, rectified, path):
    """
    The before/after figure warp.py writes, written here too.

    Same evidence, produced at the same moment the geometry was decided: the screen
    boundary drawn on the camera frame, next to what came out of it. Without this a
    capture_warp session has no record of where the corners actually landed, and a
    round that went wrong is only visible by opening the pairs one at a time.
    """
    from projector_distortion.utils.visualize import warp_before_after
    from warp import contour_from_corners, debug_overlay

    overlay = debug_overlay(surface_img, contour_from_corners(rectifier.corners),
                            rectifier.src)
    corners = [tuple(int(v) for v in p) for p in rectifier.corners]
    cv2.imwrite(path, warp_before_after(overlay, rectified, corners))


def build_rectifier(corners, size, mode, frame_shape=None):
    """The round's camera -> flat screen mapping, or None if the corners are unusable."""
    if corners is None:
        return None
    from warp import Rectifier, contour_from_corners

    quad = np.asarray(corners, np.float32).reshape(4, 2)
    if frame_shape is not None:
        frac = abs(cv2.contourArea(quad)) / (frame_shape[0] * frame_shape[1])
        if frac < MIN_QUAD_FRAC:
            print(f"    the 4 corners cover only {100 * frac:.1f}% of the frame; "
                  f"that is a mis-click, not the screen.")
            return None
    try:
        return Rectifier(contour_from_corners(quad), size, mode=mode)
    except (ValueError, SystemExit) as e:
        print(f"    could not build the {mode} warp: {e}")
        return None


def shoot_surface(cam, flush, inset=2, work=(1280, 720)):
    """
    Hold at the background frame until the operator presses 's'.

    That shot is the scene with no content projected on it - the restoration target -
    so it is taken by hand: the objects have to be placed and the operator out of
    frame first.

    While it waits, the screen boundary is detected on every frame and the rectified
    view is shown beside the camera one. That is the whole point of doing it here: the
    operator is already standing at the rig, and a warp that will not resolve is
    something to fix by nudging the projector now, not to discover at the warp stage
    with the scene long since dismantled.

    When the detector cannot find the screen - a dark room, a low-contrast surface, a
    frame edge that runs off the sensor - 'c' freezes the current frame and takes the
    four corners by hand. Those stay pinned for the rest of the round, because they
    describe where the rig is pointing rather than what one frame happened to show;
    'r' goes back to detecting.

    The corners in force when 's' is pressed are returned with the shot, so warp.py
    can reuse them instead of re-detecting on its own.

    Returns (frame, corners); frame is None if 'q' was pressed. `corners` may be None
    when nothing was found and nothing was clicked - the shot is still taken, warp.py
    just falls back to detecting for itself.
    """
    from warp import click_corners

    print(f"  set the scene, then: {HINT}")
    pinned = None          # hand-clicked corners; None means keep auto-detecting
    cv2.namedWindow(PREVIEW)
    # The projector window is fullscreen and was raised a moment ago; without this the
    # preview can end up behind it, and keys go to whatever is actually in front.
    try:
        cv2.setWindowProperty(PREVIEW, cv2.WND_PROP_TOPMOST, 1)
    except cv2.error:
        pass                                  # older OpenCV builds lack the property
    try:
        while True:
            frame = grab_fresh(cam, 1)
            if frame is None:
                print("warning: camera read failed.")
                return None, None

            corners, rectified = _live_warp(frame, inset, work, corners=pinned)
            cv2.imshow(PREVIEW, _framing_canvas(frame, corners, rectified, pinned))

            key = cv2.waitKey(30) & 0xFF
            if key in (ord("c"), ord("C")):
                # Freeze this frame to click on; the live loop resumes with the
                # result pinned, so the preview shows what those corners really do.
                clicked = click_corners(frame.copy())
                if clicked is not None:
                    pinned = clicked
                    print("    corners clicked; 'r' returns to auto-detection.")
                continue
            if key in (ord("r"), ord("R")) and pinned is not None:
                pinned = None
                print("    back to auto-detection.")
                continue
            if key in (ord("s"), ord("S"), 13, 32):     # also enter / space
                shot = grab_fresh(cam, flush)
                # Clicked corners belong to the rig, not to one frame, so they carry
                # over to the kept shot. Auto ones are re-derived from that frame.
                final_corners = pinned
                if final_corners is None and shot is not None:
                    final_corners, _ = _live_warp(shot, inset, work)
                return shot, final_corners
            if key in (ord("q"), ord("Q"), 27):         # also esc
                return None, None
    finally:
        cv2.destroyWindow(PREVIEW)


def capture_round(cam, light_files, sid, distorted_dir, params, settle_ms, every,
                  rectifier=None, warped_dir=None, final=None):
    """
    Project every light frame once and capture it. Returns (saved, aborted).

    With `rectifier` set, each frame is also rectified into `warped_dir` as it is
    taken. The raw frame is still written either way - the whole point of keeping
    raw/ is that a warp can be redone later without going back to the rig.
    """
    from projector_distortion.pipeline.live import WINDOW

    saved = 0
    for i, path in enumerate(light_files, start=1):
        light = cv2.imread(path)
        if light is None:
            print(f"warning: could not read {path}; skipped.")
            continue

        cv2.imshow(WINDOW, light)
        frame, aborted = settle_and_shoot(cam, settle_ms)
        if aborted:
            print("  stopped by user.")
            return saved, True
        if frame is None:
            print("warning: camera read failed; ending the round.")
            return saved, True

        light_id = os.path.splitext(os.path.basename(path))[0]
        if light_id.startswith(LIGHT_PREFIX):
            light_id = light_id[len(LIGHT_PREFIX):]
        name = f"{DISTORTED_PREFIX}{sid}_{light_id}.jpg"
        if cv2.imwrite(os.path.join(distorted_dir, name), frame, params):
            saved += 1
        if rectifier is not None:
            cv2.imwrite(os.path.join(warped_dir, name),
                        shrink(rectifier(frame), final), params)

        if i % max(1, every) == 0 or i == 1:
            cv2.imshow(CAPTURE_VIEW, shrink(frame, (480, 360)))
            cv2.waitKey(1)
        if i % 25 == 0 or i == len(light_files):
            print(f"    {i}/{len(light_files)} projected, {saved} captured", flush=True)
    return saved, False


def cmd_capture(args):
    from projector_distortion.pipeline.live import (
        CAM_FPS, CAM_HEIGHT, CAM_WIDTH, WINDOW, open_webcam, place_window,
    )

    screen = pick(args.screen, "capture", "screen", default=1)
    camera = pick(args.camera, "capture", "camera", default=0)
    backend = cfg("capture", "cam_backend", default="auto")
    rounds = pick(args.rounds, "capture", "rounds", default=10)
    limit = pick(args.limit, "capture", "limit", default=50)
    settle_ms = pick(args.settle_ms, "capture", "settle_ms", default=1200)
    flush = int(cfg("capture", "flush", default=3))
    every = int(cfg("capture", "preview_every", default=10))
    round_settle = float(cfg("capture", "round_settle", default=2.0))
    background_path = cfg_path("capture", "background",
                               default="data/live/BaseBackGround.jpg")

    light_dir = projected_dir()
    light_pool = list(images_in(light_dir).values())
    if not light_pool:
        raise SystemExit(
            f"no light frames in {light_dir}\n"
            f"    run: python Data.py make_light")

    # Every round draws its own sample from the whole pool, so N scenes see N different
    # sets of projected frames instead of the same short list N times. A null seed
    # seeds from OS entropy, which is what makes two sessions accumulate rather than
    # repeat; set capture.seed in collect.yaml to make a run reproducible.
    per_round = min(limit, len(light_pool)) if limit else len(light_pool)
    rng = random.Random(cfg("capture", "seed"))

    surface_dir = os.path.join(args.raw, "surface")
    distorted_dir = os.path.join(args.raw, "distorted")
    os.makedirs(surface_dir, exist_ok=True)
    os.makedirs(distorted_dir, exist_ok=True)
    print(f"raw:       {args.raw}")

    # --warp additionally rectifies as it goes, into the warp_<MMDD> that Data.py warp
    # would fill - named after this raw folder, not after today, so the two stay
    # paired. raw is written either way, so a bad round can still be redone with
    # `Data.py warp --review` without touching the rig again.
    do_warp = args.warp
    warp_mode = cfg("warp", "mode", default="boundary")
    final = tuple(cfg("warp", "final_size", default=[640, 360]))
    warp_root = warp_dir(dated_suffix(args.raw))
    out_surface = os.path.join(warp_root, "surface")
    out_distorted = os.path.join(warp_root, "distorted")
    debug_dir = os.path.join(warp_root, "debug")
    want_debug = do_warp and bool(cfg("warp", "debug", default=True))
    if do_warp:
        os.makedirs(out_surface, exist_ok=True)
        os.makedirs(out_distorted, exist_ok=True)
        if want_debug:
            os.makedirs(debug_dir, exist_ok=True)
        print(f"warped:    {warp_root}")

    background = (cv2.imread(background_path)
                  if background_path and os.path.exists(background_path) else None)
    if background is None:
        print(f"warning: background image not found ({background_path}); "
              f"using a flat grey frame.")
        background = np.full((720, 1280, 3), 32, np.uint8)

    monitor = place_window(screen, background)
    cam = open_webcam(camera, backend)
    if cam is None:
        cv2.destroyAllWindows()
        raise SystemExit(
            f"cannot open webcam index {camera}\n"
            f"    try another --camera index, set capture.cam_backend in "
            f"configs/collect.yaml, and close any other app holding the camera.")
    # Many drivers ignore this; settle_and_shoot() is what actually keeps the frame
    # current, by reading for the whole settle window.
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    params = jpeg_params()
    print(f"{per_round} light frame(s) drawn per round from a pool of "
          f"{len(light_pool)} x {rounds} round(s) = {per_round * rounds} "
          f"pair(s)")
    print(f"  {settle_ms}ms per capture -> about "
          f"{per_round * settle_ms / 1000 / 60:.1f} min of shooting per round")

    # surfaceId -> the 4 screen corners as seen at the moment the surface was shot.
    # warp.py prefers these over detecting for itself: they were measured with the
    # operator watching the rectified preview, which is the one moment anyone can
    # tell a good boundary from a bad one.
    calibrated = {}
    inset = int(cfg("warp", "inset", default=2))
    work = tuple(cfg("warp", "work_size", default=[1280, 720]))

    done_rounds, surface_shots, captured, warped_rounds = 0, 0, 0, 0
    t0 = time.time()
    try:
        for r in range(rounds):
            print(f"\nround {r + 1}/{rounds}")
            cv2.imshow(WINDOW, background)
            cv2.waitKey(1)

            surface, corners = shoot_surface(cam, flush, inset, work)
            if surface is None:
                print("aborted before the surface shot.")
                break
            sid = datetime.now().strftime("%m%d%H%M%S")   # no '_': parsed as the id
            name = f"{SURFACE_PREFIX}{sid}.jpg"
            cv2.imwrite(os.path.join(surface_dir, name), surface, params)
            surface_shots += 1
            if corners is not None:
                calibrated[sid] = [[round(float(x), 2), round(float(y), 2)]
                                   for x, y in corners]
                print(f"  surface shot: {name}  (corners recorded)")
            else:
                print(f"  surface shot: {name}  (no boundary detected; warp will "
                      f"look for one itself)")

            # One rectifier per round, from the corners that were on screen when the
            # shot was taken. Every capture in the round goes through the identical
            # mapping, which is what keeps the pair pixel-aligned.
            rectifier = None
            if do_warp:
                rectifier = build_rectifier(corners, work, warp_mode, surface.shape)
                if rectifier is None:
                    print("    no usable corners: this round is captured raw only. "
                          "`Data.py warp` can still rectify it later.")
                else:
                    rectified = shrink(rectifier(surface), final)
                    cv2.imwrite(os.path.join(out_surface, name), rectified, params)
                    if want_debug:
                        write_debug(surface, rectifier, rectified,
                                    os.path.join(debug_dir, f"{sid}_warp.jpg"))
                    warped_rounds += 1
            # The preview window took focus, which can pull the projector window off
            # its monitor on Windows; put it back before anything is projected.
            place_window(screen, background, announce=False)
            time.sleep(round_settle)

            light_files = rng.sample(light_pool, per_round)
            saved, aborted = capture_round(cam, light_files, sid, distorted_dir,
                                           params, settle_ms, every,
                                           rectifier=rectifier,
                                           warped_dir=out_distorted, final=final)
            captured += saved
            done_rounds += 1
            if aborted:
                break
    except KeyboardInterrupt:
        print("\ninterrupted.")
    finally:
        cam.release()
        cv2.destroyAllWindows()

    write_meta(args.raw, "capture", {
        "light_dir": light_dir, "light_pool": len(light_pool),
        "light_frames_per_round": per_round, "seed": cfg("capture", "seed"),
        "rounds_requested": rounds, "rounds_done": done_rounds,
        "surface_shots": surface_shots, "captures": captured,
        "camera": {"index": camera, "backend": backend,
                   "requested": [CAM_WIDTH, CAM_HEIGHT, CAM_FPS]},
        "screen_index": screen,
        "monitor": dict(zip(monitor._fields, monitor)) if monitor else None,
        "settle_ms": settle_ms, "flush": flush,
        "background": background_path, "elapsed_sec": round(time.time() - t0, 1),
        "corners": calibrated,
        "warped_while_capturing": bool(do_warp), "warped_rounds": warped_rounds,
        "debug": want_debug,
        "warp_mode": warp_mode if do_warp else None,
        "final_size": list(final) if do_warp else None,
    })
    print(f"\n{surface_shots} surface shot(s), {captured} capture(s) in "
          f"{time.time() - t0:.0f}s -> {args.raw}")
    if do_warp:
        print(f"{warped_rounds}/{done_rounds} round(s) also rectified "
              f"({warp_mode}, {final[0]}x{final[1]}) -> {warp_root}")
        if warped_rounds < done_rounds:
            print("run `python Data.py warp` to rectify the rounds that were not.")
        else:
            print(f"next: python demo.py --input {warp_root}")
    else:
        print(f"next: python Data.py warp --raw {args.raw}")
    return 0


def build_parser():
    p = argparse.ArgumentParser(
        prog="Data.py capture", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default=None,
                   help="where the captures go (default: <session.dir>/raw_<MMDD>, "
                        "today's)")
    p.add_argument("--screen", type=int, default=None,
                   help="monitor index for the projector, 0 = primary "
                        "(default: capture.screen in collect.yaml; `check` prints "
                        "the table)")
    p.add_argument("--camera", type=int, default=None,
                   help="webcam index (default: capture.camera)")
    p.add_argument("--rounds", type=int, default=None,
                   help="scene setups; each starts with a fresh surface shot "
                        "(default: capture.rounds)")
    p.add_argument("--limit", type=int, default=None,
                   help="light frames drawn per round, at random from the whole "
                        "light folder. 0 uses every frame (default: capture.limit)")
    p.add_argument("--warp", action="store_true",
                   help="also rectify as you shoot, into warp_<MMDD>/surface and "
                        "warp_<MMDD>/distorted. raw_<MMDD>/ is written either way "
                        "(`Data.py capture_warp` is this flag)")
    p.add_argument("--settle-ms", type=int, default=None,
                   help="milliseconds a light frame is held before its capture is "
                        "kept; the camera is read throughout, so this is also how "
                        "long auto-exposure gets to re-converge "
                        "(default: capture.settle_ms)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    # capture makes today's raw folder; --raw adds to a specific one instead.
    args.raw = resolve_raw(args.raw, create=True)
    return cmd_capture(args)


if __name__ == "__main__":
    raise SystemExit(main())
