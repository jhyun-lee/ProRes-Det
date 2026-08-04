# data/

A toy dataset that runs out of the box. Real data drops in with no code change as
long as it follows the filename convention below — no counts or paths are hard-coded.

Unlike `weights/`, this directory is tracked in git, so it is there right after a
clone.

```
data/
├── collect.py              build your own dataset with a projector and a webcam
├── record.py               project a clip and record the camera's view (no models)
├── sample_input/           input *and* ground truth for demo / evaluate / train
│   ├── pro/    projected_<oriId>_<beamId>.jpg   22 files, 640×360
│   ├── beam/   output_video_<beamId>.jpg        22 files, mixed sizes
│   ├── clean/  Ori<oriId>.jpg      10 files, 640×360  ← training target / PSNR·SSIM ref
│   └── labels/ Ori<oriId>.txt      10 files, 107 boxes total  ← mAP ref (YOLO format)
├── live/                   input for demo.py --live
│   ├── BeamVideo.mp4       clip to play through the projector
│   │                       5,858 frames @30fps = 3.3 min, 854×480, 55 MiB
│   └── BaseBackGround.jpg  background shown during calibration, 1280×960
├── sample_video/           two short clips, an alternative to BeamVideo
└── recordings/             where record.py writes (git-ignored)
```

`clean/` and `labels/` sit inside `sample_input/`, so `--input` and `--gt` take the
same path. Both are optional: without `clean/` the run still works, it just cannot be
scored.

Image sizes need not match. The pipeline resizes everything to the model's
`input_size` (640×360 by default), which is why `beam/` mixes 854×480 and 1280×720
without any special handling.

## Filename convention

Three views of one moment are tied together by ids inside the filename.

```
projected_0409001429_0404023332_294_75.jpg
          └───┬────┘ └──────┬───────┘
            oriId         beamId

  → sample_input/clean/Ori0409001429.jpg                  (ground truth screen)
  → sample_input/labels/Ori0409001429.txt                 (detection ground truth)
  → sample_input/beam/output_video_0404023332_294_75.jpg  (frame the projector emitted)
```

| Role | Filename | In the model |
|---|---|---|
| `pro` | `projected_<oriId>_<beamId>.jpg` | input ch 0:3 — camera view of the projected screen |
| `beam` | `output_video_<beamId>.jpg` | input ch 3:6 — the frame the projector emitted |
| `clean` | `Ori<oriId>.jpg` | training target / PSNR·SSIM reference |
| `label` | `Ori<oriId>.txt` | detection mAP reference |

- No `_` in `oriId` — everything up to the first `_` is taken as the oriId.
  `beamId` may contain `_` (the example above is `0404023332_294_75`).
- One `clean` backing several `pro` captures is normal. Here 10 clean images back 22
  `pro` files, 1–3 each.
- `beam` is 1:1 with `pro`.
- Recognised extensions: `.jpg` `.jpeg` `.png` `.bmp`.
- A `pro` with no matching `beam` is skipped with a warning — the run does not fail.
- `clean` and `label` are optional. Missing `clean` only drops PSNR/SSIM; `evaluate.py`
  scores just the samples that have both.

## Layouts

Two are auto-detected by which folder exists.

| Name | Folders |
|---|---|
| `flat` (used here) | `pro/` `beam/` (+ `clean/`) |
| `research` | `ProjectorImage/` `BeamImage/` `OriginalImage/` |

```bash
python demo.py --input data/sample_input --gt data/sample_input   # flat
python demo.py --input /path/to/WarpData_0520                     # research, auto-detected
```

If images sit loose in one folder it is handled as `mixed`, split by the `projected_`
and `output_video_` prefixes.

## Label format

Standard YOLO — one line per box, `<cls_id> <cx> <cy> <w> <h>`, all normalised to 0–1.

```
0 0.139406 0.263969 0.122381 0.207129
1 0.505193 0.162010 0.127800 0.211946
5 0.311010 0.202954 0.125993 0.199101
```

`cls_id` runs `0..16` and follows the `names` order in
[configs/detection.yaml](../projector_distortion/configs/detection.yaml) — 11 fruits
(Apple … Watermelon) then 6 animals (Cat … Snake). All 17 classes appear in the bundled
labels.

## Collecting your own — `collect.py`

The sample set above was made this way. Four stages; `check` and `capture` need the rig,
`beam` and `warp` are plain file work.

```bash
python data/collect.py check                                     # monitors + webcam
python data/collect.py beam    --src data/live/BeamVideo.mp4     # video -> beam frames
python data/collect.py capture --screen 2 --rounds 3             # project and shoot
python data/collect.py warp                                      # rectify into pairs
```

One session folder holds every stage, already in the layout the rest of the repo reads:

```
data/collected_<MMDD>/
├── beam/   output_video_<beamId>.jpg           [beam]     what the projector emits
├── raw/    ori/Ori<oriId>.jpg                  [capture]  camera frames, unrectified
│           pro/projected_<oriId>_<beamId>.jpg  [capture]
├── clean/  Ori<oriId>.jpg                      [warp]     rectified, 640×360
├── pro/    projected_<oriId>_<beamId>.jpg      [warp]     rectified, 640×360
├── debug/  <oriId>_warp.jpg                    [warp]     before/after evidence
└── collect_meta.json                           every stage's settings and counts
```

```bash
python demo.py     --input data/collected_0803 --gt data/collected_0803
python train.py --data-root data/collected_0803
```

**How a round works.** `capture` projects the background, waits for you to press `s`, and
that shot becomes `clean` — the scene with nothing projected on it, so place the objects
and step out of frame first. Its timestamp becomes the `oriId`, and every capture of that
round carries it, which is how many `pro` files end up pointing at one `clean`. Then each
beam frame is projected once and captured once. `--rounds N` repeats with a new scene.

**Why `warp` exists.** In the camera the screen is a trapezoid and the objects sit at a
different scale in every capture, so nothing before this stage is trainable. `warp` finds
the screen boundary in the clean shot and rectifies that scene's clean frame *and* all its
captures with the identical mapping, so the pair shares a pixel grid and the only
difference left is the projected light. Default `--warp boundary` is the 4-corner
homography plus a correction from the measured edge curves — exact for a flat screen seen
off-axis, and still right when the edges bow. `--warp tps` reproduces the older thin-plate
spline warp, but needs `opencv-python<5` (OpenCV 5 removed the shape module).

Check `debug/<oriId>_warp.jpg` on a short trial session before committing to a long one.

**Timing.** `--settle-ms` (wait after showing a frame) and `--flush` (buffered camera
frames to drop) are what keep a capture matched to the frame that caused it. If `pro`
looks like the *previous* beam frame, raise both.

**Labels are not collected.** `clean/` has to be annotated by hand into
`<session>/labels/Ori<oriId>.txt` before `evaluate.py` can score mAP. Restoration
training and PSNR/SSIM need no labels.

Full option tables: [README_running.md](../README_running.md).

## Just recording — `record.py`

`collect.py` builds a *dataset*: still pairs, rectified and id-matched. `record.py`
builds a *video*: it projects a clip and records the camera's view as one mp4, with no
restoration, no detection and no weights loaded.

```bash
python data/record.py --screen 2 --seconds 30
python data/record.py --warp --rec-size 640 360     # record the rectified screen
```

Output is `data/recordings/rec_<MMDDHHMMSS>.mp4` plus a `.json` holding the settings the
run actually used and the measured capture rate. The directory is git-ignored. Use it
to capture raw distorted footage before there is a checkpoint, or to check a rig end to
end. Full option table: [README_running.md](../README_running.md).

## Swapping in real data

```bash
# 1) fill the same structure and just run
python demo.py
python evaluate.py

# 2) or point at the original dataset directly
python train.py --data-root /mnt/.../0_ImageData/1_WarpData_0520 --epochs 30
```

Training requires `clean` targets. `train.py` searches this order and reads whichever
exists — no symlink or copy needed:

```
--data-root/OriginalImage/   →   --data-root/clean/   →   --gt/clean/
```


## Size

`data/` is 82 MiB tracked, almost all of it video: `live/BeamVideo.mp4` (55 MiB) and
`sample_video/` (23 MiB). The dataset proper is 4.4 MiB. Delete the clips if you never
use `--live` or `record.py`.

---

[한국어](README_data.ko.md) · [← README](../README.md)
