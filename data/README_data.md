# data/

Everything about the dataset: what ships, how filenames tie the views together, the label
format, and the two tools that build more of it.

Script options for `demo.py` / `evaluate.py` / `train.py` are in
[README_running.md](../README_running.md).

- [What ships](#what-ships)
- [The three splits](#the-three-splits)
- [Filename convention](#filename-convention)
- [Layouts](#layouts)
- [Label format](#label-format)
- [`collect.py` — build a dataset with the rig](#collectpy--build-a-dataset-with-the-rig)
- [`record.py` — project and record, no models](#recordpy--project-and-record-no-models)
- [Swapping in real data](#swapping-in-real-data)

---

## What ships

A dataset that runs out of the box, split three ways under `SampleData/`. It is tracked in
git, so it is there right after a clone.

```
data/
├── collect.py              build your own dataset with a projector and a webcam
├── record.py               project a clip and record the camera's view (no models)
├── SampleData/             the dataset, split for train / validation / test
│   ├── sample_train/       what train.py fits on          180 MiB
│   │   ├── distorted/  1,000 files, 640×360
│   │   ├── light/        990 files, 1280×720 · 854×480 · 856×480
│   │   └── surface/       20 files, 640×360     — no labels/, training needs none
│   ├── sample_eval/        what demo.py and evaluate.py default to   4.2 MiB
│   │   ├── distorted/     22 files, 640×360
│   │   ├── light/         22 files, 1280×720 and 854×480
│   │   ├── surface/       10 files, 640×360
│   │   └── labels/        10 files, YOLO txt, 107 boxes
│   ├── sample_test/        held out; only evaluate.py --input reaches it   44 MiB
│   │   ├── distorted/    200 files, 640×360
│   │   ├── light/        198 files, 854×480
│   │   ├── surface/        5 files, 640×360
│   │   └── labels/         7 files, LabelMe json, 53 boxes over those 5 scenes
│   └── sample_video/       two short clips, an alternative to BeamVideo    23 MiB
│                           4,262 and 5,381 frames @30fps, 854×480 / 856×480
├── live/                   input for demo.py --live and record.py
│   ├── BeamVideo.mp4       clip to play through the projector
│   │                       5,858 frames @30fps = 3.3 min, 854×480, 55 MiB
│   └── BaseBackGround.jpg  background shown during calibration, 1280×960
└── recordings/             where record.py writes (git-ignored)
```

| Path | Used as |
|---|---|
| `<split>/distorted/` | Model input ch 0:3 — the camera's view of the projected screen |
| `<split>/light/` | Model input ch 3:6 — the frame the projector emitted |
| `<split>/surface/` | Training target, and the PSNR/SSIM reference |
| `<split>/labels/` | The mAP reference — YOLO txt or LabelMe json |

Every split is self-contained: `surface/` and `labels/` sit beside `distorted/`, so
`--input` and `--gt` take the same path. Both are optional. Without them a run still works,
it just cannot be scored.

Image sizes need not match. Everything is resized to the model's `input_size` (640×360 by
default), which is why `light/` mixes 1280×720, 854×480 and 856×480 with no special
handling.

`data/` is 305 MiB tracked. `live/BeamVideo.mp4` (55 MiB) and `SampleData/sample_video/`
(23 MiB) are only needed by `--live` and `record.py`; delete them if you use neither.

---

## The three splits

The scenes do not overlap. No `surfaceId` in `sample_train` appears in `sample_eval` or
`sample_test`, so a checkpoint trained on the first is scored on scenes it has never seen.

| Split | Pairs | Scenes | Labels | Read by |
|---|---|---|---|---|
| `sample_train` | 1,000 (994 pair a light) | 20 | none | `train.py`, via `train.data` in [restoration.yaml](../projector_distortion/configs/restoration.yaml) |
| `sample_eval` | 22 | 10 | YOLO txt, 107 boxes | `demo.py` and `evaluate.py` — the default for both |
| `sample_test` | 200 | 5 | LabelMe json, 53 boxes | nothing by default; pass `--input`/`--gt` |

```bash
python train.py                                          # sample_train
python evaluate.py                                       # sample_eval
python evaluate.py --input data/SampleData/sample_test \
                   --gt    data/SampleData/sample_test   # sample_test
```

Two notes on the counts. Six `sample_train` captures reference a light frame the split does
not carry; they are skipped with a warning and 994 triplets remain. `sample_test/labels/`
holds two extra annotations (`Ori0428095917`, `Ori0531135441`) whose scenes have no
`surface`/`distorted` in the split — unmatched labels are simply never loaded, which is why
the box count is 53 and not 80.

`sample_train` and `sample_test` were collected before the filename rename, so they carry
the `projected_` / `output_video_` / `Ori` spelling; `sample_eval` carries the current
`distorted_` / `light_` / `surface_` one. Both are read the same way — see below.

---

## Filename convention

Three views of one moment are tied together by ids inside the filename. No counts or
paths are hard-coded anywhere, so real data drops in as long as it follows this.

```
distorted_0409001429_0404023332_294_75.jpg
          └───┬────┘ └──────┬───────┘
           surfaceId      lightId

  → sample_eval/surface/surface_0409001429.jpg           (ground truth screen)
  → sample_eval/labels/surface_0409001429.txt            (detection ground truth)
  → sample_eval/light/light_0404023332_294_75.jpg        (frame the projector emitted)
```

| Role | Filename | Pre-rename spelling | In the model |
|---|---|---|---|
| `distorted` | `distorted_<surfaceId>_<lightId>.jpg` | `projected_…` | input ch 0:3 |
| `light` | `light_<lightId>.jpg` | `output_video_…` | input ch 3:6 |
| `surface` | `surface_<surfaceId>.jpg` | `Ori<surfaceId>.jpg` | training target / PSNR·SSIM reference |
| `label` | `surface_<surfaceId>.txt` or `.json` | `Ori<surfaceId>.…` | detection mAP reference |

- No `_` in `surfaceId`. Everything up to the first `_` is taken as the surfaceId.
- `lightId` may contain `_`. Above it is `0404023332_294_75`.
- One `surface` backing several `distorted` captures is normal. In `sample_eval` 10 surface
  images back 22 `distorted` files, 1–3 each; in `sample_test` 5 back 200.
- `light` is 1:1 with `distorted`.
- Recognised extensions: `.jpg` `.jpeg` `.png` `.bmp`.
- A `distorted` with no matching `light` is skipped with a warning. The run does not fail.
- `surface` and `label` are optional. A missing `surface` only drops PSNR/SSIM, and
  `evaluate.py` scores just the samples that have both.
- The pre-rename column is read but never written. It is what `sample_train` and
  `sample_test` carry, and what an already-collected session keeps working under.

---

## Layouts

Auto-detected, by which folder exists.

| Name | Folders |
|---|---|
| `flat` (used here) | `distorted/` `light/` (+ `surface/`) |
| `legacy-flat` | `pro/` `beam/` (+ `clean/`) — sessions collected before the rename |
| `research` | `ProjectorImage/` `BeamImage/` `OriginalImage/` |

```bash
python demo.py --input data/SampleData/sample_eval                # flat
python demo.py --input /path/to/WarpData_0520                     # research
```

Images sitting loose in one folder are handled as `mixed`, split by the `distorted_` and
`light_` prefixes.

---

## Label format

Two are read, picked by the file's extension. Both end up as the same pixel boxes, so a
split may use either and `--gt` never needs telling which.

### YOLO `.txt` — `sample_eval`

One line per box, `<cls_id> <cx> <cy> <w> <h>`, all normalised to 0–1.

```
0 0.139406 0.263969 0.122381 0.207129
1 0.505193 0.162010 0.127800 0.211946
5 0.311010 0.202954 0.125993 0.199101
```

`cls_id` runs `0..16` and follows the `names` order in
[configs/detection.yaml](../projector_distortion/configs/detection.yaml) — 11 fruits
(Apple … Watermelon), then 6 animals (Cat … Snake). All 17 classes appear in
`sample_eval`'s labels.

### LabelMe `.json` — `sample_test`

What the LabelMe annotator writes, unmodified. Boxes are `rectangle` shapes with two
corner points in **absolute pixels**, and the class is a *name*, not an id.

```json
{
  "imageWidth": 640, "imageHeight": 360,
  "shapes": [
    {"label": "BlueBerry", "shape_type": "rectangle",
     "points": [[63.64, 27.97], [144.56, 105.43]]}
  ]
}
```

- Points are read against the file's own `imageWidth`/`imageHeight` and rescaled to
  whatever `input_size` the run uses, so re-running at 480×270 needs no re-annotation.
- The corners are ordered on read, so a box drawn bottom-right to top-left is fine.
- `label` is matched against the detector's own class names — the YOLO checkpoint's list,
  or `names` from `detection.yaml` for SSD. A name outside that list is skipped with a
  warning: scoring a class the detector cannot emit would only ever count as a false
  negative.
- Only `rectangle` shapes are used. Polygons and points are ignored.

If both `surface_<id>.txt` and `surface_<id>.json` exist for one scene, the `.txt` wins.

---

## `collect.py` — build a dataset with the rig

The sample set above was made this way. Four stages, in order. `check` and `capture` drive
the projector and the webcam; `light` and `warp` are plain file work and run anywhere.

```bash
python data/collect.py check                                  # monitors + webcam
python data/collect.py light   --src data/live/BeamVideo.mp4  # video -> light frames
python data/collect.py capture --screen 2 --rounds 3          # project and shoot
python data/collect.py warp                                   # rectify into pairs
```

Everything lands in one session folder, already in the layout the rest of the repo reads.
`--root` moves it; the default is one folder per day, so a second run of a stage extends
the first.

```
data/collected_<MMDD>/
├── light/      light_<lightId>.jpg                    [light]    what the projector emits
├── raw/        surface/surface_<surfaceId>.jpg        [capture]  camera frames, unrectified
│               distorted/distorted_<surfaceId>_<lightId>.jpg     [capture]
├── surface/    surface_<surfaceId>.jpg                [warp]     rectified, 640×360
├── distorted/  distorted_<surfaceId>_<lightId>.jpg    [warp]     rectified, 640×360
├── debug/      <surfaceId>_warp.jpg                   [warp]     before/after evidence
└── collect_meta.json                                  every stage's settings and counts
```

```bash
python demo.py  --input data/collected_0803
python train.py --data-root data/collected_0803
```

### Camera flags

`check` and `capture` share these.

| Option | Default |
|---|---|
| `--camera N` | `0` |
| `--cam-width` · `--cam-height` | `1280` · `960` |
| `--cam-fps` | `30` |
| `--cam-backend` | `auto` · `any` · `dshow` · `msmf` · `v4l2` |

### `check`

Lists the displays and opens the webcam, so `capture` is not the first attempt.

```
  --screen 0 -> 2560x1440 at (0,0) (primary)  \\.\DISPLAY1
  --screen 1 -> 1920x1080 at (2560,0)         \\.\DISPLAY2
```

### `light` — video → light frames

| Option | Default | Meaning |
|---|---|---|
| `--src <path>` | `data/live/BeamVideo.mp4` | Video file, or a folder of them |
| `--out <dir>` | `<root>/light` | — |
| `--step N` | `30` | Keep every Nth frame. 30 = one per second at 30fps |
| `--size W H` | `1280 720` | `0 0` keeps the source resolution |
| `--quality N` | `95` | JPEG quality |
| `--limit N` | `0` | At most N frames per video |
| `--video-index N` | `1000` | The first video's index in the filename |

### `capture` — projector + webcam → raw captures

| Option | Default | Meaning |
|---|---|---|
| `--screen N` | `1` | Monitor the projector is attached to. `check` prints the table |
| `--light <dir>` | `<root>/light` | — |
| `--background <path>` | `data/live/BaseBackGround.jpg` | Projected while the surface shot is taken |
| `--rounds N` | `1` | Scene setups; each starts with a fresh surface shot |
| `--limit N` | `0` | Light frames per round |
| `--shuffle` | off | Random light order, so a short round still spans the clip |
| `--seed N` | `42` | For `--shuffle` |
| `--settle-ms N` | `150` | Wait after showing a frame before capturing it |
| `--flush N` | `3` | Buffered camera frames to drop before each capture |
| `--round-settle F` | `2.0` | Seconds between the surface shot and the first projection |
| `--preview-every N` | `10` | Refresh the capture preview every N frames |
| `--jpeg-quality N` | `95` | — |

**How a round works.** `capture` projects the background and waits for you to press `s`.
That shot becomes `surface` — the scene with nothing projected on it, so place the objects
and step out of frame first. Its timestamp becomes the `surfaceId`, and every capture of
that round carries it. That is how many `distorted` files end up pointing at one
`surface`. Then each light frame is projected once and captured once. `--rounds N`
repeats with a new scene.

**Timing.** `--settle-ms` and `--flush` are what keep a capture matched to the frame that
caused it. If `distorted` looks like the *previous* light frame, raise both.

### `warp` — rectify into aligned pairs

| Option | Default | Meaning |
|---|---|---|
| `--warp boundary` | default | 4 corners + the measured edge bow |
| `--warp homography` | — | Corners only. For a flat screen with a clean boundary |
| `--warp tps` | — | The legacy thin-plate spline. Needs `opencv-python<5` |
| `--surface <dir>` · `--distorted <dir>` | `<root>/raw/surface` · `<root>/raw/distorted` | — |
| `--points N` | `20` | Boundary correspondences (tps and the debug overlay) |
| `--inset N` | `2` | Pixels to pull the boundary in, off the bright projection rim |
| `--work-size W H` | `1280 720` | Rectification resolution |
| `--final-size W H` | `640 360` | Saved resolution — the model input |
| `--limit N` | `0` | Process at most N scenes |
| `--no-debug` | off | Skip the before/after overlays in `<root>/debug/` |

**Why it exists.** In the camera the screen is a trapezoid and the objects sit at a
different scale in every capture, so nothing before this stage is trainable. `warp` finds
the screen boundary in the surface shot, then rectifies that scene's surface frame *and*
all its captures with the identical mapping. The pair ends up sharing a pixel grid, and
the only difference left is the projected light.

`boundary` is exact for a flat screen seen off-axis and still right when the edges bow.
OpenCV 5 removed the shape module, which is what `tps` needs.

Check `<session>/debug/<surfaceId>_warp.jpg` on a short trial session before committing to a
long one. It shows the detected boundary, the sampled points and the rectified result
together.

### Labels are not collected

`surface/` has to be annotated by hand into `<session>/labels/surface_<surfaceId>.txt` — or
`.json`, if you annotate in LabelMe — before `evaluate.py` can score mAP. Restoration
training and PSNR/SSIM need no labels, which is why `sample_train` ships without any.

---

## `record.py` — project and record, no models

`collect.py` builds a *dataset*: still pairs, rectified and id-matched. `record.py` builds
a *video*. It projects a clip at its own fps and records the camera's view as one mp4.
No restoration, no detection, no weights loaded — so it runs on a machine that has none.

Use it to capture raw distorted footage before there is a checkpoint, or to prove a rig
works end to end.

| | Default | Change it with |
|---|---|---|
| Projected clip | `data/live/BeamVideo.mp4` | `--clip <path>` |
| Background | `data/live/BaseBackGround.jpg` | `--background <path>` |
| Output | `data/recordings/rec_<MMDDHHMMSS>.mp4` + a `.json` beside it | `--out <path>` |

| Option | Default | Meaning |
|---|---|---|
| `--screen N` | `1` | Monitor the projector is attached to. `0` = primary |
| `--loop` | off | Restart the clip instead of stopping at its end |
| `--seconds F` | `0` | Stop after N seconds. `0` = no limit |
| `--max-frames N` | `0` | Stop after N projected frames |
| `--start-delay F` | `0` | Hold the background this long before recording starts |
| `--fps F` | `0` | mp4 header fps. `0` = measure the camera's real rate |
| `--fps-probe N` | `24` | Frames used to measure the rate. Buffered, not dropped |
| `--codec <fourcc>` | `mp4v` | `avc1`, `XVID`, … |
| `--rec-size W H` | `0 0` | Resize before encoding. `0 0` keeps the camera resolution |
| `--warp` | off | Calibrate once and record the rectified screen |
| `--manual-calib` | off | With `--warp`: click the 4 corners |
| `--calib-settle F` | `0.8` | Seconds to wait after each calibration flash |
| `--no-preview` | off | Skip the small camera preview window |
| `--preview-every N` | `5` | Refresh the preview every N projected frames |

Camera flags are the same as `collect.py`'s.

```bash
python data/record.py --screen 2 --seconds 30
python data/record.py --clip data/SampleData/sample_video/mIni_Video_1.mp4 --loop
python data/record.py --warp --rec-size 640 360
```

Recording starts as soon as the window is up — no keypress. `q` stops early, as do the end
of the clip, `--seconds` and `--max-frames`.

The header fps is **measured**, not requested. Webcams routinely deliver something other
than what they were asked for and report a third number, and a wrong header is exactly
what makes a recording play back in fast or slow motion. The probe frames are buffered
until the rate is known, so measuring costs no footage.

The `.json` beside the mp4 records the clip, camera and monitor settings the run actually
used, plus the measured rate and how many frames the encoder dropped.

---

## Swapping in real data

```bash
# 1) fill the same structure under data/SampleData/, then just run
python demo.py
python evaluate.py

# 2) or point at the original dataset directly
python demo.py --input /mnt/.../WarpData_0520
```

Nothing hard-codes a split name. `--input`/`--gt` and `--data-root` take any folder in a
recognised layout, so a real dataset can keep its own directory names and split however it
likes; `data/SampleData/` is only where the bundled one happens to sit.

Training reads its three directories from `train.data` in
[configs/restoration.yaml](../projector_distortion/configs/restoration.yaml). Each takes a
directory, a glob, or a list. Precedence, the `--data-root` fallback and the surface-target
search order: [README_running.md](../README_running.md#where-the-training-data-comes-from).

---

[한국어](README_data.ko.md) · [← README](../README.md)
