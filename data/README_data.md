# data/

Everything about the dataset: what ships, how filenames tie the views together, the label
format, and the two tools that build more of it.

Script options for `demo.py` / `evaluate.py` / `train.py` are in
[README_running.md](../README_running.md).

- [What ships](#what-ships)
- [Filename convention](#filename-convention)
- [Layouts](#layouts)
- [Label format](#label-format)
- [`collect.py` — build a dataset with the rig](#collectpy--build-a-dataset-with-the-rig)
- [`record.py` — project and record, no models](#recordpy--project-and-record-no-models)
- [Swapping in real data](#swapping-in-real-data)

---

## What ships

A toy dataset that runs out of the box. It is tracked in git, so it is there right after a
clone.

```
data/
├── collect.py              build your own dataset with a projector and a webcam
├── record.py               project a clip and record the camera's view (no models)
├── sample_input/           input *and* ground truth for demo / evaluate / train
│   ├── pro/    projected_<oriId>_<beamId>.jpg   22 files, 640×360
│   ├── beam/   output_video_<beamId>.jpg        22 files, 1280×720 and 854×480
│   ├── clean/  Ori<oriId>.jpg                   10 files, 640×360
│   └── labels/ Ori<oriId>.txt                   10 files, 107 boxes
├── live/                   input for demo.py --live and record.py
│   ├── BeamVideo.mp4       clip to play through the projector
│   │                       5,858 frames @30fps = 3.3 min, 854×480, 55 MiB
│   └── BaseBackGround.jpg  background shown during calibration, 1280×960
├── sample_video/           two short clips, an alternative to BeamVideo
│                           4,262 and 5,381 frames @30fps, 854×480, 23 MiB
└── recordings/             where record.py writes (git-ignored)
```

| Path | Used as |
|---|---|
| `sample_input/pro/` | Model input ch 0:3 — the camera's view of the projected screen |
| `sample_input/beam/` | Model input ch 3:6 — the frame the projector emitted |
| `sample_input/clean/` | Training target, and the PSNR/SSIM reference |
| `sample_input/labels/` | The mAP reference, YOLO format |

`clean/` and `labels/` sit inside `sample_input/`, so `--input` and `--gt` take the same
path. Both are optional. Without them a run still works, it just cannot be scored.

Image sizes need not match. Everything is resized to the model's `input_size` (640×360 by
default), which is why `beam/` mixes 1280×720 and 854×480 with no special handling.

`data/` is 81 MiB tracked, almost all of it video — `live/BeamVideo.mp4` (55 MiB) and
`sample_video/` (23 MiB). The dataset proper is 4.2 MiB. Delete the clips if you never use
`--live` or `record.py`.

---

## Filename convention

Three views of one moment are tied together by ids inside the filename. No counts or
paths are hard-coded anywhere, so real data drops in as long as it follows this.

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
| `pro` | `projected_<oriId>_<beamId>.jpg` | input ch 0:3 |
| `beam` | `output_video_<beamId>.jpg` | input ch 3:6 |
| `clean` | `Ori<oriId>.jpg` | training target / PSNR·SSIM reference |
| `label` | `Ori<oriId>.txt` | detection mAP reference |

- No `_` in `oriId`. Everything up to the first `_` is taken as the oriId.
- `beamId` may contain `_`. Above it is `0404023332_294_75`.
- One `clean` backing several `pro` captures is normal. Here 10 clean images back 22 `pro`
  files, 1–3 each.
- `beam` is 1:1 with `pro`.
- Recognised extensions: `.jpg` `.jpeg` `.png` `.bmp`.
- A `pro` with no matching `beam` is skipped with a warning. The run does not fail.
- `clean` and `label` are optional. A missing `clean` only drops PSNR/SSIM, and
  `evaluate.py` scores just the samples that have both.

---

## Layouts

Two are auto-detected, by which folder exists.

| Name | Folders |
|---|---|
| `flat` (used here) | `pro/` `beam/` (+ `clean/`) |
| `research` | `ProjectorImage/` `BeamImage/` `OriginalImage/` |

```bash
python demo.py --input data/sample_input --gt data/sample_input   # flat
python demo.py --input /path/to/WarpData_0520                     # research
```

Images sitting loose in one folder are handled as `mixed`, split by the `projected_` and
`output_video_` prefixes.

---

## Label format

Standard YOLO. One line per box, `<cls_id> <cx> <cy> <w> <h>`, all normalised to 0–1.

```
0 0.139406 0.263969 0.122381 0.207129
1 0.505193 0.162010 0.127800 0.211946
5 0.311010 0.202954 0.125993 0.199101
```

`cls_id` runs `0..16` and follows the `names` order in
[configs/detection.yaml](../projector_distortion/configs/detection.yaml) — 11 fruits
(Apple … Watermelon), then 6 animals (Cat … Snake). All 17 classes appear in the bundled
labels.

---

## `collect.py` — build a dataset with the rig

The sample set above was made this way. Four stages, in order. `check` and `capture` drive
the projector and the webcam; `beam` and `warp` are plain file work and run anywhere.

```bash
python data/collect.py check                                  # monitors + webcam
python data/collect.py beam    --src data/live/BeamVideo.mp4   # video -> beam frames
python data/collect.py capture --screen 2 --rounds 3           # project and shoot
python data/collect.py warp                                    # rectify into pairs
```

Everything lands in one session folder, already in the layout the rest of the repo reads.
`--root` moves it; the default is one folder per day, so a second run of a stage extends
the first.

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
python demo.py  --input data/collected_0803 --gt data/collected_0803
python train.py --pro-dir data/collected_0803/pro --beam-dir data/collected_0803/beam \
                --clean-dir data/collected_0803/clean
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

### `beam` — video → beam frames

| Option | Default | Meaning |
|---|---|---|
| `--src <path>` | `data/live/BeamVideo.mp4` | Video file, or a folder of them |
| `--out <dir>` | `<root>/beam` | — |
| `--step N` | `30` | Keep every Nth frame. 30 = one per second at 30fps |
| `--size W H` | `1280 720` | `0 0` keeps the source resolution |
| `--quality N` | `95` | JPEG quality |
| `--limit N` | `0` | At most N frames per video |
| `--video-index N` | `1000` | The first video's index in the filename |

### `capture` — projector + webcam → raw captures

| Option | Default | Meaning |
|---|---|---|
| `--screen N` | `1` | Monitor the projector is attached to. `check` prints the table |
| `--beam <dir>` | `<root>/beam` | — |
| `--background <path>` | `data/live/BaseBackGround.jpg` | Projected while the clean shot is taken |
| `--rounds N` | `1` | Scene setups; each starts with a fresh clean shot |
| `--limit N` | `0` | Beam frames per round |
| `--shuffle` | off | Random beam order, so a short round still spans the clip |
| `--seed N` | `42` | For `--shuffle` |
| `--settle-ms N` | `150` | Wait after showing a frame before capturing it |
| `--flush N` | `3` | Buffered camera frames to drop before each capture |
| `--round-settle F` | `2.0` | Seconds between the clean shot and the first projection |
| `--preview-every N` | `10` | Refresh the capture preview every N frames |
| `--jpeg-quality N` | `95` | — |

**How a round works.** `capture` projects the background and waits for you to press `s`.
That shot becomes `clean` — the scene with nothing projected on it, so place the objects
and step out of frame first. Its timestamp becomes the `oriId`, and every capture of that
round carries it. That is how many `pro` files end up pointing at one `clean`. Then each
beam frame is projected once and captured once. `--rounds N` repeats with a new scene.

**Timing.** `--settle-ms` and `--flush` are what keep a capture matched to the frame that
caused it. If `pro` looks like the *previous* beam frame, raise both.

### `warp` — rectify into aligned pairs

| Option | Default | Meaning |
|---|---|---|
| `--warp boundary` | default | 4 corners + the measured edge bow |
| `--warp homography` | — | Corners only. For a flat screen with a clean boundary |
| `--warp tps` | — | The legacy thin-plate spline. Needs `opencv-python<5` |
| `--ori <dir>` · `--pro <dir>` | `<root>/raw/ori` · `<root>/raw/pro` | — |
| `--points N` | `20` | Boundary correspondences (tps and the debug overlay) |
| `--inset N` | `2` | Pixels to pull the boundary in, off the bright projection rim |
| `--work-size W H` | `1280 720` | Rectification resolution |
| `--final-size W H` | `640 360` | Saved resolution — the model input |
| `--limit N` | `0` | Process at most N scenes |
| `--no-debug` | off | Skip the before/after overlays in `<root>/debug/` |

**Why it exists.** In the camera the screen is a trapezoid and the objects sit at a
different scale in every capture, so nothing before this stage is trainable. `warp` finds
the screen boundary in the clean shot, then rectifies that scene's clean frame *and* all
its captures with the identical mapping. The pair ends up sharing a pixel grid, and the
only difference left is the projected light.

`boundary` is exact for a flat screen seen off-axis and still right when the edges bow.
OpenCV 5 removed the shape module, which is what `tps` needs.

Check `<session>/debug/<oriId>_warp.jpg` on a short trial session before committing to a
long one. It shows the detected boundary, the sampled points and the rectified result
together.

### Labels are not collected

`clean/` has to be annotated by hand into `<session>/labels/Ori<oriId>.txt` before
`evaluate.py` can score mAP. Restoration training and PSNR/SSIM need no labels.

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
python data/record.py --clip data/sample_video/mIni_Video_1.mp4 --loop
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
# 1) fill the same structure, then just run
python demo.py
python evaluate.py

# 2) or point at the original dataset directly
python demo.py --input /mnt/.../WarpData_0520 --gt /mnt/.../WarpData_0520
```

Training reads its three directories from `train.data` in
[configs/restoration.yaml](../projector_distortion/configs/restoration.yaml). Each takes a
directory, a glob, or a list. Precedence, the `--data-root` fallback and the clean-target
search order: [README_running.md](../README_running.md#where-the-training-data-comes-from).

---

[한국어](README_data.ko.md) · [← README](../README.md)
