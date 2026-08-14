# data/

Everything about the dataset: what ships, how filenames tie the views together, the label
format, and the tools that build more of it.

Script options for `demo.py` / `evaluate.py` / `train.py` are in
[README_running.md](../README_running.md).

- [What ships](#what-ships)
- [The three splits](#the-three-splits)
- [Filename convention](#filename-convention)
- [Layouts](#layouts)
- [Label format](#label-format)
- [`Data.py` — build a dataset with the rig](#datapy--build-a-dataset-with-the-rig)
- [`Data.py record` — project and record, no models](#datapy-record--project-and-record-no-models)
- [Swapping in real data](#swapping-in-real-data)

---

## What ships

A dataset that runs out of the box, split three ways under `SampleData/`. It is tracked in
git, so it is there right after a clone.

```
data/
├── check.py                list the displays, preview the webcam
├── make_light.py           video -> the light frames a projector will throw
├── capture.py              projector + webcam -> raw captures
├── warp.py                 raw captures -> rectified, aligned pairs
├── record.py               project a clip and record the camera's view (no models)
├── common.py               collect.yaml access, session folders, shared helpers
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
├── live/                   the projection sources, split by role       121 MiB
│   ├── train_light_1.mp4   what `Data.py make_light` turns into training light frames
│   │                       4,262 frames @30fps = 2.4 min, 854×480, 15 MiB
│   ├── train_light_2.mp4   5,381 frames @30fps = 3.0 min, 856×480, 7 MiB
│   ├── train_light_3.mp4  24,272 frames @30fps = 13.5 min, 854×480, 44 MiB
│   ├── test_light.mp4      held out — demo.py --live and `Data.py record` default to it
│   │                       5,858 frames @30fps = 3.3 min, 854×480, 55 MiB
│   └── BaseBackGround.jpg  background shown during calibration, 1280×960
├── Create_Data/            where a collection session lands (git-ignored)
└── recordings/             where `Data.py record` writes (git-ignored)
```

`Data.py` at the repo root is the entry point for the five stage scripts above
(`common.py` is shared plumbing, not a stage). They are standalone programs, not a
package, so each is runnable on its own too.

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

### The projection sources are split too

`live/` holds the clips a projector plays, and the split runs through them as well:

| Clip | Feeds | Used by |
|---|---|---|
| `train_light_1.mp4` · `_2` · `_3` | the training split | `Data.py make_light` defaults to all three |
| `test_light.mp4` | the held-out split | `demo.py --live` and `Data.py record` default to it |

Keeping them apart is what makes the held-out split honest: a set collected with
`Data.py` shares no projected frame with `test_light.mp4`, so a checkpoint fitted on it
has never seen the light it is scored against. `--src` and `--clip` override either side.

`data/` is 349 MiB tracked, and 121 MiB of that is `live/`. The clips are only needed to
*collect* or to run `--live` — the frames they produce already sit in `SampleData/*/light/`,
so training and evaluation run without them. Delete them if you need neither.

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

## `Data.py` — build a dataset with the rig

The sample set above was made this way. `Data.py` at the repo root dispatches to the
scripts under `data/`, and every stage keeps its own `--help`:

```bash
python Data.py                     # list the stages
python Data.py capture --help      # that stage's options
```

| Stage | Script | Does | Needs the rig |
|---|---|---|---|
| `check` | `check.py` | List the displays, preview the webcam | yes |
| `make_light` | `make_light.py` | Video → light frames | no |
| `capture` | `capture.py` | Project and shoot → raw captures | yes |
| `capture_warp` | `capture.py --warp` | The same, rectifying as it shoots | yes |
| `warp` | `warp.py` | Raw captures → aligned pairs | no |
| `record` | `record.py` | Project a clip and record it, no models | yes |

`make_light` and `warp` are plain file work, so a session can be shot on the rig machine
and rectified anywhere else. Running a script directly is equivalent —
`python data/warp.py --review` and `python Data.py warp --review` are the same program.

```bash
python Data.py check                       # monitors + webcam
python Data.py make_light                  # video -> light frames
python Data.py capture_warp --screen 2     # shoot 10 scenes, rectify as it goes
```

Or keep the two halves apart, which is what leaves the geometry redoable later:

```bash
python Data.py capture --screen 2
python Data.py warp
```

`make_light` reads the three `train_light_*.mp4` clips by default. `test_light.mp4` is
left out on purpose, so a set collected here shares no projected frame with the held-out
split.

Extracting the light frames is a separate stage on purpose — `capture` never extracts
anything itself, it only samples what is already in `projected/`. There is no cache and no
"already done" check: every run stamps a fresh timestamp into the filenames, so running
`make_light` twice over one clip writes a second copy of every frame rather than skipping
it.

### Where it lands

Everything goes under `session.dir` in
[configs/collect.yaml](../projector_distortion/configs/collect.yaml) — `data/Create_Data`
by default, and git-ignored:

```
data/Create_Data/
├── projected/       light_<lightId>.jpg                 [make_light] what the projector emits
├── raw_<MMDD>/      surface/surface_<surfaceId>.jpg     [capture]    camera frames, unrectified
│                    distorted/distorted_<surfaceId>_<lightId>.jpg
│                    collect_meta.json                                settings, counts, corners
└── warp_<MMDD>/     surface/surface_<surfaceId>.jpg     [warp]       rectified, 640×360
                     distorted/distorted_<surfaceId>_<lightId>.jpg
                     debug/<surfaceId>_warp.jpg                       before/after evidence
                     collect_meta.json
```

`projected/` is deliberately not dated: the frames a clip yields are the same on any day,
so a per-day copy would cost disk and buy nothing. The captures are dated, because a rig
gets moved and a scene gets rebuilt between days.

A warp folder is named after the raw it came from, not after the day `warp` ran, so
`raw_0813` and `warp_0813` describe the same captures however long the gap. `capture`
fills today's raw; `warp` takes the newest raw unless `--raw` names another.

`warp_<MMDD>/` is a `flat` layout carrying the current filenames, so nothing needs
converting:

```bash
python demo.py  --input data/Create_Data/warp_0813
python train.py --data-root data/Create_Data/warp_0813
```

### Settings that are not flags

Only what genuinely changes between runs stayed on the command line. Resolutions, the
camera backend, the hue bands, the warp geometry and the codec are properties of the rig
or of the naming convention, so they are set once in
[configs/collect.yaml](../projector_distortion/configs/collect.yaml):

| Block | Owns |
|---|---|
| `session:` | `dir`, the `projected` / `raw_` / `warp_` names, `jpeg_quality` (95) |
| `light:` | `src`, `step`, `augment`, `size` (1280×720), `hue_bands`, `first_video_index`, `seed` |
| `capture:` | `screen`, `camera`, `cam_backend`, `background`, `rounds`, `limit`, `settle_ms`, `flush`, `round_settle`, `preview_every`, `seed` |
| `warp:` | `mode`, `work_size` (1280×720), `final_size` (640×360), `points` (20), `inset` (2), `debug` |
| `record:` | `clip`, `out_dir`, `screen`, `camera`, `cam_backend`, `background`, `codec`, `fps_probe`, `preview_every`, `max_queued` |

It is a second config file rather than a section of `live.yaml` because these scripts run
without torch and never import the pipeline package.

The requested camera resolution and rate are the exception, fixed at 1280×960 @30fps —
`CAM_WIDTH` / `CAM_HEIGHT` / `CAM_FPS` in `projector_distortion/pipeline/live.py`, shared
with `demo.py --live`. Drivers frequently ignore the request, so what the camera actually
opened at is printed at startup and stored in `collect_meta.json`.

### How often the scene has to change

`capture` takes one `surface` shot per **round**, so the objects on the screen change once
per round — `--rounds` is how many scenes you will be asked to build.

Each round draws `--limit` light frames at random from the whole `projected/` folder, so
ten rounds are lit by ten different draws rather than by one list ten times. The defaults
are `--rounds 10 --limit 50` = **500 pairs over 10 scenes**.

Nothing is on a clock. The surface shot blocks until you press `s`, so take as long as the
scene needs; the numbers below are only the automatic part that follows.

| Command | Scenes | Shots per scene | Pairs | Shooting per round |
|---|---|---|---|---|
| `capture` (defaults) | 10 | 50 | **500** | ~1 min |
| `capture --rounds 20 --limit 25` | 20 | 25 | 500 | ~30 s |
| `capture --rounds 4 --limit 250` | 4 | 250 | 1,000 | ~5 min |

"Shooting per round" is the capture loop alone, at the default `capture.settle_ms` of
1200 ms per frame; the run prints its own estimate before it starts. Building the scene is
yours and is not counted.

`--limit 0` means *every* frame in the pool, which after a default `make_light` is
thousands — one scene, hours of shooting, and the least scene variety a session can have.
That is the wrong trade for restoration: the model needs many surfaces, not many lights on
one surface.

Sampling is unseeded, so running two sessions on different days accumulates variety
instead of repeating a draw. Set `capture.seed` in `collect.yaml` to make one reproducible.

### `check`

Lists the displays and opens the webcam, so `capture` is not the first attempt.

| Option | Default |
|---|---|
| `--camera N` | `capture.camera` |

```
  --screen 0 -> 2560x1440 at (0,0) (primary)  \\.\DISPLAY1
  --screen 1 -> 1920x1080 at (2560,0)         \\.\DISPLAY2
```

### `make_light` — video → light frames

| Option | Default | Meaning |
|---|---|---|
| `--src <path>...` | `light.src` — the three `train_light_*.mp4` | Any mix of video files and folders of them |
| `--out <dir>` | `<session.dir>/projected` | — |
| `--step N` | `light.step` (`30`) | Keep every Nth frame. 30 = one per second at 30fps |
| `--augment <mode>` | `light.augment` (`full`) | Copies per source frame: `full` = original + inverted + 4 hue rotations (6×), `invert` = 2×, `none` = 1× |
| `--limit N` | `0` | At most N *source* frames per video, before augmentation |

Written as `light_<tag>_<videoIdx>_<frameIdx>_<variant>.jpg`, where `variant` is `0`,
`invert` or the hue angle. The folder is `projected/` but the files stay `light_*`:
`projected_` is already the pre-rename prefix for *distorted* files, so a light frame
named that way would parse as a distorted one.

Frames are saved at `light.size` (1280×720) and `session.jpeg_quality` (95), with the
first video numbered `light.first_video_index` (`1000`). `warp` downscales again to the
model's input size, so this only has to sit comfortably above it. Set `light.seed` to fix
the hue angles for a reproducible run.

### `capture` — projector + webcam → raw captures

| Option | Default | Meaning |
|---|---|---|
| `--raw <dir>` | `<session.dir>/raw_<MMDD>`, today's | Add to a specific capture folder instead |
| `--screen N` | `capture.screen` (`1`) | Monitor the projector is attached to. `check` prints the table |
| `--camera N` | `capture.camera` (`0`) | — |
| `--rounds N` | `capture.rounds` (`10`) | Scene setups; each starts with a fresh surface shot |
| `--limit N` | `capture.limit` (`50`) | Light frames drawn per round, at random from the whole pool. `0` uses every frame |
| `--settle-ms N` | `capture.settle_ms` (`1200`) | How long a light frame is held before its capture is kept |
| `--warp` | off | Also rectify as you shoot. `Data.py capture_warp` is this flag |

**How a round works.** `capture` projects the background and waits for you to press `s`.
That shot becomes `surface` — the scene with nothing projected on it, so place the objects
and step out of frame first. Its timestamp becomes the `surfaceId`, and every capture of
that round carries it. That is how many `distorted` files end up pointing at one
`surface`. Then each light frame is projected once and captured once. `--rounds N` repeats
with a new scene.

**The framing preview.** While you are arranging the scene, the screen boundary is
detected live and the rectified view is shown beside the camera one, so a warp that will
not resolve gets fixed at the rig instead of at the warp stage with the scene long since
dismantled.

| Key | Does |
|---|---|
| `s` · `enter` · `space` | Take the surface shot and start the round |
| `c` | Freeze the frame and click the 4 corners yourself, when detection cannot find them |
| `r` | Drop the clicked corners, go back to auto-detection |
| `q` · `esc` | Abort |

Whatever corners are in force when `s` is pressed are written into `collect_meta.json`,
and `warp` reuses them instead of detecting again — they were measured with someone
watching the rectified preview, which is the one moment anybody can tell a good boundary
from a bad one. `warp --redetect` ignores them.

**Timing.** `capture.settle_ms` (1200 ms) and `capture.flush` (3) are what keep a capture
matched to the frame that caused it. The camera is read for the whole settle window rather
than slept through: a webcam hands back whatever it has queued, and auto-exposure only
adjusts on frames the driver actually delivers. Consecutive light frames are inverted or
hue-rotated copies of each other, so almost every step is a large brightness swing and an
AE loop needs the better part of a second. If `distorted` looks like the *previous* light
frame, raise `settle_ms`.

**`capture_warp`.** With `--warp`, one rectifier is built per round from the corners on
screen at the surface shot, and the rectified `surface/`, `distorted/` and `debug/` are
written into `warp_<MMDD>/` alongside the raw. The raw is kept either way, so a round that
came out wrong can still be redone with `Data.py warp --review` without going back to the
rig. A round whose corners are unusable is captured raw only and reported as such.

### `warp` — rectify into aligned pairs

| Option | Default | Meaning |
|---|---|---|
| `--raw <dir>` | the newest `raw_<MMDD>` | Captures to rectify |
| `--out <dir>` | `<session.dir>/warp_<MMDD>`, named after the raw | — |
| `--mode boundary` | `warp.mode` (default) | 4 corners + the measured edge bow |
| `--mode homography` | — | Corners only. For a flat screen with a clean boundary |
| `--mode tps` | — | The legacy thin-plate spline. Needs `opencv-python<5` |
| `--limit N` | `0` | Process at most N scenes |
| `--review` | off | Show every scene and wait for a verdict |
| `--manual` | off | Skip auto-detection, click the corners on every scene |
| `--redetect` | off | Ignore the corners `capture` recorded and detect again |

Rectification runs at `warp.work_size` (1280×720) with `warp.points` (20) boundary
correspondences and the boundary pulled `warp.inset` (2) px in off the bright projection
rim, then saves at `warp.final_size` (640×360).

**Why it exists.** In the camera the screen is a trapezoid and the objects sit at a
different scale in every capture, so nothing before this stage is trainable — and a
session that only has `raw_<MMDD>/` is not a layout the loaders recognise at all. `warp`
finds the screen boundary in the surface shot, then rectifies that scene's surface frame
*and* all its captures with the identical mapping. The pair ends up sharing a pixel grid,
and the only difference left is the projected light.

`boundary` is exact for a flat screen seen off-axis and still right when the edges bow.
OpenCV 5 removed the shape module, which is what `tps` needs.

**`--review`.** Boundary detection is a contour heuristic on a photograph: right most of
the time, and wrong in ways the summary line cannot show. `--review` puts each scene on
screen — the detected quad next to the rectified result — and waits:

| Key | Does |
|---|---|
| `enter` · `space` · `a` | Accept this scene |
| `m` | Click the 4 corners by hand instead |
| `r` | Re-run the detector |
| `s` | Skip this scene |
| `A` | Accept this and every scene after it, no more prompts |
| `q` | Stop; the scenes already written are kept |

Without `--review`, a scene whose boundary cannot be found is reported and skipped.
Either way, check `warp_<MMDD>/debug/<surfaceId>_warp.jpg` on a short trial session before
committing to a long one — it shows the detected boundary, the sampled points and the
rectified result together.

### Labels are not collected

`surface/` has to be annotated by hand into `<session>/labels/surface_<surfaceId>.txt` — or
`.json`, if you annotate in LabelMe — before `evaluate.py` can score mAP. Restoration
training and PSNR/SSIM need no labels, which is why `sample_train` ships without any.

---

## `Data.py record` — project and record, no models

The other stages build a *dataset*: still pairs, rectified and id-matched. `record` builds
a *video*. It projects a clip at its own fps and records the camera's view as one mp4.
No restoration, no detection, no weights loaded — so it runs on a machine that has none.

It stands outside the session folders the other stages share: one mp4 is not the
distorted/light/surface triplets a training set is made of, so there is nothing for `warp`
or `train.py` to pick up. Use it to capture raw distorted footage before there is a
checkpoint, or to prove a rig works end to end.

| | Default | Change it with |
|---|---|---|
| Projected clip | `record.clip` — `data/live/test_light.mp4` | `--clip <path>` |
| Background | `record.background` — `data/live/BaseBackGround.jpg` | `collect.yaml` |
| Output | `record.out_dir/rec_<MMDDHHMMSS>.mp4` + a `.json` beside it | `--out <path>` |

| Option | Default | Meaning |
|---|---|---|
| `--screen N` | `record.screen` (`1`) | Monitor the projector is attached to. `0` = primary |
| `--camera N` | `record.camera` (`0`) | — |
| `--loop` | off | Restart the clip instead of stopping at its end |
| `--seconds F` | `0` | Stop after N seconds. `0` = no limit |
| `--warp` | off | Calibrate once and record the rectified screen |

The mp4 is `record.codec` (`mp4v`) at the camera's own resolution, the rate is measured
over `record.fps_probe` (24) frames, the preview refreshes every `record.preview_every`
(5) projected frames, and `record.max_queued` (32) camera frames may wait on the encoder.

```bash
python Data.py record --screen 2 --seconds 30
python Data.py record --clip data/live/train_light_1.mp4 --loop
python Data.py record --warp
```

Recording starts as soon as the window is up — no keypress. `q` stops early, as do the end
of the clip and `--seconds`.

The header fps is **measured**, not requested — there is no flag to set it, on purpose.
Webcams routinely deliver something other than what they were asked for and report a
third number, and a wrong header is exactly what makes a recording play back in fast or
slow motion. The probe frames are buffered until the rate is known, so measuring costs no
footage.

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
