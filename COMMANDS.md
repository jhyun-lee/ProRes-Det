# Command reference

Every command this project accepts, in one place. Options are grouped by the script
that owns them; defaults are what actually runs, including the ones that come from a
config file rather than the flag.

For *why* a setting lives where it does, see [README_running.md](README_running.md).
This page is the lookup table, not the explanation.

- [At a glance](#at-a-glance)
- [Install](#install)
- [`demo.py` — restore → detect](#demopy--restore--detect)
- [`evaluate.py` — score before vs after](#evaluatepy--score-before-vs-after)
- [`train.py` — retrain the restorer](#trainpy--retrain-the-restorer)
- [`Data.py` — collect a dataset](#datapy--collect-a-dataset)
- [Keys at the interactive prompts](#keys-at-the-interactive-prompts)
- [Tests](#tests)
- [Where output lands](#where-output-lands)
- [Settings that have no flag](#settings-that-have-no-flag)

---

## At a glance

Run everything from the repo root.

| Command | Needs a rig | Does |
|---|:---:|---|
| `python src/demo.py` | — | Restore + detect the 200 bundled eval pairs |
| `python src/demo.py --video-pair <mp4>` | — | Replay a recording into a restored video |
| `python src/demo.py --live` | **yes** | Drive the webcam + projector live |
| `python src/evaluate.py` | — | Score before vs after: mAP, PSNR, SSIM |
| `python src/train.py` | — | Retrain the restorer on `sample_train` |
| `python src/Data.py` | — | List the collection stages |
| `python src/Data.py check` | **yes** | List monitors, preview the webcam |
| `python src/Data.py make_light` | — | Video → the light frames to project |
| `python src/Data.py capture` | **yes** | Project and shoot → raw captures |
| `python src/Data.py capture_warp` | **yes** | The same, rectifying as it shoots |
| `python src/Data.py warp` | — | Raw captures → rectified, aligned pairs |
| `python src/Data.py record` | **yes** | Project a clip and record it, no models |
| `python -m pytest -q` | — | 151 tests |

The first three of `demo` / `evaluate` / `train` need no arguments: the sample data and
all three checkpoints are tracked in git, so a fresh clone runs as-is.

An editable install also puts three console scripts on the PATH. They are thin wrappers
around the same files and take the identical flags:

| Console script | Same as |
|---|---|
| `pdf-demo` | `python src/demo.py` |
| `pdf-evaluate` | `python src/evaluate.py` |
| `pdf-train` | `python src/train.py` |

---

## Install

Python >= 3.9. Extras: `[yolo]` `[train]` `[live]` `[test]` `[all]`.

```bash
pip install -e ".[all]" -r requirements-cuda.txt    # GPU
pip install -e ".[all]"                             # CPU only
```

Why the CUDA file is separate, and how to confirm the GPU took:
[README.md](README.md#1-install).

---

## `demo.py` — restore → detect

Three modes in one script. Which one runs is decided by the flags, not a subcommand:
nothing = offline, `--video-pair` = replay, `--live` = the rig. `--live` and
`--video-pair` together are refused.

### Shared

| Option | Default | Meaning |
|---|---|---|
| `--input <dir>` | `datasets/SampleData/sample_eval` | Folder of `distorted/` + `light/` pairs (offline) |
| `--output <dir>` | `output` | Where run folders go |
| `--limit N` | `0` | Process at most N frames. `0` = all |
| `--no-detect` | off | Restoration only, without editing `detection.yaml` |
| `--save-every N` | `1` | Write images every N frames. `0` = csv only |
| `--device cuda\|cpu` | cuda if present | — |
| `--clip <mp4>` | see below | The projected clip. Live plays it; replay pairs against it |

`--clip` defaults differently per mode: live plays `datasets/live/test_light.mp4`, replay
uses whatever the recording's `.json` says was actually projected.

### Offline (default)

| Option | Default | Meaning |
|---|---|---|
| `--video` | off | Also write the 2×2 panels as `result.mp4` |

```bash
python src/demo.py
python src/demo.py --input datasets/SampleData/sample_test
python src/demo.py --limit 5 --save-every 0
python src/demo.py --input /path/to/pairs --video
```

### Replay — `--video-pair`

Two mp4s in, one or more out. No hardware.

| Option | Default | Meaning |
|---|---|---|
| `--video-pair <mp4>` | — (required) | A `Data.py record` mp4 to replay |
| `--video-out <kind>...` | `panel` | `panel` (2×2) · `restored` (the restored frame alone). Both allowed |
| `--no-review` | off | Skip the calibration verdict window, for an unattended run |

```bash
python src/demo.py --video-pair datasets/recordings/rec_0828145219.mp4
python src/demo.py --video-pair rec.mp4 --no-detect --video-out restored
python src/demo.py --video-pair rec.mp4 --video-out restored panel
python src/demo.py --video-pair rec.mp4 --clip datasets/live/train_light_1.mp4
python src/demo.py --video-pair rec.mp4 --no-review --save-every 0
```

One `--video-out` writes `result.mp4`; several write one `result_<kind>.mp4` each, off
a single restoration pass.

### Live — `--live`

Needs a webcam pointed at a screen a projector is throwing to.

| Option | Default | Meaning |
|---|---|---|
| `--live` | off | Drive the rig |
| `--screen N` | `rig.screen` (`1`) | Monitor the projector is on. `0` = primary |
| `--camera N` | `rig.camera` (`0`) | Webcam index |
| `--debug-view` | off | Extra window: the pre-warp feed with the quad on it |

```bash
python src/demo.py --live
python src/demo.py --live --screen 2
python src/demo.py --live --save-every 30 --debug-view
```

---

## `evaluate.py` — score before vs after

| Option | Default | Meaning |
|---|---|---|
| `--input <dir>` | `datasets/SampleData/sample_eval` | Pairs to score |
| `--gt <dir>` | `datasets/SampleData/sample_eval` | Where `surface/` and `labels/` live |
| `--output <dir>` | `output` | Where the report goes |
| `--limit N` | `0` | Score at most N pairs. `0` = all |
| `--iou F` | `0.5` | IoU threshold for a true positive |

```bash
python src/evaluate.py
python src/evaluate.py --input datasets/SampleData/sample_test --gt datasets/SampleData/sample_test
python src/evaluate.py --iou 0.75 --limit 10
```

`--input` and `--gt` take the same path for every bundled split, because each split
carries its own `surface/` and `labels/` beside `distorted/`.

Scoring several detectors in one report is a config edit, not a flag — put a list in
`detector.backend` and each entry becomes a row.

---

## `train.py` — retrain the restorer

| Option | Default | Meaning |
|---|---|---|
| `--data-root <dir>` | `train.data` in `restoration.yaml` | Training pairs |
| `--gt <dir>` | under `--data-root` | Where `surface/` lives when it is not under `--data-root` |
| `--val-root <dir>` | `train.validation.root` | Full validation split, restored after every epoch |
| `--val-gt <dir>` | under `--val-root` | Separate validation directory containing `surface/` |
| `--val-save-images N` | `10` | Previews per epoch; `-1` all, `0` none. Metrics always cover all validation pairs |
| `--out <dir>` | `runs` | Where checkpoints go |
| `--epochs N` | `30` | From `train.epochs` |
| `--batch-size N` | `4` | From `train.batch_size` |
| `--lr F` | `0.0002` | From `train.lr` |
| `--sample N` | `0` | Cap the triplet count — a seeded random draw. `0` = all |
| `--resume <ckpt>` | — | Continue from a checkpoint |
| `--device cuda\|cpu` | cuda if present | — |

```bash
python src/train.py
python src/train.py --epochs 60 --batch-size 8
python src/train.py --sample 200 --epochs 2          # a quick smoke run
python src/train.py --data-root datasets/Create_Data/warp_0813
python src/train.py --resume runs/<tag>/best.pt
python src/train.py --data-root datasets/SampleData/sample_train --val-root datasets/SampleData/sample_eval
```

Only training pairs update weights. The best checkpoint maximizes mean validation
PSNR at `model.input_size`; `sample_test` is reserved for final evaluation. Validation
loss, PSNR, SSIM, and per-image metrics are logged every epoch. Training and validation
surface IDs must be disjoint.

The architecture is **not** a flag. Ablations are the `ablation:` block in
`restoration.yaml`; the run directory and checkpoint name pick up a tag naming what was
turned off, so a variant stays identifiable from its path.

---

## `Data.py` — collect a dataset

One entry point that dispatches to the standalone scripts under `datasets/`. Each stage
keeps its own `--help`, and running the script directly is equivalent. What each stage
*does*, how a round works and how often the scene has to change:
[datasets/README_data.md](datasets/README_data.md).

```bash
python src/Data.py                  # list the stages
python src/Data.py capture --help   # that stage's options
python datasets/warp.py --review    # identical to `python src/Data.py warp --review`
```

| Stage | Script | Needs a rig |
|---|---|:---:|
| `check` | `datasets/check.py` | **yes** |
| `make_light` | `datasets/make_light.py` | — |
| `capture` | `datasets/capture.py` | **yes** |
| `capture_warp` | `datasets/capture.py --warp` | **yes** |
| `warp` | `datasets/warp.py` | — |
| `record` | `datasets/record.py` | **yes** |

### `check` — monitors + webcam

| Option | Default |
|---|---|
| `--camera N` | `capture.camera` (`0`) |

```bash
python src/Data.py check
python src/Data.py check --camera 1
```

### `make_light` — video → light frames

| Option | Default | Meaning |
|---|---|---|
| `--src <path>...` | the three `train_light_*.mp4` | Video files, or folders of them |
| `--out <dir>` | `datasets/Create_Data/projected` | — |
| `--limit N` | `0` | At most N *source* frames per video, before augmentation |

```bash
python src/Data.py make_light
python src/Data.py make_light --src /path/clip.mp4 --limit 100
```

### `capture` — projector + webcam → raw captures

| Option | Default | Meaning |
|---|---|---|
| `--raw <dir>` | `datasets/Create_Data/raw_<MMDD>` | Add to a specific capture folder |
| `--screen N` | `capture.screen` (`1`) | Monitor the projector is on |
| `--camera N` | `capture.camera` (`0`) | — |
| `--rounds N` | `10` | Scene setups; each starts with a fresh surface shot |
| `--limit N` | `50` | Light frames drawn per round. `0` = every frame |
| `--warp` | off | Also rectify as you shoot (this is `capture_warp`) |

```bash
python src/Data.py capture --screen 2
python src/Data.py capture_warp --screen 2
python src/Data.py capture --rounds 20 --limit 25      # 500 pairs over 20 scenes
python src/Data.py capture --rounds 4 --limit 250      # 1,000 pairs over 4 scenes
```

### `warp` — raw captures → aligned pairs

| Option | Default | Meaning |
|---|---|---|
| `--raw <dir>` | the newest `raw_<MMDD>` | Which capture folder to rectify |
| `--out <dir>` | `warp_<MMDD>` matching the raw | — |
| `--limit N` | `0` | Process at most N scenes |
| `--review` | off | Check every scene on screen and fix the bad ones |

```bash
python src/Data.py warp
python src/Data.py warp --review
python src/Data.py warp --raw datasets/Create_Data/raw_0813
```

### `record` — project and record, no models

| Option | Default | Meaning |
|---|---|---|
| `--clip <mp4>` | `datasets/live/test_light.mp4` | Clip to project |
| `--screen N` | `record.screen` (`1`) | Monitor the projector is on |
| `--camera N` | `record.camera` (`0`) | — |
| `--seconds F` | `0` | Stop after N seconds. `0` = no limit |
| `--loop` | off | Restart the clip instead of stopping at its end |
| `--warp` | off | Calibrate once and record the rectified screen |
| `--out <path>` | `datasets/recordings/rec_<MMDDHHMMSS>.mp4` | — |

```bash
python src/Data.py record --screen 2 --seconds 30
python src/Data.py record --clip datasets/live/train_light_1.mp4 --loop
python src/Data.py record --warp
```

---

## Keys at the interactive prompts

Every window that blocks a run, and what it takes. `enter` and `space` are
interchangeable with `a` wherever both are listed.

| Where | Key | Does |
|---|---|---|
| `capture` framing preview | `s` · `enter` · `space` | Keep this shot as the round's `surface` |
| | `c` | Freeze the frame and click the 4 corners |
| | `r` | Back to auto-detection after clicking |
| | `q` · `esc` | Stop the session |
| `warp --review` | `enter` · `space` · `a` | Accept this scene |
| | `A` | Accept this and every scene after it |
| | `m` | Click the 4 corners by hand |
| | `r` | Re-run the detector |
| | `s` | Skip this scene |
| | `q` | Stop; scenes already written are kept |
| `demo.py --live` calibration | `enter` · `space` · `a` | Accept the warp and start |
| | `m` | Click the 4 corners |
| | `r` | Re-run calibration |
| | `q` | Abort the run |
| `demo.py --video-pair` calibration | `enter` · `a` | Accept the warp and start |
| | `m` | Click the 4 corners |
| | `q` | Abort the run |
| Corner clicking (any mode) | left click ×4 | Place the corners, any order |
| | `u` | Undo the last corner |
| | `q` | Abort |
| `--live` / `record` running | `q` | Stop early |

---

## Tests

| Command | Runs |
|---|---|
| `python -m pytest -q` | All 151 |
| `python -m pytest tests/test_replay.py -q` | The replay pipeline alone |
| `python -m pytest -q -k "align or clip"` | Anything matching the expression |
| `python -m pytest -q --collect-only` | List without running |

No hardware needed for any of them. Tests that need a checkpoint or an optional module
skip cleanly when it is absent rather than failing.

| File | Covers |
|---|---|
| `tests/test_pipeline.py` | Filename ids, sample discovery, metrics, `RunRecorder`, argparse defaults, the live worker contract |
| `tests/test_replay.py` | Timestamp pairing, the sidecar, screen detection, multi-video output, window sizing |
| `tests/test_restoration.py` | Config tags, every ablation toggle, checkpoint round trip, the shipped weights |
| `tests/test_detection.py` | Registry, label normalisation, the size gate, both backends |
| `tests/test_collect.py` | `datasets/warp.py` and `datasets/make_light.py` without a rig |

---

## Where output lands

| Path | Written by | Tracked |
|---|---|---|
| `output/<timestamp>/` | `demo.py` | no |
| `output/Eval_<dataset>/` | `evaluate.py` | no |
| `runs/<tag>/` | `train.py` | no |
| `datasets/Create_Data/projected/` | `Data.py make_light` | no |
| `datasets/Create_Data/raw_<MMDD>/` | `Data.py capture` | no |
| `datasets/Create_Data/warp_<MMDD>/` | `Data.py warp` | no |
| `datasets/recordings/` | `Data.py record` | no |

Inside a `demo.py` run folder:

| File | When |
|---|---|
| `run_meta.json` | always |
| `detections.csv` | always |
| `captures/` · `frames_all/` | `--save-every` > 0 |
| `calib/` | live and replay runs |
| `result.mp4` | `--video`, live, or replay with one `--video-out` |
| `result_<kind>.mp4` | replay with several `--video-out` |

Each run gets its own timestamped folder, so repeated runs never overwrite each other.

---

## Settings that have no flag

**No flag names a model.** These are read from
`projector_distortion/configs/*.yaml`, because none of them is what changes between two
runs:

| Setting | Where |
|---|---|
| Restoration backend · checkpoint · input size | `model:` in `restoration.yaml` |
| Architecture and ablations | `ablation:` in `restoration.yaml` |
| Epochs, batch, lr, accumulation, loss weights | `train:` in `restoration.yaml` |
| Detection backend · checkpoints · confidence | `detector:` / `weights:` in `detection.yaml` |
| Class names, box size gate, inference size | `detection.yaml` |
| Live rig: monitor, webcam, latency, calibration review | `rig:` in `live.yaml` |
| Collection rig, session folders, warp geometry, codec | `collect.yaml` |

The two that do have an override — `--screen` and `--camera` — are the ones someone
gets wrong on the first run of a session, where an aborted run should not cost a file
edit.

```yaml
# projector_distortion/configs/detection.yaml
detector:
  backend: ssd      # no ultralytics needed
  # backend: none   # restoration only, for every run
```

```yaml
# projector_distortion/configs/live.yaml
rig:
  screen: 1
  camera: 0
  offset: 6                  # projector -> camera latency, in frames
  review_calibration: true   # false for an unattended run
```

---

[← README](README.md)
