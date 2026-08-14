# Running ProRes-Det

Options, input and output for the three root entry points. For install and the
one-liners that just work, see [README.md](README.md).

Dataset collection is `Data.py` at the repo root, which dispatches to the scripts under
`data/`; it is documented in [data/README_data.md](data/README_data.md).

- [Shared flags](#shared-flags)
- [`demo.py` — restore → detect](#demopy--restore--detect)
- [`demo.py --live` — webcam + projector](#demopy---live--webcam--projector)
- [`evaluate.py` — score before vs after](#evaluatepy--score-before-vs-after)
- [`train.py` — retrain the restorer](#trainpy--retrain-the-restorer)
- [Output format](#output-format)
- [Making restoration faster](#making-restoration-faster)

---

## What the config files own

**No flag names a model.** Which restorer runs, which detector runs, which checkpoints
they load and at what confidence — all of it is read from
`projector_distortion/configs/*.yaml`. Those settings each had a flag once; none of
them was what actually changed between two runs, and together they made `--help`
unreadable.

| Setting | Where |
|---|---|
| Live rig: monitor, webcam, projector→camera latency | `rig:` in `live.yaml` |
| Collection rig, session folders, warp geometry | `collect.yaml` — read by `Data.py`, not by these three |
| Restoration backend | `model.backend` in `restoration.yaml` |
| Restoration checkpoint | `model.weights` in `restoration.yaml` |
| What the restorer runs at | `model.input_size` in `restoration.yaml` |
| Architecture / ablation / capacity | `ablation:` in `restoration.yaml` — see [Ablation](#ablation) |
| Training epochs, batch, lr, accumulation, loss weights | `train:` in `restoration.yaml` |
| Detection backend | `detector.backend` in `detection.yaml` — `yolo` · `ssd` · `none`, or any name registered with `@register_detector`. A list runs one per row in `evaluate.py` |
| Detector checkpoints | `weights.<backend>` in `detection.yaml` |
| Detector confidence floor | `detector.conf` in `detection.yaml` |
| Class names | `names:` in `detection.yaml` — a YOLO checkpoint's own names win over the list |
| Box size gate | `detector.min_width` / `min_height` / `min_area` in `detection.yaml` |
| Detector inference size | `detector.imgsz` in `detection.yaml` |

The inference scripts never need the architecture settings anyway: a checkpoint carries
the architecture it was trained with.

The flags that remain are about the run in front of you — which data, where the output
goes, how much of it to process, and on the live rig which hardware. Full list:
`python <script>.py --help`. `--device cuda|cpu` is on `demo.py` and `train.py`;
`evaluate.py` always resolves it (cuda if present).

---

## `demo.py` — restore → detect

Restoration and detection only. It never reads ground truth and never scores anything —
that is `evaluate.py`'s job.

| | Default | Change it with |
|---|---|---|
| Input | `data/SampleData/sample_eval/` (`distorted/` + `light/`) | `--input <dir>` |
| Output | `output/<timestamp>/` | `--output <dir>` |

| Option | Default | Meaning |
|---|---|---|
| `--limit N` | `0` | Cap how many pairs are processed. `0` = all |
| `--save-every N` | `1` | Image save interval. `0` keeps the csv only |
| `--video` | off | Also write the 2×2 panels as `result.mp4` |
| `--device cuda\|cpu` | cuda if present | — |

Images are written at JPEG quality 92. Box size gating comes from
`detector.min_width` / `min_height` / `min_area` in `configs/detection.yaml`; every box
that clears it is kept, because metric code must not lose duplicates.

```bash
python demo.py --input /path/to/pairs
python demo.py --limit 5 --save-every 0
```

To restore without detecting, set `detector.backend: none` in
`configs/detection.yaml`. Comparing detectors is `evaluate.py`'s job — that is where a
list of backends produces one row each.

`--save-every 0` writes no images at all, which is what you want when only the csv and
the summary matter. Each run lands in its own `output/<timestamp>/`, so repeated runs
never overwrite each other.

---

## `demo.py --live` — webcam + projector

Needs real hardware: a webcam pointed at a screen a projector is throwing to.

### The rig is configured, not flagged

Which monitor, which webcam and how far the camera lags are properties of the physical
setup, so they live in
[configs/live.yaml](projector_distortion/configs/live.yaml) — set once per machine:

```yaml
rig:
  screen: 1               # monitor the projector is on; 0 = primary
  camera: 0               # webcam index
  cam_backend: auto       # auto | any | dshow | msmf | v4l2
  offset: 6               # projector → camera latency, in frames
```

`offset` is the one worth measuring: the camera frame showing light frame N arrives
`offset` frames later, so it is what pairs a capture with the frame that caused it. Too
low feeds the model the previous frame's light, too high a future one. Measure it once
with `--debug-view`.

The requested camera resolution and rate are fixed at 1280×960 @30fps
(`CAM_WIDTH` / `CAM_HEIGHT` / `CAM_FPS` in `projector_distortion/pipeline/live.py`).
Drivers routinely ignore the request, which is why the startup line prints what the
camera actually opened at.

`Data.py` and the scripts under `data/` read the same kind of settings from
[configs/collect.yaml](projector_distortion/configs/collect.yaml), under `session:`,
`light:`, `capture:`, `warp:` and `record:`. Two files rather than one because the
collection scripts run without torch and never import the pipeline package.

### Flags

| | Default | Change it with |
|---|---|---|
| Projected clip | `data/live/test_light.mp4` — the held-out source | `--clip <path>` |
| Calibration background | `data/live/BaseBackGround.jpg` | fixed (`DEFAULT_LIVE_BG` in `projector_distortion/cli.py`) |
| Output | `output/<timestamp>/` + `calib/` + `result.mp4` | `--output <dir>` |

| Option | Default | Meaning |
|---|---|---|
| `--manual-calib` | off | Click the 4 corners instead of auto-detecting them |
| `--debug-view` | off | Live window with the pre-warp camera feed and the quad |

The run goes to the end of the clip, and the analysis stride is always measured rather
than named — see below.

```bash
python demo.py --live
python demo.py --live --save-every 30 --debug-view
```

Press `q` in the `Combined_View` window to stop.

On non-Windows systems this needs `pip install -e ".[live]"` for `screeninfo`. Windows
drives monitor placement through the Win32 API instead.

### Playback and analysis run at different rates

The projector plays the clip at the clip's own fps no matter how slow the model is.
Restore+detect runs on a worker thread over an evenly spaced subset — every Nth frame. The
recorded panel plays like the clip, only at a lower rate. Frames in between are still
projected and captured, just not scored.

N is measured, not named: the run times its first dozen analysed frames, discards the
CUDA warmup outlier, and fixes N from the median — the densest stride that stays
perfectly evenly spaced. It is never re-tuned mid-run, because changing N is itself
what makes the analysed video uneven.

The summary reports both rates:

```
projector 29.1 fps (450 frames) | analysis 13.0 fps (every 2 frame(s): 201 analysed,
249 skipped, 0 dropped)
```

`skipped` is by design — those frames were never meant for the model. `dropped` means the
worker missed its deadline. That is the number to watch.

Denser than the measured value costs evenness, which is why it is not offered as a
knob. On a 30 fps clip with a worker taking ~41 ms:

| stride | projector | analysis | frames analysed | spacing |
|---|---|---|---|---|
| `1` | 28.5 fps | 21.8 fps | 76% | 89% even |
| `2` (measured here) | 28.8 fps | 13.9 fps | 48% | 100% even |
| `3` | 28.8 fps | 9.4 fps | 32% | 100% even |

`1` analyses 1.6x more frames and barely touches playback, but it cannot hit every slot:
a 30 fps budget is 33 ms and the worker needs 41. The run picks `2` — the densest stride
that stays exactly even — because an unevenly sampled panel is harder to read than a
sparser one.

### Calibration

Black/white flash → the two camera shots are differenced, so ambient light cancels and
only the projection remains → the largest 4-corner contour is the screen → one homography
is computed and reused for every frame.

**It happens once, before the loop.** If the camera or projector moves mid-run, the warp
stays wrong for the rest of the session. Watch for that with `--debug-view`.

#### The run stops and shows you the warp

Because the estimate is made once and reused, a quad that is off by a corner poisons
every frame quietly — the panel still looks plausible and the residual is nonsense.
So the run holds the rectified first frame on screen, next to the raw camera view with
the detected quad drawn on it, and waits:

```
    auto calibration -> warp target 968x545:
      TL (   241,    118)
      TR (  1183,    131)
      BR (  1176,    701)
      BL (   233,    688)
      covers 41% of the 1280x960 camera frame
      edges  top 942  right 570  bottom 943  left 570 px
    [calibration] enter/a accept | m click the corners | r re-detect | q quit
```

| Key | Does |
|---|---|
| `enter` · `space` · `a` | Accept and start the run |
| `m` | Click the 4 corners on the live feed yourself. Any order — they get sorted |
| `r` | Flash black/white again and re-detect. Worth a try after killing a lamp or a reflection |
| `q` | Abort the run |

Read the two numbers under the corners: **covers** should be roughly the fraction of
frame the screen really occupies, and opposite **edges** should be close to equal. A
quad that grabbed a window or a lamp usually shows up as one of those being wildly off,
before you even look at the picture.

If auto-detection found nothing there is no preview and no accept — only `m`, `r` and
`q`. `--manual-calib` skips straight to clicking, and the review still runs afterwards.

Set `rig.review_calibration: false` in
[configs/live.yaml](projector_distortion/configs/live.yaml) for an unattended run,
where there is nobody to press a key. Auto-detection then falls back to manual clicking
on failure, exactly as it did before.

`output/<run>/calib/` is written even when detection failed — that is when it matters.

| File | What to look for |
|---|---|
| `quad.jpg` | Do the 4 points sit on the screen corners? |
| `mask.jpg` | Is the white region just the screen, or did lights get caught? |
| `diff.jpg` | Is the flash difference strong enough? If not, raise `CALIB_SETTLE` in `pipeline/live.py` |
| `warped.jpg` | Is the rectified result actually square? |
| `frame_pre.jpg` | The run's first camera frame, raw |
| `frame_post.jpg` | That frame rectified, at the model input size |
| `frame_compare.jpg` | Both side by side. Also shown once in `Warp_FirstFrame` |

The first four come from the flashes *before* the loop. The three `frame_*` files come
from the run itself, so they show the warp the frames were actually rectified with. If the
projection drifted in between, that is where it shows.

### Long unattended recording

```bash
python demo.py --live --save-every 300
```

The csv covers every frame while images land sparsely, which keeps disk usage predictable.
`--save-every 0` drops them entirely.

---

## `evaluate.py` — score before vs after

Only samples that have both `surface` and `label` are scored.

| | Default | Change it with |
|---|---|---|
| Input | `data/SampleData/sample_eval/` (`distorted/` + `light/`) | `--input <dir>` |
| GT | `data/SampleData/sample_eval/` (`surface/` + `labels/`) | `--gt <dir>` |
| Output | `output/Eval_<input dataset>/` | `--output <dir>` |

`labels/` may hold YOLO `.txt` or LabelMe `.json`; the extension decides which reader runs,
so the held-out split needs nothing but its path. LabelMe stores class *names*, and those
are matched against the detector's own class list — see
[data/README_data.md](data/README_data.md#label-format).

```bash
python evaluate.py --input data/SampleData/sample_test --gt data/SampleData/sample_test
```

Writes `report.json`, `per_class_<backend>.csv`, `per_image_<backend>.csv`.

| Option | Default | Meaning |
|---|---|---|
| `--iou <float>` | `0.5` | IoU threshold for a true positive |
| `--limit N` | `0` | Cap how many pairs are scored |

`evaluate.py` has no model flags at all — not even `--device`. Which backends it scores
comes from `detector.backend` in `configs/detection.yaml`, and a list there is what puts
one row per backend in the same report:

```yaml
detector:
  backend: [yolo, ssd]
```

```bash
python evaluate.py --iou 0.5
```

The report directory is named after the input dataset, so `data/SampleData/sample_eval`
scores into `output/Eval_sample_eval/` and the test split into `output/Eval_sample_test/`.
Re-running the same dataset replaces it, and stale per-backend csvs from the previous run
are cleared first.

A summary table is printed. Per-class P / R / F1 / AP goes into the csv.

```python
import pandas as pd
pc = pd.read_csv("output/Eval_sample_eval/per_class_yolo.csv")
pc.pivot_table(index="name", columns="source", values="ap")   # per-class AP shift
```

`mAP` here is the single-IoU-threshold average precision over classes — VOC style, area
under the interpolated PR curve. Not COCO's IoU-averaged metric.

> Each backend in the list loads its own checkpoint from `weights.<backend>`, which is
> what keeps a `[yolo, ssd]` comparison from running one backend's weights through the
> other.

---

## `train.py` — retrain the restorer

Needs the training extra: `pip install -e ".[train]"`. Only complete `distorted` / `light` /
`surface` triplets are used.

| | Default | Change it with |
|---|---|---|
| Data | `train.data` in `configs/restoration.yaml` | `--data-root <dir>`, or edit `train.data` |
| Output | `runs/<MMDD_HHMM>_<epochs>ep_<tag>/` | `--out <dir>` |

Writes `restorer_<tag>_best.pt`, `epoch_N.pt`, `loss_log.csv`, `loss_plots.png`.

### Where the training data comes from

The three directories are configured, not hard-coded. Each takes a directory, a glob, or a
list, because real captures come date-partitioned and the light frames usually sit under a
root of their own.

```yaml
train:
  data:
    distorted: "D:/captures/WarpData_*_pro"
    surface:   "D:/captures/WarpData_*_ori"
    light:     "D:/captures/Learning_video_frames"
```

Resolution order:

```
--data-root   >   train.data   >   data/SampleData/sample_train
```

`train.data` ships pointed at `sample_train`, the only split with no labels — training
never needs them. `sample_eval` and `sample_test` are held out for `evaluate.py`; pointing
`train.data` at either would score the model on what it fitted.

`--data-root` names one folder holding all three roles and wins outright, so a session
built by `Data.py` needs nothing else — point it at the `warp_<MMDD>/` folder. Leave it
off and the three configured directories are used; those are globbed as written, so a
relative path there is read against the working directory, not the project root — run
from the repo root, or give absolute paths.

```bash
python train.py --epochs 30                                 # uses train.data
python train.py --data-root data/Create_Data/warp_0813      # one folder, config ignored
```

`--data-root` detects the layout and looks for surface targets in this order, reading
whichever exists — no symlink or copy needed:

```
--data-root/OriginalImage/  →  --data-root/surface/  →  --gt/surface/
                            →  --data-root/clean/    →  --gt/clean/   (pre-rename)
```

A `distorted` with no matching `surface` or `light` is counted and skipped, not fatal. The run
prints how many went each way.

```
data: 994 triplets of 1,000 distorted image(s) from distorted=data/SampleData/sample_train/distorted
      skipped 6 without a light, 0 without a surface
```

### Options

| Option | Default | Meaning |
|---|---|---|
| `--epochs N` | `30` | From `train.epochs` in `configs/restoration.yaml` |
| `--batch-size N` | `4` | From `train.batch_size` |
| `--lr F` | `0.0002` | From `train.lr` |
| `--sample N` | `0` | Cap how many triplets are used |
| `--resume <ckpt>` | — | Continue from a checkpoint. Its architecture wins over `ablation:` |
| `--no-amp` | off | Disable mixed precision on CUDA |
| `--device cuda\|cpu` | cuda if present | — |

Fixed rather than flagged: the gradient accumulation window is `train.accum_steps` in
the YAML, the seed is `42` (`SEED` in `train.py`), DataLoader workers are `4`, or `8` for
batch > 4, and every epoch writes an `epoch_N.pt`.

```bash
python train.py --epochs 30
python train.py --resume runs/0730_1948_30ep_FULL/restorer_FULL_best.pt --epochs 10
```

Loss is `L1 + perceptual + SSIM + wavelet`, weighted from `train.loss`. Each triplet is
resized to 360×640 then randomly cropped to 180×320. What the loss measures and why:
[weights/README_weights.md](weights/README_weights.md#the-residual-convention).

### Ablation

Ten structural pieces can be switched off individually, in the `ablation:` block of
`projector_distortion/configs/restoration.yaml`. Whatever is off lands in `tag`, which
goes into both the run folder and the checkpoint filename.

| Set to `false` | Turns off | Tag |
|---|---|---|
| `use_prenorm` | The pre-LayerNorm of NAFSEBlock | `NoPre` |
| `use_naf_norm` | LayerNorm2d inside NAFBlock | `NoNorm` |
| `use_simple_gate` | SimpleGate (`x1*x2`), replaced by GELU | `NoGate` |
| `use_naf_scale` | The learnable residual scales beta / gamma | `NoScale` |
| `use_ca` | Channel attention; the block becomes a plain NAFBlock | `NoCA` |
| `use_skip1` | U-Net skip enc1 → dec1 (full resolution) | `NoSkip1` |
| `use_skip2` | U-Net skip enc2 → dec2 (1/2) | `NoSkip2` |
| `use_skip3` | U-Net skip enc3 → dec3 (1/4) | `NoSkip3` |
| `use_bottleneck` | The 1/8-resolution bottleneck, replaced by Identity | `NoBott` |
| `use_tanh` | The output tanh; the residual becomes unbounded | `NoTanh` |

Several combine: `NoCA-NoSkip3`. Nothing off is `FULL`.

Capacity sits in the same block: `base_dim` (48), `enc_depth` (`[2, 2, 3]`), `dec_depth`
(`[2, 2, 2]`), `bottleneck_depth` (2), `dw_expand` (2), `ffn_expand` (2), `ca_reduction`
(16).

```yaml
# projector_distortion/configs/restoration.yaml
ablation:
  use_ca: false
```

```bash
python train.py --epochs 30      # writes runs/<date>_30ep_NoCA/
```

Checkpoints embed their own architecture config, so the `ablation:` block does not have
to stay set for inference — point `model.weights` at the new checkpoint and run:

```yaml
# projector_distortion/configs/restoration.yaml
model:
  weights: runs/0730_1948_30ep_NoCA/restorer_NoCA_best.pt
```

The exception is a *legacy* checkpoint saved as a bare `state_dict` with no config in it.
Those fall back to whatever `ablation:` currently says, so an ablated legacy checkpoint
needs that block set to match it before it will load correctly. Every checkpoint
`train.py` writes carries its own config, so this only affects weights from elsewhere.

---

## Output format

`demo.py` writes one directory per run:

```
output/<run_name>/
├── run_meta.json      config · environment · calibration · summary, all in one file
├── detections.csv     one row per box; `source` separates distorted / restored
├── captures/          the footage itself, no boxes drawn on it
│   ├── <id>_distorted.jpg      before restoration
│   └── <id>_restored.jpg       after restoration
├── frames_all/        the 2×2 figures
│   └── <id>_panel.jpg          light · distorted+boxes · restored+boxes · residual
├── calib/             --live only
└── result.mp4         video of the 2×2 panels (--live or --video)
```

`captures/` holds the run un-annotated. That is what lets `evaluate.py` score this
restoration afterwards, or another detector re-run over identical pixels. Boxes burnt into
a jpg cannot be undone.

Only three kinds are written. The annotated views, the residual heatmap and the light frame are
tiles of the panel already, so writing them again cost four extra encodes a frame and
bought nothing.

`--save-every 0` skips the image directories entirely. `detections.csv` still covers every
frame.

`evaluate.py` writes `report.json` + `per_class_*.csv` + `per_image_*.csv` instead.
`train.py` writes `restorer_<tag>_best.pt` + `loss_log.csv` + `loss_plots.png` under
`runs/`.

---

## Making restoration faster

Restoration is ~46% of a live frame, so it is the first thing to shrink. The network is
fully convolutional, which makes its working resolution a runtime knob — no retraining.
Set it with `model.input_size` in `projector_distortion/configs/restoration.yaml`.
Measured on the bundled set with the `yolo` detector:

| `model.input_size` | restore | detection mAP | PSNR gain | SSIM gain |
|---|---|---|---|---|
| `[320, 180]` | 9.7 ms | 0.9866 | +8.95 dB | +0.163 |
| `[480, 270]` | 13.4 ms | **1.0000** | +11.32 dB | +0.183 |
| `[640, 360]` (default) | 20.5 ms | **1.0000** | **+13.17 dB** | **+0.218** |
| `[854, 480]` | 41.2 ms | 1.0000 | +9.89 dB | +0.145 |

```yaml
# projector_distortion/configs/restoration.yaml
model:
  input_size: [480, 270]
```

`480 270` costs a third of the restoration time and detection does not notice. Only the
PSNR/SSIM gain narrows. Below that, mAP starts to slip.

Going **above** 640×360 is worse on both counts. The checkpoint was trained on 180×320
crops resized from 360×640, and 854×480 is far enough outside that scale that quality
drops while costing twice the time.

Mixed precision and `torch.compile` are not worth reaching for, which is why neither is
wired up. fp16 autocast measured 2% *slower* — the network is small enough to be
memory-bound, so autocast adds more than it saves. compile needs a Triton build Windows
does not ship.

A genuinely smaller network needs retraining. At 640×360:

| `ablation:` change | params | forward |
|---|---|---|
| shipped | 4,184,259 | 13.8 ms |
| `base_dim: 32` | 1,878,659 | 9.7 ms |
| `enc_depth: [1,1,1]` + `dec_depth: [1,1,1]` + `bottleneck_depth: 1` | 2,418,837 | 7.3 ms |
| `use_ca: false` | 4,116,147 | 13.1 ms |

Shallower blocks buy more than narrower ones. Dropping channel attention buys 5% and is
not worth the retrain.

---

[한국어](README_running.ko.md) · [← README](README.md)
