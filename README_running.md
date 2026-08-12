# Running ProRes-Det

Options, input and output for the three root entry points. For install and the
one-liners that just work, see [README.md](README.md).

`collect.py` and `record.py` live under `data/` and are documented in
[data/README_data.md](data/README_data.md).

- [Shared flags](#shared-flags)
- [`demo.py` — restore → detect](#demopy--restore--detect)
- [`demo.py --live` — webcam + projector](#demopy---live--webcam--projector)
- [`evaluate.py` — score before vs after](#evaluatepy--score-before-vs-after)
- [`train.py` — retrain the restorer](#trainpy--retrain-the-restorer)
- [Output format](#output-format)
- [Making restoration faster](#making-restoration-faster)

---

## Shared flags

All three scripts understand these. Full list: `python <script>.py --help`.

| Flag | Default | Meaning |
|---|---|---|
| `--restorer <name>` | `naf_se_unet` | Restoration backend; any name registered with `@register_restorer` |
| `--restorer-weights <path>` | from `restoration.yaml` | Restoration checkpoint |
| `--device cuda\|cpu` | cuda if present | — |
| `--input-size W H` | `640 360` | What the restorer runs at |
| `--restoration-config <yaml>` | — | Merged over `configs/restoration.yaml` |
| `--detection-config <yaml>` | — | Merged over `configs/detection.yaml` |

`demo.py` and `evaluate.py` add the detection flags. `train.py` does not — it only ever
fits the restoration network.

| Flag | Default | Meaning |
|---|---|---|
| `--detector <name>` | `yolo` | `yolo` · `ssd` · `none`, or any backend registered with `@register_detector` |
| `--det-weights <path>` | per backend, from `detection.yaml` | Detector checkpoint |
| `--conf <float>` | `0.25` | Detector confidence floor |

Class names come from `names:` in `configs/detection.yaml` — override them with a small
`--detection-config` YAML. A YOLO checkpoint's own names win over that list.

`train.py` also takes the 10 `--no-*` ablation flags. See [Ablation](#ablation). The two
inference scripts do not need them: a checkpoint carries the architecture it was trained
with.

---

## `demo.py` — restore → detect

Restoration and detection only. It never reads ground truth and never scores anything —
that is `evaluate.py`'s job.

| | Default | Change it with |
|---|---|---|
| Input | `data/SampleData/sample_eval/` (`distorted/` + `light/`) | `--input <dir>` |
| Output | `output/<timestamp>/` | `--output <dir>` · `--name <name>` |

| Option | Default | Meaning |
|---|---|---|
| `--limit N` | `0` | Cap how many pairs are processed. `0` = all |
| `--save-every N` | `1` | Image save interval. `0` keeps the csv only |
| `--save-kinds a,b` | all three | Subset of `distorted`, `restored`, `panel` |
| `--video` | off | Also write the 2×2 panels as `result.mp4` |

Images are written at JPEG quality 92. Box size gating comes from
`detector.min_width` / `min_height` / `min_area` in `configs/detection.yaml`; every box
that clears it is kept, because metric code must not lose duplicates.

```bash
python demo.py --detector ssd --conf 0.4
python demo.py --detector none --save-kinds restored
python demo.py --input /path/to/pairs --name my_run
```

`--save-every 0` writes no images at all. Useful when only the csv and the summary
matter, e.g. sweeping detectors:

```bash
for D in yolo ssd; do python demo.py --detector $D --save-every 0 --name run_$D; done
```

---

## `demo.py --live` — webcam + projector

Needs real hardware: a webcam pointed at a screen a projector is throwing to.

| | Default | Change it with |
|---|---|---|
| Projected clip | `data/live/BeamVideo.mp4` | `--clip <path>` |
| Calibration background | `data/live/BaseBackGround.jpg` | `--background <path>` |
| Camera | index 0, 1280×960 @30fps | `--camera N` · `--cam-width/height/fps` · `--cam-backend` |
| Output | `output/<timestamp>/` + `calib/` + `result.mp4` | `--output <dir>` · `--name <name>` |

| Option | Default | Meaning |
|---|---|---|
| `--screen N` | `1` | Monitor index the projector is attached to. `0` = primary |
| `--offset N` | `6` | Projector→camera latency, in frames |
| `--analyse-every N` | `0` | Restore+detect every Nth frame. `0` = measure and pick |
| `--max-frames N` | `0` | `0` = until the clip ends |
| `--manual-calib` | off | Click the 4 corners instead of auto-detecting them |
| `--debug-view` | off | Live window with the pre-warp camera feed and the quad |
| `--calib-settle F` | `0.8` | Seconds to wait after each calibration flash |
| `--cam-backend` | `auto` | `auto` · `any` · `dshow` · `msmf` · `v4l2` |

```bash
python demo.py --live --screen 2
python demo.py --live --screen 2 --save-every 30 --debug-view
```

Press `q` in the `Combined_View` window to stop.

On non-Windows systems this needs `pip install -e ".[live]"` for `screeninfo`. Windows
drives monitor placement through the Win32 API instead.

### Playback and analysis run at different rates

The projector plays the clip at the clip's own fps no matter how slow the model is.
Restore+detect runs on a worker thread over an evenly spaced subset — every Nth frame. The
recorded panel plays like the clip, only at a lower rate. Frames in between are still
projected and captured, just not scored.

`--analyse-every 0` times the first dozen analysed frames, discards the CUDA warmup
outlier, and fixes N from the median. It is never re-tuned mid-run, because changing N is
itself what makes the analysed video uneven.

The summary reports both rates:

```
projector 29.1 fps (450 frames) | analysis 13.0 fps (every 2 frame(s): 201 analysed,
249 skipped, 0 dropped)
```

`skipped` is by design — those frames were never meant for the model. `dropped` means the
worker missed its deadline. That is the number to watch.

Denser than the auto value costs evenness. On a 30 fps clip with a worker taking ~41 ms:

| `--analyse-every` | projector | analysis | frames analysed | spacing |
|---|---|---|---|---|
| `1` | 28.5 fps | 21.8 fps | 76% | 89% even |
| `2` (auto here) | 28.8 fps | 13.9 fps | 48% | 100% even |
| `3` | 28.8 fps | 9.4 fps | 32% | 100% even |

`1` analyses 1.6x more frames and barely touches playback. It just cannot hit every slot:
a 30 fps budget is 33 ms and the worker needs 41. Pick it when coverage matters more than
a smooth panel.

### Calibration

Black/white flash → the two camera shots are differenced, so ambient light cancels and
only the projection remains → the largest 4-corner contour is the screen → one homography
is computed and reused for every frame.

**It happens once, before the loop.** If the camera or projector moves mid-run, the warp
stays wrong for the rest of the session. Watch for that with `--debug-view`.
Auto-detection falls back to manual clicking on failure; `--manual-calib` starts there.

`output/<run>/calib/` is written even when detection failed — that is when it matters.

| File | What to look for |
|---|---|
| `quad.jpg` | Do the 4 points sit on the screen corners? |
| `mask.jpg` | Is the white region just the screen, or did lights get caught? |
| `diff.jpg` | Is the flash difference strong enough? If not, raise `--calib-settle` |
| `warped.jpg` | Is the rectified result actually square? |
| `frame_pre.jpg` | The run's first camera frame, raw |
| `frame_post.jpg` | That frame rectified, at the model input size |
| `frame_compare.jpg` | Both side by side. Also shown once in `Warp_FirstFrame` |

The first four come from the flashes *before* the loop. The three `frame_*` files come
from the run itself, so they show the warp the frames were actually rectified with. If the
projection drifted in between, that is where it shows.

### Long unattended recording

```bash
python demo.py --live --screen 2 --save-every 300 --save-kinds panel
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
| Output | `output/Eval_<input dataset>/` | `--output <dir>` · `--name <name>` |

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
| `--detector yolo,ssd` | `yolo` | Comma separated here: one row per backend, compared in one run |
| `--iou <float>` | `0.5` | IoU threshold for a true positive |
| `--limit N` | `0` | Cap how many pairs are scored |

```bash
python evaluate.py --detector yolo,ssd --iou 0.5
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

> `--det-weights` names one checkpoint, so it can belong to only one backend. With
> `--detector a,b` it is warned about and ignored, and every backend falls back to
> `configs/detection.yaml`.

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
built by `collect.py` needs nothing else. Leave it off and the three configured
directories are used; those are globbed as written, so a relative path there is read
against the working directory, not the project root — run from the repo root, or give
absolute paths.

```bash
python train.py --epochs 30                              # uses train.data
python train.py --data-root data/collected_0803          # one folder, config ignored
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
| `--epochs N` | `30` | From `configs/restoration.yaml` |
| `--batch-size N` | `4` | ″ |
| `--accum-steps N` | `4` | Gradient accumulation window |
| `--lr F` | `0.0002` | ″ |
| `--sample N` | `0` | Cap how many triplets are used |
| `--num-workers N` | `4`, or `8` for batch > 4 | DataLoader workers |
| `--resume <ckpt>` | — | Continue from a checkpoint. Its architecture wins over `--no-*` |
| `--seed N` | `42` | — |
| `--no-amp` | off | Disable mixed precision on CUDA |
| `--save-every N` | `1` | Epochs between `epoch_N.pt` saves |

```bash
python train.py --epochs 30
python train.py --resume runs/0730_1948_30ep_FULL/restorer_FULL_best.pt --epochs 10
```

Loss is `L1 + perceptual + SSIM + wavelet`, weighted from `train.loss`. Each triplet is
resized to 360×640 then randomly cropped to 180×320. What the loss measures and why:
[weights/README_weights.md](weights/README_weights.md#the-residual-convention).

### Ablation

Ten structural pieces can be switched off individually. Whatever is off lands in `tag`,
which goes into both the run folder and the checkpoint filename.

| Flag | Turns off | Tag |
|---|---|---|
| `--no-prenorm` | The pre-LayerNorm of NAFSEBlock | `NoPre` |
| `--no-naf-norm` | LayerNorm2d inside NAFBlock | `NoNorm` |
| `--no-simple-gate` | SimpleGate (`x1*x2`), replaced by GELU | `NoGate` |
| `--no-naf-scale` | The learnable residual scales beta / gamma | `NoScale` |
| `--no-ca` | Channel attention; the block becomes a plain NAFBlock | `NoCA` |
| `--no-skip1` | U-Net skip enc1 → dec1 (full resolution) | `NoSkip1` |
| `--no-skip2` | U-Net skip enc2 → dec2 (1/2) | `NoSkip2` |
| `--no-skip3` | U-Net skip enc3 → dec3 (1/4) | `NoSkip3` |
| `--no-bottleneck` | The 1/8-resolution bottleneck, replaced by Identity | `NoBott` |
| `--no-tanh` | The output tanh; the residual becomes unbounded | `NoTanh` |

Several combine: `NoCA-NoSkip3`. Nothing off is `FULL`.

Capacity can be overridden too: `--base-dim` (48), `--enc-depth` (`2,2,3`), `--dec-depth`
(`2,2,2`), `--bottleneck-depth` (2), `--dw-expand` (2), `--ffn-expand` (2),
`--ca-reduction` (16).

```bash
python train.py --no-ca --epochs 30
```

Checkpoints embed their own architecture config, so the flags never need repeating — and
`demo.py` / `evaluate.py` do not accept them at all:

```bash
python demo.py --restorer-weights runs/0730_1948_30ep_NoCA/restorer_NoCA_best.pt
```

The exception is a *legacy* checkpoint saved as a bare `state_dict` with no config in it.
Those fall back to the default (`FULL`) architecture; an ablated one has to be described
through the `ablation:` block of a `--restoration-config` YAML instead. Every checkpoint
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
bought nothing. Drop kinds further with `--save-kinds`:

```bash
python demo.py --save-kinds panel                # comparison figures only
python demo.py --save-kinds distorted,restored   # re-scorable pixels only
```

`--save-every 0` skips the image directories entirely. `detections.csv` still covers every
frame.

`evaluate.py` writes `report.json` + `per_class_*.csv` + `per_image_*.csv` instead.
`train.py` writes `restorer_<tag>_best.pt` + `loss_log.csv` + `loss_plots.png` under
`runs/`.

---

## Making restoration faster

Restoration is ~46% of a live frame, so it is the first thing to shrink. The network is
fully convolutional, which makes its working resolution a runtime knob — no retraining.
Measured on the bundled set with `--detector yolo`:

| `--input-size` | restore | detection mAP | PSNR gain | SSIM gain |
|---|---|---|---|---|
| `320 180` | 9.7 ms | 0.9866 | +8.95 dB | +0.163 |
| `480 270` | 13.4 ms | **1.0000** | +11.32 dB | +0.183 |
| `640 360` (default) | 20.5 ms | **1.0000** | **+13.17 dB** | **+0.218** |
| `854 480` | 41.2 ms | 1.0000 | +9.89 dB | +0.145 |

```bash
python demo.py --live --screen 2 --input-size 480 270
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

| `train.py` flags | params | forward |
|---|---|---|
| shipped | 4,184,259 | 13.8 ms |
| `--base-dim 32` | 1,878,659 | 9.7 ms |
| `--enc-depth 1,1,1 --dec-depth 1,1,1 --bottleneck-depth 1` | 2,418,837 | 7.3 ms |
| `--no-ca` | 4,116,147 | 13.1 ms |

Shallower blocks buy more than narrower ones. Dropping channel attention buys 5% and is
not worth the retrain.

---

[한국어](README_running.ko.md) · [← README](README.md)
