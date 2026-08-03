# Running ProRes-Det

Input, output and options for each entry point. For the one-liners that just work, see
[Quick start](../README.md#3-running).

- [`demo.py` — restore → detect](#demopy--restore--detect)
- [`evaluate.py` — score before vs after](#evaluatepy--score-before-vs-after)
- [`train.py` — retrain the restoration model](#trainpy--retrain-the-restoration-model)
- [`demo.py --live` — webcam + projector](#demopy---live--webcam--projector)
- [Output format](#output-format)

Flags shared by all three scripts: `--restorer-weights`, `--det-weights`, `--device`,
`--fp16`, `--classes`, `--restoration-config`, `--detection-config`. Full list:
`python <script>.py --help`.

---

## `demo.py` — restore → detect

| | Default path | Option to change it |
|---|---|---|
| Input | `data/sample_input/` (`pro/` + `beam/`) | `--input <dir>` |
| GT (optional) | `data/sample_gt/` (`clean/` + `labels/`) | `--gt <dir>` |
| Output | `output/<timestamp>/` | `--output <dir>` · `--name <name>` |

Without GT only PSNR/SSIM is skipped; the run continues.

| Option | Meaning |
|---|---|
| `--detector yolo\|ssd\|none` | Detection backend (default yolo) |
| `--conf <float>` | Detector confidence floor (default 0.25) |
| `--limit N` | Cap how many pairs are processed (0 = all) |
| `--best-per-class` | Keep only the highest-confidence box per class |
| `--save-every N` | Image save interval. `0` keeps csv only |
| `--save-kinds a,b` | Save only selected image kinds (`captured,restored,panel`, …) |
| `--max-saved-frames N` | Hard cap on how many frame sets land on disk |
| `--jpeg-quality N` | JPEG quality for saved images (default 92) |
| `--video` | Also write the 2×2 panels as `result.mp4` |

```bash
python demo.py --detector ssd --conf 0.4
python demo.py --detector none --save-kinds captured,restored,beam
python demo.py --input /path/to/pairs --gt /path/to/gt --name my_run
```

`--save-every 0` writes no images at all — useful when you only want the csv and the
summary, e.g. when sweeping detectors:

```bash
for D in yolo ssd; do python demo.py --detector $D --save-every 0 --name run_$D; done
```

---

## `evaluate.py` — score before vs after

Only samples that have both `clean` and `label` are scored.

| | Default path | Option to change it |
|---|---|---|
| Input | `data/sample_input/` (`pro/` + `beam/`) | `--input <dir>` |
| GT | `data/sample_gt/` (`clean/` + `labels/`) | `--gt <dir>` |
| Output | `output/eval/`<br>`report.json`, `per_class_<backend>.csv`, `per_image_<backend>.csv` | `--output <dir>` · `--name <name>` |

| Option | Meaning |
|---|---|
| `--detectors yolo,ssd` | Compare several backends in one run (one row each) |
| `--iou <float>` | IoU threshold for a true positive (default 0.5) |
| `--limit N` | Cap how many pairs are scored |
| `--best-per-class` | Keep only the highest-confidence box per class |

```bash
python evaluate.py --detectors yolo,ssd --iou 0.5
```

A summary table is printed; per-class P / R / F1 / AP goes into the csv.

```python
import pandas as pd
pc = pd.read_csv("output/eval/per_class_yolo.csv")
pc.pivot_table(index="name", columns="source", values="ap")   # per-class AP shift
```

`mAP` here is the single-IoU-threshold average precision over classes (VOC style, area
under the interpolated PR curve) — not COCO's IoU-averaged metric.

> `--det-weights` names one checkpoint, so it can only belong to one backend. With
> `--detectors a,b`, say which one owns it via `--detector`; otherwise it is warned
> about and ignored, and both backends fall back to `configs/detection.yaml`.

---

## `train.py` — retrain the restoration model

Only complete `pro` / `beam` / `clean` triplets are used.

| | Default path | Option to change it |
|---|---|---|
| Input | `data/sample_input/` (`pro/` + `beam/`) | `--data-root <dir>` |
| Target | `data/sample_gt/clean/` — **required** | `--gt <dir>` |
| Output | `runs/<MMDD_HHMM>_<epochs>ep_<tag>/`<br>`restorer_<tag>_best.pt`, `epoch_N.pt`, `loss_log.csv`, `loss_plots.png` | `--out <dir>` |

Clean targets are searched in this order, and whichever exists is read directly — no
symlink or copy needed (filename convention: [data/README_data.md](../data/README_data.md)):

```
--data-root/OriginalImage/   →   --data-root/clean/   →   --gt/clean/
```

| Option | Meaning |
|---|---|
| `--epochs N` `--batch-size N` `--lr F` `--accum-steps N` | Defaults come from `configs/restoration.yaml` |
| `--sample N` | Cap how many triplets are used |
| `--num-workers N` | DataLoader workers (default 4, or 8 for batch > 4) |
| `--resume <ckpt>` | Continue from a checkpoint (its architecture wins over `--no-*`) |
| `--seed N` | Default 42 |
| `--no-amp` | Disable mixed precision on CUDA |
| `--save-every N` | Epochs between `epoch_N.pt` saves |

```bash
python train.py --epochs 30
python train.py --data-root /path/to/dataset --epochs 30
python train.py --resume runs/0730_1948_30ep_FULL/restorer_FULL_best.pt --epochs 10
```

Needs the training extra: `pip install -e ".[train]"`.

### Ablation

Ten structural pieces can be switched off individually. Whatever is off lands in `tag`,
which goes into both the run folder and the checkpoint filename (`NoCA`,
`NoCA-NoSkip3`, …).

```
--no-prenorm  --no-naf-norm  --no-simple-gate  --no-naf-scale  --no-ca
--no-skip1    --no-skip2     --no-skip3        --no-bottleneck  --no-tanh
```

Capacity can be overridden too: `--base-dim`, `--enc-depth`, `--dec-depth`,
`--bottleneck-depth`, `--dw-expand`, `--ffn-expand`, `--ca-reduction`.

```bash
python train.py --no-ca --epochs 30
```

Checkpoints embed their own architecture config, so the flags never need repeating:

```bash
python demo.py --restorer-weights runs/0730_1948_30ep_NoCA/restorer_NoCA_best.pt
```

---

## `demo.py --live` — webcam + projector

Needs real hardware: a webcam pointed at a screen a projector is throwing to.

| | Default | Option to change it |
|---|---|---|
| Projected clip | `data/live/BeamVideo.mp4` | `--clip <path>` |
| Calibration background | `data/live/BaseBackGround.jpg` | `--background <path>` |
| Camera | webcam 0, 1280×960 @30fps | `--camera N` · `--cam-width/height/fps` · `--cam-backend` |
| Output | `output/<timestamp>/` + `calib/` + `result.mp4` | `--output <dir>` · `--name <name>` |

| Option | Meaning |
|---|---|
| `--screen N` | Monitor index the projector is attached to (0 = primary) |
| `--offset N` | Projector→camera latency in frames (default 6) |
| `--manual-calib` | Click the 4 corners instead of auto-detecting them |
| `--debug-view` | Show the pre-warp camera feed with the quad, live |
| `--calib-settle F` | Seconds to wait after each calibration flash (default 0.8) |
| `--max-frames N` | 0 = until the clip ends |

```bash
python demo.py --live --screen 2
python demo.py --live --screen 2 --save-every 30 --debug-view
```

If you do not know `--screen`, pass anything and run — the detected monitor table is
printed first. Press `q` in the `Combined_View` window to stop.

### How calibration works

Black/white flash → the two camera shots are differenced (only the projection changes,
so ambient light cancels) → the largest 4-corner contour is the screen → one homography
is computed and reused for every frame.

**Calibration happens once, before the loop.** If the camera or projector moves mid-run,
the warp stays wrong for the rest of the session. Watch for that with `--debug-view`.
Auto-detection falls back to manual clicking on failure; `--manual-calib` starts there.

### When calibration looks wrong

`output/<run>/calib/` is written even when detection failed — that is when it matters.

| File | What to look for |
|---|---|
| `raw_points.jpg` | Do the 4 points sit exactly on the screen corners? |
| `mask.jpg` | Is the white region just the screen, or did lights/windows get caught? |
| `diff.jpg` | Is the black/white flash difference strong enough? If not, raise `--calib-settle` |
| `warped.jpg` | Is the rectified result actually square? |

### Long unattended recording

```bash
python demo.py --live --screen 2 --save-every 300 --max-saved-frames 50 --jpeg-quality 85
```

The csv covers every frame while images land sparsely, which keeps disk usage
predictable.

On non-Windows systems this needs `pip install -e ".[live]"` for `screeninfo`. Windows
drives monitor placement through the Win32 API instead.

---

## Output format

`demo.py` writes one directory per run:

```
output/<run_name>/
├── run_meta.json      config · environment · calibration · summary (all in one file)
├── detections.csv     one row per box, `source` separates captured / restored
├── frames.csv         one row per frame (PSNR/SSIM/latency included)
├── frames/            sampled images, --save-every apart
│   ├── <id>_captured.jpg      before restoration (no boxes)
│   ├── <id>_restored.jpg      after restoration (no boxes)  ← metric input
│   ├── <id>_*_det.jpg         the annotated versions
│   ├── <id>_residual.jpg      heatmap of the removed light
│   └── <id>_panel.jpg         2×2 comparison panel
├── calib/             calibration evidence (--live only)
└── result.mp4         video of the 2×2 panels (--live or --video)
```

Keeping clean and annotated images separate is the point. It is what makes
recomputing PSNR/SSIM and re-running a different detector possible after the fact.

`--save-every 0` skips `frames/` entirely; the csv files always cover every frame.

`evaluate.py` writes `report.json` + `per_class_*.csv` + `per_image_*.csv` instead, and
`train.py` writes `restorer_<tag>_best.pt` + `loss_log.csv` + `loss_plots.png` under
`runs/`.

---

[한국어](README_running.ko.md) · [← README](../README.md)
