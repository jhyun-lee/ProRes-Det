# Running ProRes-Det

Input, output and options for each entry point. For the one-liners that just work, see
[Quick start](README.md#3-running).

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

Restoration and detection only. It never touches ground truth and never scores
anything - `evaluate.py` does that.

| | Default path | Option to change it |
|---|---|---|
| Input | `data/sample_input/` (`pro/` + `beam/`) | `--input <dir>` |
| Output | `output/<timestamp>/` | `--output <dir>` · `--name <name>` |

| Option | Meaning |
|---|---|
| `--detector yolo\|ssd\|none` | Detection backend (default yolo) |
| `--conf <float>` | Detector confidence floor (default 0.25) |
| `--limit N` | Cap how many pairs are processed (0 = all) |
| `--best-per-class` | Keep only the highest-confidence box per class |
| `--save-every N` | Image save interval. `0` keeps csv only |
| `--save-kinds a,b` | Which image kinds to save (default: everything except `beam`) |
| `--max-saved-frames N` | Hard cap on how many frame sets land on disk |
| `--jpeg-quality N` | JPEG quality for saved images (default 92) |
| `--video` | Also write the 2×2 panels as `result.mp4` |

```bash
python demo.py --detector ssd --conf 0.4
python demo.py --detector none --save-kinds restored,residual
python demo.py --input /path/to/pairs --name my_run
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
| Output | `output/Eval_<input dataset>/`<br>`report.json`, `per_class_<backend>.csv`, `per_image_<backend>.csv` | `--output <dir>` · `--name <name>` |

| Option | Meaning |
|---|---|
| `--detectors yolo,ssd` | Compare several backends in one run (one row each) |
| `--iou <float>` | IoU threshold for a true positive (default 0.5) |
| `--limit N` | Cap how many pairs are scored |
| `--best-per-class` | Keep only the highest-confidence box per class |

```bash
python evaluate.py --detectors yolo,ssd --iou 0.5
```

The report directory is named after the input dataset, so `data/sample_input` scores
into `output/Eval_sample_input/`. Re-running the same dataset replaces it, and stale
per-backend csvs from the previous run are cleared first.

A summary table is printed; per-class P / R / F1 / AP goes into the csv.

```python
import pandas as pd
pc = pd.read_csv("output/Eval_sample_input/per_class_yolo.csv")
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
symlink or copy needed (filename convention: [data/README_data.md](data/README_data.md)):

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

### The first frame's warp

`output/<run>/warp/` is written once, when the first camera frame reaches the loop, and
the comparison figure is put on screen in the `Warp_FirstFrame` window while the run
carries on.

| File | What it is |
|---|---|
| `first_frame_pre_warp.jpg` | the raw camera frame |
| `first_frame_pre_warp_quad.jpg` | the same frame with the calibration quad on it |
| `first_frame_post_warp.jpg` | rectified, at the model input size |
| `first_frame_compare.jpg` | both side by side — the figure shown on screen |

`calib/` comes from the black/white flashes taken *before* the loop; this comes from the
run itself, so it is the warp the frames were actually rectified with. If the projection
drifted between calibration and the first frame, this is where it shows.

### Long unattended recording

```bash
python demo.py --live --screen 2 --save-every 300 --max-saved-frames 50 --jpeg-quality 85
```

The csv covers every frame while images land sparsely, which keeps disk usage
predictable.

On non-Windows systems this needs `pip install -e ".[live]"` for `screeninfo`. Windows
drives monitor placement through the Win32 API instead.

---

## `data/collect.py` — build a dataset with the rig

Four stages. `check` and `capture` drive the projector and the webcam; `beam` and `warp`
are plain file work and run anywhere. Everything lands in one session folder
(`data/collected_<MMDD>/` by default, `--root` moves it).

| Stage | Command | In → out |
|---|---|---|
| check | `python data/collect.py check` | monitor table + webcam preview |
| beam | `python data/collect.py beam --src videos/` | video → `<session>/beam/` |
| capture | `python data/collect.py capture --screen 2` | projector + webcam → `<session>/raw/` |
| warp | `python data/collect.py warp` | `raw/` → aligned `<session>/clean/` + `pro/` |

```bash
python data/collect.py beam    --src data/live/BeamVideo.mp4 --step 30
python data/collect.py capture --screen 2 --rounds 3 --limit 200 --shuffle
python data/collect.py warp
python demo.py --input data/collected_0803 --gt data/collected_0803
```

`capture`:

| Option | Meaning |
|---|---|
| `--screen N` | Monitor the projector is attached to (`check` prints the table) |
| `--rounds N` | Scene setups; each one starts with a fresh clean shot, `s` takes it |
| `--limit N` · `--shuffle` | Beam frames per round, and whether to spread them over the clip |
| `--settle-ms N` | Wait after showing a frame before capturing it (default 150) |
| `--flush N` | Buffered camera frames to drop first (default 3) — this is what keeps `pro` matched to its `beam` |
| `--background <path>` | Image projected while the clean shot is taken |

`warp`:

| Option | Meaning |
|---|---|
| `--warp boundary` | Default: corner homography + the measured edge bow |
| `--warp homography` | Corners only, for a flat screen and a clean boundary |
| `--warp tps` | The legacy thin-plate spline; needs `opencv-python<5`, which still ships the shape module |
| `--inset N` | Pixels to pull the boundary in, off the bright projection rim (default 2) |
| `--final-size W H` | Saved resolution, default 640×360 — the model input |
| `--no-debug` | Skip the before/after overlays in `<session>/debug/` |

Check `<session>/debug/<oriId>_warp.jpg` before capturing a full session: it shows the
detected boundary, the sampled points and the rectified result together. Naming, the
`oriId`/`beamId` contract and what to do about labels are in
[data/README_data.md](data/README_data.md).

---

## Output format

`demo.py` writes one directory per run:

```
output/<run_name>/
├── run_meta.json      config · environment · calibration · summary (all in one file)
├── detections.csv     one row per box, `source` separates distorted / restored
├── captures/          the footage itself, no boxes drawn on it
│   ├── <id>_distorted.jpg      before restoration
│   └── <id>_restored.jpg      after restoration
├── frames/            annotated views, --save-every apart
│   ├── <id>_distorted_det.jpg  before restoration, with boxes
│   ├── <id>_restored_det.jpg  after restoration, with boxes
│   └── <id>_residual.jpg      heatmap of the removed light
├── frames_all/        the 2×2 figures, tiles captioned (a)…(d)
├── calib/             calibration evidence (--live only)
├── warp/              the first frame before and after rectification (--live only)
└── result.mp4         video of the 2×2 panels (--live or --video)
```

`captures/` holds the run un-annotated. That is what lets `evaluate.py` score this
restoration afterwards, or another detector re-run over identical pixels - boxes burnt
into a jpg cannot be undone.

`beam` is the one kind off by default, since the panel already shows it. Add it with
`--save-kinds`, which also accepts any other subset:

```bash
python demo.py --save-kinds distorted,restored,panel,beam
```

`--save-every 0` skips the image directories entirely; `detections.csv` still covers
every frame.

`evaluate.py` writes `report.json` + `per_class_*.csv` + `per_image_*.csv` instead, and
`train.py` writes `restorer_<tag>_best.pt` + `loss_log.csv` + `loss_plots.png` under
`runs/`.

---

[한국어](README_running.ko.md) · [← README](README.md)
