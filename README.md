# ProRes-Det

**Pro**jector **Res**toration + **Det**ection · [한국어](README.ko.md)

Point a camera at a screen while a projector is throwing light on it and the original
content is wrecked. This framework **removes the projected light (restore)** and
**measures object detection before and after** that restoration.

Both the restoration model and the detector are **swappable** behind two small
interfaces.

```
        camera capture (pro)  ─┐
                               ├─▶ restorer ─▶ residual ─▶ restored = pro − residual
   projected source (beam)  ──┘                              │
                                                             │
              detector(pro) ◀── before              after ──▶ detector(restored)
                     └────────── compare: mAP / PSNR / SSIM ──────────┘
```

---

## 1. Setup

Requires Python ≥ 3.9.

```bash
git clone <repo> && cd ProRes-Det

pip install -e "."              # base: ssd + none backends
pip install -e ".[yolo]"        # + ultralytics        → --detector yolo
pip install -e ".[train]"       # + pytorch-msssim, …  → train.py
pip install -e ".[live]"        # + screeninfo         → --live (not needed on Windows)
pip install -e ".[test]"        # + pytest
pip install -e ".[all]"         # everything
```

> Square brackets are glob characters in some shells (zsh). **Quoting makes it work
> in bash / zsh / PowerShell / cmd alike.**

Base dependencies are `torch torchvision opencv-python numpy PyYAML tqdm`.

Optional dependencies are imported **inside the function that uses them**, not at the
top of a module. `--detector ssd` works without `ultralytics` installed, and skipping
`--live` means `screeninfo` and the Win32 plumbing never load.

---

## 2. Layout

```
ProRes-Det/
├── demo.py                   restore → detect, end to end (offline / --live)
├── evaluate.py               detection before/after + restoration quality
├── train.py                  fine-tune the restoration model
├── setup.py  requirements.txt  LICENSE
│
├── projector_distortion/     ── the library
│   ├── __init__.py           public API
│   ├── cli.py                shared args, config precedence, console entry points
│   ├── config.py             YAML loading + path resolution
│   ├── data.py               sample discovery, label loading, training dataset
│   ├── configs/              default settings (YAML)
│   ├── models/               restoration + detection models
│   ├── pipeline/             offline / live run loops
│   └── utils/                image, visualization and recording helpers
│
├── tests/                    pytest suite
├── weights/                  3 checkpoints → weights/README.md
├── data/                     sample dataset → data/README.md
└── output/                   run artefacts
```

### What each module does

| File | Role |
|---|---|
| [demo.py](demo.py) | Entry point. Parse args → build models → run the offline/live pipeline → print a summary |
| [evaluate.py](evaluate.py) | Entry point. Score only samples that have GT: P/R/F1/mAP plus PSNR/SSIM, then write the report |
| [train.py](train.py) | Entry point. Training loop (L1 + perceptual + SSIM + wavelet loss) |
| [cli.py](projector_distortion/cli.py) | Args, config precedence and model construction shared by all three entry points |
| [config.py](projector_distortion/config.py) | YAML load/merge, path resolution against the project root |
| [data.py](projector_distortion/data.py) | Pairs pro/beam/clean/label by filename, and provides the training patch dataset |
| [models/base.py](projector_distortion/models/base.py) | `BaseRestorer` · `BaseDetector` · `Detection` · detector registry — **the extension point** |
| [models/restoration.py](projector_distortion/models/restoration.py) | Restoration network (3-level U-Net), structural config, checkpoint I/O, pipeline wrapper |
| [models/detection.py](projector_distortion/models/detection.py) | yolo (ultralytics) and ssd (torchvision) wrappers, box size filter |
| [pipeline/offline.py](projector_distortion/pipeline/offline.py) | Batch over image pairs on disk. No hardware needed |
| [pipeline/live.py](projector_distortion/pipeline/live.py) | Webcam + projector rig: monitor placement, calibration, warping, worker thread |
| [utils/image.py](projector_distortion/utils/image.py) | BGR ↔ tensor conversion, resize, PSNR / SSIM / IoU |
| [utils/visualize.py](projector_distortion/utils/visualize.py) | Box drawing, 2×2 comparison panel, calibration overlay |
| [utils/recording.py](projector_distortion/utils/recording.py) | `RunRecorder` — owns everything written to the output directory |

### Bundled dataset

Sample data ships with the repo so every command runs with no arguments.

| Path | Contents | Used as |
|---|---|---|
| [data/sample_input/pro/](data/sample_input/pro) | 22 camera captures of the projected screen | model input ch 0:3 |
| [data/sample_input/beam/](data/sample_input/beam) | 22 frames the projector emitted | model input ch 3:6 |
| [data/sample_gt/clean/](data/sample_gt/clean) | 10 screens with no projection | training target / PSNR·SSIM reference |
| [data/sample_gt/labels/](data/sample_gt/labels) | 10 YOLO-format detection labels | mAP reference |
| [data/live/BeamVideo.mp4](data/live/BeamVideo.mp4) | Clip to play through the projector (3.3 min) | `--live` input |
| [data/live/BaseBackGround.jpg](data/live/BaseBackGround.jpg) | Background shown during calibration | `--live` input |

The filename convention that pairs `pro` ↔ `beam` ↔ `clean`, and how to swap in real
data, are in [data/README.md](data/README.md); checkpoint details are in
[weights/README.md](weights/README.md).

---

## 3. Running

### 3.1 Quick start

```bash
python demo.py        # restore + detect 22 bundled pairs → output/<timestamp>/
python evaluate.py    # before/after mAP + PSNR/SSIM table → output/eval/
```

Runs with no arguments. The three below only change the model; input and output paths
are unchanged.

```bash
python demo.py --detector ssd     # swap the detector (no ultralytics needed)
python demo.py --detector none    # restoration only, skip detection
python demo.py --limit 5          # first 5 pairs only
```

### 3.2 `demo.py` — restore → detect

| | Default path | Option to change it |
|---|---|---|
| **Input** | `data/sample_input/` (`pro/` + `beam/`) | `--input <dir>` |
| **GT** (optional) | `data/sample_gt/` (`clean/` + `labels/`) | `--gt <dir>` |
| **Output** | `output/<timestamp>/` | `--output <dir>` · `--name <name>` |

Without GT only PSNR/SSIM is skipped; the run continues.

| Option | Meaning |
|---|---|
| `--detector yolo\|ssd\|none` | Detection backend (default yolo) |
| `--conf <float>` | Detector confidence floor (default 0.25) |
| `--limit N` | Cap how many pairs are processed (0 = all) |
| `--save-every N` | Image save interval. `0` keeps csv only |
| `--save-kinds a,b` | Save only selected image kinds (`captured,restored,panel`, …) |
| `--video` | Also write the 2×2 panels as `result.mp4` |

### 3.3 `evaluate.py` — score before vs after

Only samples that have **both** `clean` and `label` are scored.

| | Default path | Option to change it |
|---|---|---|
| **Input** | `data/sample_input/` (`pro/` + `beam/`) | `--input <dir>` |
| **GT** | `data/sample_gt/` (`clean/` + `labels/`) | `--gt <dir>` |
| **Output** | `output/eval/`<br>`report.json`, `per_class_<backend>.csv`, `per_image_<backend>.csv` | `--output <dir>` · `--name <name>` |

| Option | Meaning |
|---|---|
| `--detectors yolo,ssd` | Compare several backends in one run (one row each) |
| `--iou <float>` | IoU threshold for a true positive (default 0.5) |
| `--limit N` | Cap how many pairs are scored |

```bash
python evaluate.py --detectors yolo,ssd --iou 0.5
```

A summary table is printed; per-class P / R / F1 / AP goes into the csv.

```python
import pandas as pd
pc = pd.read_csv("output/eval/per_class_yolo.csv")
pc.pivot_table(index="name", columns="source", values="ap")   # per-class AP shift
```

### 3.4 `train.py` — retrain the restoration model

Only complete `pro` / `beam` / `clean` triplets are used.

| | Default path | Option to change it |
|---|---|---|
| **Input** | `data/sample_input/` (`pro/` + `beam/`) | `--data-root <dir>` |
| **Target** | `data/sample_gt/clean/` — **required** | `--gt <dir>` |
| **Output** | `runs/<MMDD_HHMM>_<epochs>ep_<tag>/`<br>`restorer_<tag>_best.pt`, `epoch_N.pt`, `loss_log.csv`, `loss_plots.png` | `--out <dir>` |

| Option | Meaning |
|---|---|
| `--epochs N` `--batch-size N` `--lr F` | Defaults come from `configs/restoration.yaml` |
| `--sample N` | Cap how many triplets are used |
| `--resume <ckpt>` | Continue from a checkpoint |
| `--no-ca` and 9 more | Ablation. Whatever is switched off lands in `tag` (`NoCA`, `NoCA-NoSkip3`, …) |

```bash
python train.py --epochs 30
python train.py --data-root /path/to/dataset --epochs 30
python train.py --no-ca --epochs 30
```

Checkpoints embed their own architecture config, so the flags never need repeating.

```bash
python demo.py --restorer-weights runs/0730_1948_30ep_NoCA/restorer_NoCA_best.pt
```

### 3.5 `demo.py --live` — webcam + projector

| | Default | Option to change it |
|---|---|---|
| **Projected clip** | `data/live/BeamVideo.mp4` | `--clip <path>` |
| **Calibration background** | `data/live/BaseBackGround.jpg` | `--background <path>` |
| **Camera** | webcam 0 | `--camera N` · `--cam-width/height/fps` · `--cam-backend` |
| **Output** | `output/<timestamp>/` + `calib/` + `result.mp4` | `--output <dir>` · `--name <name>` |

| Option | Meaning |
|---|---|
| `--screen N` | Monitor index the projector is attached to (0 = primary) |
| `--offset N` | Projector→camera latency in frames (default 6) |
| `--manual-calib` | Click the 4 corners instead of auto-detecting them |
| `--debug-view` | Show the pre-warp camera feed with the quad, live |
| `--max-frames N` | 0 = until the clip ends |

```bash
python demo.py --live --screen 2
python demo.py --live --screen 2 --save-every 30 --debug-view
```

If you do not know `--screen`, pass anything and run — the detected monitor table is
printed first.

Sequence: black/white flash → 4 corners auto-detected from the difference image →
homography computed once → every frame warped with it. Auto-detection falls back to
manual clicking on failure. If the result looks wrong, check `--debug-view` and the
intermediate images in `output/*/calib/` (`mask.jpg`, `diff.jpg`, `warped.jpg`).

### 3.6 Output format

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

**Keeping clean and annotated images separate** is the point. It is what makes
recomputing PSNR/SSIM and re-running a different detector possible after the fact.

`evaluate.py` writes `report.json` + `per_class_*.csv` + `per_image_*.csv` instead.

---

## 4. YAML configuration

```
configs/*.yaml  <  YAML passed via --restoration-config / --detection-config  <  CLI flags
```

| File | Holds |
|---|---|
| [configs/restoration.yaml](projector_distortion/configs/restoration.yaml) | Restoration weights path, input size, structural toggles, training hyperparameters and loss weights |
| [configs/detection.yaml](projector_distortion/configs/detection.yaml) | Detection backend, per-backend weights paths, conf threshold, box size filter, the 17 class names |

Relative paths inside a config resolve **against the project root, not the working
directory**, so `python demo.py` works from anywhere.

To override only part of it, pass a YAML with just the keys you want changed — it is
merged recursively.

```yaml
# my_det.yaml
detector:
  backend: ssd
  conf: 0.4
```

```bash
python demo.py --detection-config my_det.yaml
```

---

## 5. Swapping modules

### Add a detector

```python
from projector_distortion.models.base import BaseDetector, Detection, register_detector

@register_detector("mydet")
class MyDetector(BaseDetector):
    name = "mydet"

    def __init__(self, weights, class_names=None, conf=0.25, device="cpu", **_):
        super().__init__(class_names or [], conf, device)
        self.net = load_my_model(weights)

    def detect(self, bgr):
        return [Detection(cls_id, self.label_of(cls_id), score, (x1, y1, x2, y2))
                for cls_id, score, (x1, y1, x2, y2) in self.net(bgr)]
```

That is the whole change. `--detector mydet` works immediately, and the pipeline,
recording and evaluation code stay exactly as they were.

### Swap the restorer

Subclass `BaseRestorer` and implement
`restore(pro_bgr, beam_bgr) -> (restored_bgr, residual_bgr)`. That is the whole
interface.

```
input   (B, 6, H, W) = cat([pro, beam])  in [-1, 1]
output  (B, 3, H, W) = residual
restored = (pro - residual).clamp(-1, 1)
```

#### Why predict the residual instead of the clean image

The network never draws the restored image. It only emits the light to subtract
from `pro`.

**1) Preserving the original becomes the default.**
Where no projector light lands, residual ≈ 0 is enough and the `pro` pixel passes
through untouched. "Do nothing" is the identity, so the network only has to learn
**what must change**. Regressing clean directly forces it to regenerate perfectly good
background too, which smears regions that should have been left alone.

**2) It blocks the "paint the objects in" overfit.**
If the network outputs clean directly, the fastest way to drive the loss down is to
largely ignore the input and **reproduce a screen memorised from the training set**.
This dataset makes that especially tempting: 10 clean images back 22 `pro` captures,
so clean repeats and memorising the target per `oriId` pays off. A model trained that
way **makes the detector see objects that were never there**, which destroys the whole
point of the evaluation. The `restored = pro − residual` structure forces the output to
**always derive from real camera pixels**, closing that shortcut.

**3) Values cannot blow up.**
The output `tanh` bounds residual to [-1, 1], and `clamp(-1, 1)` bounds the result
after the subtraction — two layers of range control. (`--no-tanh` removes the first one;
that is one of the ablation switches.)

#### How it is enforced — the loss is on `restored`, not on the residual

The key is that **the subtraction lives inside the graph**. The residual is never given
a target of its own; only the subtracted result is compared against clean. What to
subtract is left for the network to discover.

```python
residual = net(torch.cat([pro, beam], dim=1))     # network output
restored = (pro - residual).clamp(-1, 1)          # subtraction inside the graph
loss = (0.93 * L1(restored, clean)
      + 2.04 * Perceptual(restored, clean)
      + 0.53 * (1 - SSIM(restored, clean))
      + 0.90 * WaveletHF(restored, clean))        # all four measure `restored`
```

| Loss term | What it measures | What it penalises |
|---|---|---|
| `L1` | Absolute pixel error | Global colour / brightness drift |
| `Perceptual` (VGG19 relu3_3) | Feature-map distance | Pixel-close results whose structure is broken |
| `1 − SSIM` | Local luminance, contrast, structure | Flat output that only matched the mean |
| `WaveletHF` (Haar LH/HL/HH, **LL excluded**) | Edges and texture only | Blurring everything to lower the loss |

Dropping the low-frequency (LL) band is the point of `WaveletHF`. Blurring the whole
image still lowers L1, but not the high-frequency term. That is what makes the residual
follow the actual boundaries of the projected light.

The weights live under `train.loss` in
[configs/restoration.yaml](projector_distortion/configs/restoration.yaml) and come from
an Optuna sweep on this dataset. Implementation: [train.py](train.py).

> The interface itself will accept a restorer that emits clean directly. In that case
> the `residual` visualisation and the `residual_mean` metric lose their meaning, and
> both advantages above are gone.

---

## 6. License

MIT. See [LICENSE](LICENSE).
