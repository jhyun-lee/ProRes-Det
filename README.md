# ProRes-Det

**Pro**jector **Res**toration + **Det**ection

Point a camera at a screen while a projector is throwing light on it and the original
content is wrecked. This framework removes the projected light (restore) and
measures object detection before and after that restoration.

Both the restoration model and the detector are swappable behind two small
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

> Square brackets are glob characters in some shells (zsh). Quoting makes it work
> in bash / zsh / PowerShell / cmd alike.

Base dependencies are `torch torchvision opencv-python numpy PyYAML tqdm`.

Optional dependencies are imported inside the function that uses them, not at the
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
├── docs/                     README_running.md — per-script options, in/out
├── tests/                    pytest suite
├── weights/                  3 checkpoints, not in git → README_weights.md
├── data/                     sample dataset → README_data.md
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
| [models/base.py](projector_distortion/models/base.py) | `BaseRestorer` · `BaseDetector` · `Detection` · detector registry — the extension point |
| [models/restoration.py](projector_distortion/models/restoration.py) | Restoration network (3-level U-Net), structural config, checkpoint I/O, pipeline wrapper |
| [models/detection.py](projector_distortion/models/detection.py) | yolo (ultralytics) and ssd (torchvision) wrappers, box size filter |
| [pipeline/offline.py](projector_distortion/pipeline/offline.py) | Batch over image pairs on disk. No hardware needed |
| [pipeline/live.py](projector_distortion/pipeline/live.py) | Webcam + projector rig: monitor placement, calibration, warping, worker thread |
| [utils/image.py](projector_distortion/utils/image.py) | BGR ↔ tensor conversion, resize, PSNR / SSIM / IoU |
| [utils/visualize.py](projector_distortion/utils/visualize.py) | Box drawing, 2×2 comparison panel, calibration overlay |
| [utils/recording.py](projector_distortion/utils/recording.py) | `RunRecorder` — owns everything written to the output directory |

### Bundled dataset

The sample data below is tracked in git, so it is there right after a clone.
**The checkpoints under `weights/` are not** — `.gitignore` excludes `*.pt` / `*.pth`.
Supply them before the first run, see [weights/README_weights.md](weights/README_weights.md).

| Path | Contents | Used as |
|---|---|---|
| [data/sample_input/pro/](data/sample_input/pro) | 22 camera captures of the projected screen | model input ch 0:3 |
| [data/sample_input/beam/](data/sample_input/beam) | 22 frames the projector emitted | model input ch 3:6 |
| [data/sample_gt/clean/](data/sample_gt/clean) | 10 screens with no projection | training target / PSNR·SSIM reference |
| [data/sample_gt/labels/](data/sample_gt/labels) | 10 YOLO-format detection labels | mAP reference |
| [data/live/BeamVideo.mp4](data/live/BeamVideo.mp4) | Clip to play through the projector (3.3 min) | `--live` input |
| [data/live/BaseBackGround.jpg](data/live/BaseBackGround.jpg) | Background shown during calibration | `--live` input |

The filename convention that pairs `pro` ↔ `beam` ↔ `clean`, and how to swap in real
data, are in [data/README_data.md](data/README_data.md); checkpoint details are in
[weights/README_weights.md](weights/README_weights.md).

---

## 3. Running

```bash
python demo.py                    # restore + detect 22 bundled pairs → output/<timestamp>/
python evaluate.py                # before/after mAP + PSNR/SSIM table → output/eval/
python train.py --epochs 30       # retrain the restorer            → runs/<tag>/
python demo.py --live --screen 2  # webcam + projector rig          → output/<timestamp>/
```

The first two need no arguments once the checkpoints are in `weights/`.

`--live` needs real hardware: a webcam pointed at a screen the projector is throwing to.
`--screen N` is the index of the monitor the projector is connected to — the
framework opens a borderless fullscreen window there and plays the clip through it,
while the webcam records the result. `0` is always the primary display, so a projector
attached as a second display is usually `1` or `2`. If you do not know it, pass anything:
the detected monitor table is printed before anything else.

```
2 monitor(s) detected:
      --screen 0 -> 2560x1440 at (0,0) (primary)  \\.\DISPLAY1
      --screen 1 -> 1920x1080 at (2560,0)         \\.\DISPLAY2
```

Swapping the model leaves input and output paths unchanged:

```bash
python demo.py --detector ssd     # swap the detector (no ultralytics needed)
python demo.py --detector none    # restoration only, skip detection
python demo.py --limit 5          # first 5 pairs only
```

Input, output and the full option list for each script → [docs/README_running.md](docs/README_running.md).

---

## 4. YAML configuration

```
configs/*.yaml  <  YAML passed via --restoration-config / --detection-config  <  CLI flags
```

| File | Holds |
|---|---|
| [configs/restoration.yaml](projector_distortion/configs/restoration.yaml) | Restoration weights path, input size, structural toggles, training hyperparameters and loss weights |
| [configs/detection.yaml](projector_distortion/configs/detection.yaml) | Detection backend, per-backend weights paths, conf threshold, box size filter, the 17 class names |

Relative paths inside a config resolve against the project root, not the working
directory, so `python demo.py` works from anywhere.

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

The shipped network predicts the residual to subtract, not the clean image:

```
input   (B, 6, H, W) = cat([pro, beam])  in [-1, 1]
output  (B, 3, H, W) = residual
restored = (pro - residual).clamp(-1, 1)
```

That convention keeps the original pixels intact by default and blocks the model from
hallucinating objects it memorised during training. Why it matters and how the loss
enforces it → [weights/README_weights.md](weights/README_weights.md#the-residual-convention).

---

## 6. License

MIT. See [LICENSE](LICENSE).

---

[한국어](README.ko.md)
