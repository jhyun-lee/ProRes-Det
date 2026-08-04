# ProRes-Det

**Pro**jector **Res**toration + **Det**ection

A camera pointed at a screen sees whatever the projector throws on it. This framework
removes that light, then measures object detection before and after the removal.

The restoration model and the detector both sit behind small interfaces, so either can be
swapped without touching the pipeline.

```
        camera capture (pro)  ─┐
                               ├─▶ restorer ─▶ residual ─▶ restored = pro − residual
   projected source (beam)  ──┘                              │
                                                             │
              detector(pro) ◀── before              after ──▶ detector(restored)
                     └────────── compare: mAP / PSNR / SSIM ──────────┘
```

| Guide | Covers |
|---|---|
| [README_running.md](README_running.md) | `demo.py` · `evaluate.py` · `train.py` — every option, input, output |
| [data/README_data.md](data/README_data.md) | Dataset layout, filename rules, labels, `collect.py`, `record.py` |
| [weights/README_weights.md](weights/README_weights.md) | The three checkpoints, checkpoint format, the residual convention |

---

## 1. Install

Python ≥ 3.9.

```bash
git clone <repo> && cd ProRes-Det

conda create -n prores-det python=3.10 -y
conda activate prores-det

pip install -e "."              # base: ssd + none backends
pip install -e ".[yolo]"        # + ultralytics       → --detector yolo
pip install -e ".[train]"       # + pytorch-msssim    → train.py
pip install -e ".[live]"        # + screeninfo        → --live (not needed on Windows)
pip install -e ".[test]"        # + pytest
pip install -e ".[all]"         # everything
```

Base dependencies: `torch torchvision opencv-python numpy PyYAML tqdm`.

Quote the brackets — they are glob characters in zsh.

Optional dependencies are imported inside the function that needs them. `--detector ssd`
runs without `ultralytics`, and skipping `--live` never loads `screeninfo`.

### GPU

**On Windows the commands above install a CPU-only torch.** Nothing fails. Restoration
just takes ~380 ms a frame instead of ~13. Add `requirements-cuda.txt` for the CUDA build:

```bash
pip install -e ".[all]" -r requirements-cuda.txt
```

An extra cannot do this alone: a package has no way to name the index it wants to be
fetched from. So the index line and the version pins live in that file. `cu128` covers
RTX 50-series and everything older a 12.8 driver supports. For an older driver, swap
`cu128` for `cu121` or `cu118` inside the file.

Check what you got:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 2.9.0+cu128 True      <- good
# 2.9.0+cpu   False     <- CPU-only; rerun with -r requirements-cuda.txt
```

Every entry point prints the device it resolved as its first line, so a run that quietly
fell back says so:

```
device: cuda - NVIDIA GeForce RTX 5090, torch 2.9.0+cu128
device: cpu - torch 2.9.0+cpu is a CPU-only build, so restoration runs ~25x slower.
```

`--fp16` applies only on CUDA. On CPU it is ignored.

---

## 2. Run

```bash
python demo.py                    # restore + detect 22 bundled pairs → output/<timestamp>/
python evaluate.py                # before/after mAP + PSNR/SSIM     → output/Eval_<dataset>/
python train.py --epochs 30       # retrain the restorer             → runs/<tag>/
python demo.py --live --screen 2  # webcam + projector rig           → output/<timestamp>/
python data/collect.py capture    # collect your own data (4 stages) → data/collected_<MMDD>/
python data/record.py --screen 2  # project + record, no models      → data/recordings/
```

The first two need no arguments. Sample data and all three checkpoints are tracked in git,
so a clone runs as-is.

Run from the repo root. Config paths resolve against the project root, but the
`train.data` globs are read against the working directory.

Swapping a model leaves every path unchanged:

```bash
python demo.py --detector ssd     # no ultralytics needed
python demo.py --detector none    # restoration only
python demo.py --limit 5          # first 5 pairs
```

`--live` needs hardware: a webcam pointed at a screen a projector is throwing to.
`--screen N` is the projector's monitor index. `0` is always the primary display, so a
projector attached as a second display is usually `1` or `2`. Pass anything if you do not
know it — the monitor table is printed before anything else.

```
2 monitor(s) detected:
      --screen 0 -> 2560x1440 at (0,0) (primary)  \\.\DISPLAY1
      --screen 1 -> 1920x1080 at (2560,0)         \\.\DISPLAY2
```

Every option, input and output: [README_running.md](README_running.md).

---

## 3. Layout

```
ProRes-Det/
├── demo.py                   restore → detect, end to end (offline / --live)
├── evaluate.py               detection before/after + restoration quality
├── train.py                  fine-tune the restoration model
├── setup.py  requirements.txt  requirements-cuda.txt  LICENSE
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
├── tests/                    pytest suite, 102 tests
├── weights/                  3 checkpoints, 51 MiB, tracked in git
├── data/                     sample dataset + collect.py / record.py, 81 MiB tracked
└── output/                   run artefacts (git-ignored)
```

| File | Role |
|---|---|
| [demo.py](demo.py) | Entry point. Args → models → offline/live pipeline → summary |
| [evaluate.py](evaluate.py) | Entry point. Scores the samples that have GT: P/R/F1/mAP + PSNR/SSIM |
| [train.py](train.py) | Entry point. Training loop (L1 + perceptual + SSIM + wavelet) |
| [data/collect.py](data/collect.py) | Entry point. Dataset collection with the rig, in 4 stages |
| [data/record.py](data/record.py) | Entry point. Project a clip, record the camera. Loads no weights |
| [cli.py](projector_distortion/cli.py) | Args, config precedence, model construction — shared by all three |
| [config.py](projector_distortion/config.py) | YAML load/merge, path resolution against the project root |
| [data.py](projector_distortion/data.py) | Pairs pro/beam/clean/label by filename; the training patch dataset |
| [models/base.py](projector_distortion/models/base.py) | `BaseRestorer` · `BaseDetector` · `Detection` · detector registry |
| [models/restoration.py](projector_distortion/models/restoration.py) | Restoration network, structural config, checkpoint I/O |
| [models/detection.py](projector_distortion/models/detection.py) | yolo and ssd wrappers, box size filter |
| [pipeline/offline.py](projector_distortion/pipeline/offline.py) | Batch over image pairs on disk. No hardware |
| [pipeline/live.py](projector_distortion/pipeline/live.py) | Webcam + projector: monitor placement, calibration, warp, worker thread |
| [utils/image.py](projector_distortion/utils/image.py) | BGR ↔ tensor, resize, PSNR / SSIM / IoU |
| [utils/visualize.py](projector_distortion/utils/visualize.py) | Box drawing, 2×2 panel, calibration overlay |
| [utils/recording.py](projector_distortion/utils/recording.py) | `RunRecorder` — owns everything written to the output directory |

`setup.py` also installs `pdf-demo`, `pdf-evaluate` and `pdf-train`. Each delegates to the
matching root script, so they need an editable install of a checkout.

---

## 4. Configuration

```
configs/*.yaml  <  YAML via --restoration-config / --detection-config  <  CLI flags
```

| File | Holds |
|---|---|
| [configs/restoration.yaml](projector_distortion/configs/restoration.yaml) | Restoration weights, input size, structural toggles, training hyperparameters, loss weights, training data paths |
| [configs/detection.yaml](projector_distortion/configs/detection.yaml) | Backend, per-backend weights, conf threshold, box size filter, the 17 class names |

To override part of a config, pass a YAML with only the keys you want changed. It is
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

Relative paths inside a config resolve against the project root — except the three
`train.data` entries, which are globbed as written and so read against the working
directory.

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
`restore(pro_bgr, beam_bgr) -> (restored_bgr, residual_bgr)`. That is the whole interface.

The shipped network predicts the residual to subtract, not the clean image:

```
input   (B, 6, H, W) = cat([pro, beam])  in [-1, 1]
output  (B, 3, H, W) = residual
restored = (pro - residual).clamp(-1, 1)
```

Why that convention matters, and how the loss enforces it:
[weights/README_weights.md](weights/README_weights.md#the-residual-convention).

---

## 6. Tests

```bash
python -m pytest -q
```

---

## 7. License

MIT. See [LICENSE](LICENSE).

---

[한국어](README.ko.md)
