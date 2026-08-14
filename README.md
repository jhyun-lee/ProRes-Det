# ProRes-Det

**Pro**jector **Res**toration + **Det**ection

A camera pointed at a screen sees whatever the projector throws on it. This framework
removes that light, then measures object detection before and after the removal.

The restoration model and the detector both sit behind small interfaces, so either can be
swapped without touching the pipeline.

![Framework: live and offline capture feed one frame pair into restoration, then detection, then the run recorder. Both model stages are interchangeable backends configured from YAML plus CLI flags.](Image/Framework.png)

Both modes converge on the same `(distorted, light)` frame pair, so everything downstream
is shared. `evaluate.py` runs the detector on both sides of the restoration step and scores
the difference — mAP, PSNR, SSIM.

| Guide | Covers |
|---|---|
| [README_running.md](README_running.md) | `demo.py` · `evaluate.py` · `train.py` — every option, input, output |
| [projector_distortion/README_code.md](projector_distortion/README_code.md) | What each module owns, the call graph, the public API |
| [data/README_data.md](data/README_data.md) | The three splits, layout, filename rules, both label formats, every `Data.py` stage |
| [weights/README_weights.md](weights/README_weights.md) | The three checkpoints, checkpoint format, the residual convention |

---

## 1. Install

Python ≥ 3.9.

```bash
git clone <repo> && cd ProRes-Det
conda create -n prores-det python=3.10 -y && conda activate prores-det

pip install -e ".[all]" -r requirements-cuda.txt    # GPU
pip install -e ".[all]"                             # CPU only
```

Quote the brackets — they are glob characters in zsh.

`[all]` pulls every extra. Narrower ones exist: `[yolo]` for the `yolo` detector,
`[train]` for `train.py`, `[live]` for `--live` on non-Windows, `[test]` for pytest.

Confirm the GPU took:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 2.9.0+cu128 True
```

`requirements-cuda.txt` is needed because the default PyPI `torch` carries no CUDA on
Windows. Every entry point prints the device it resolved as its first line, so a run that
fell back to the CPU says so.

---

## 2. Run

```bash
python demo.py                    # restore + detect 22 bundled pairs → output/<timestamp>/
python evaluate.py                # before/after mAP + PSNR/SSIM     → output/Eval_<dataset>/
python train.py --epochs 30       # retrain the restorer             → runs/<tag>/
python demo.py --live             # webcam + projector rig           → output/<timestamp>/
python Data.py capture_warp       # collect your own data (see below) → data/Create_Data/warp_<MMDD>/
python Data.py record --screen 2  # project + record, no models      → data/recordings/
```

The first three need no arguments. Sample data and all three checkpoints are tracked in
git, so a clone runs as-is. An editable install also puts `pdf-demo`, `pdf-evaluate` and
`pdf-train` on the PATH; they are thin wrappers around the same three scripts.

The bundled dataset is split three ways under `data/SampleData/`, and each entry point
defaults to its own split:

| Split | Contents | Default for |
|---|---|---|
| `sample_train` | 1,000 pairs, 20 scenes, no labels | `train.py` |
| `sample_eval` | 22 pairs, 10 scenes, 107 boxes | `demo.py`, `evaluate.py` |
| `sample_test` | 200 pairs, 5 scenes, 53 boxes | held out — pass `--input`/`--gt` to reach it |

The scenes do not overlap, so a model trained on `sample_train` is scored on frames it has
never seen. Details and the two label formats: [data/README_data.md](data/README_data.md).

**Run from the repo root.** Config paths resolve against the project root, but the
`train.data` globs are read against the working directory.

Swapping a model leaves every path unchanged — it is a one-line edit in
[configs/detection.yaml](projector_distortion/configs/detection.yaml), not a flag:

```yaml
detector:
  backend: ssd      # no ultralytics needed
  # backend: none   # restoration only
```

```bash
python demo.py --limit 5          # first 5 pairs
```

`--live` needs a webcam and a projector. Which monitor it projects on, which webcam
watches, and the projector→camera latency come from
[configs/live.yaml](projector_distortion/configs/live.yaml) — set once per machine, not
per run. The monitor table prints at startup, so a wrong `rig.screen` costs one aborted
run to correct:

```
2 monitor(s) detected:
      screen 0 -> 2560x1440 at (0,0) (primary)  \\.\DISPLAY1
      screen 1 -> 1920x1080 at (2560,0)         \\.\DISPLAY2
```

```yaml
# projector_distortion/configs/live.yaml
rig:
  screen: 1
  camera: 0
  offset: 6      # projector → camera latency, in frames
```

Every option, input and output: [README_running.md](README_running.md).

Collecting your own data is `Data.py`, which dispatches to the scripts under `data/`:

```bash
python Data.py                          # list the stages
python Data.py check                    # monitors + webcam, run this first
python Data.py make_light               # video -> the light frames to project
python Data.py capture_warp --screen 2  # project, shoot and rectify: 10 scenes × 50 = 500 pairs
```

`capture` and `warp` can also be run apart, which is what leaves the geometry redoable
without the rig. Everything lands under `data/Create_Data/`, ready for `--input` and
`--data-root`. Stages, flags and the session layout:
[data/README_data.md](data/README_data.md).

![Pipeline: step 1 collects a surface shot and a projected frame, then rectifies the ROI into a distorted image; step 2 feeds the projected and distorted images to the 3-level U-Net, which predicts a residual to subtract; step 3 runs the detector on the distorted and the restored image and compares the two.](Image/Pipeline.png)

Step 1 is `Data.py`, step 2 is `train.py` and the restorer, step 3 is `demo.py` and
`evaluate.py`. The bundled sample data already carries step 1's output, so a clone starts
at step 2.

---

## 3. Layout

```
ProRes-Det/
├── demo.py                   restore → detect, end to end (offline / --live)
├── evaluate.py               detection before/after + restoration quality
├── train.py                  fine-tune the restoration model
├── Data.py                   dataset collection: one entry point for every data/ stage
├── setup.py  requirements.txt  requirements-cuda.txt  LICENSE
│
├── projector_distortion/     the library → README_code.md
│   ├── configs/              default settings (YAML)
│   ├── models/               restoration + detection models
│   ├── pipeline/             offline / live run loops
│   └── utils/                image, visualization, recording and display helpers
│
├── tests/                    pytest suite, 118 tests
├── weights/                  3 checkpoints, 51 MiB, tracked in git
├── data/                     SampleData/ (train · eval · test), live/ clips, and the
│                             collection scripts Data.py drives; 349 MiB tracked
├── Image/                    the two figures above
└── output/                   run artefacts (git-ignored)
```

What each module owns and how they call each other:
[projector_distortion/README_code.md](projector_distortion/README_code.md).

---

## 4. Configuration

```
configs/*.yaml  <  CLI flags
```

| File | Holds |
|---|---|
| [configs/restoration.yaml](projector_distortion/configs/restoration.yaml) | Restoration backend and weights, input size, structural toggles, training hyperparameters, loss weights, training data paths |
| [configs/detection.yaml](projector_distortion/configs/detection.yaml) | Backend, per-backend weights, conf threshold, box size filter, the 17 class names — which is also what LabelMe class *names* are matched against |
| [configs/live.yaml](projector_distortion/configs/live.yaml) | The `--live` rig: monitor, webcam, camera backend, projector→camera latency, whether to review the calibration |
| [configs/collect.yaml](projector_distortion/configs/collect.yaml) | Everything `Data.py` reads: session folders, light extraction, the capture rig, warp geometry, recording |

These files are where a model setting is changed; none of them has a CLI flag. The
flags that remain are about *this* run — which data, where the output goes, how much
of it to process — plus `--device` on `demo.py` and the three training
hyperparameters on `train.py`. Everything about the models is edited in place:

```yaml
# projector_distortion/configs/detection.yaml
detector:
  backend: ssd
  conf: 0.4
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

That is the whole change. `detector.backend: mydet` in
[configs/detection.yaml](projector_distortion/configs/detection.yaml) works
immediately, and the pipeline, recording and evaluation code stay exactly as they were.

### Add a restorer

The same shape, one registry along:

```python
from projector_distortion.models import BaseRestorer, register_restorer

@register_restorer("myrest")
class MyRestorer(BaseRestorer):
    name = "myrest"

    def __init__(self, weights, device="cpu", input_size=(640, 360), **_):
        self.input_size = tuple(input_size)
        self.net = load_my_model(weights)

    def restore(self, distorted_bgr, light_bgr):
        restored = self.net(distorted_bgr)            # `light` is offered, not required
        return restored, residual_or_zeros
```

Point the config at it and it runs:

```yaml
# projector_distortion/configs/restoration.yaml
model:
  backend: myrest
  weights: path/to.pt
```

Both the backend name and the checkpoint come from the config; there is no flag for
either.

`light` — the frame the projector emitted at that moment — is passed to every restorer
because it is the one signal a ProCam rig has and an ordinary restoration setting does
not. A single-image backend simply ignores it.

The second return value is the residual view. A backend that predicts the surface image
directly returns zeros there; only the residual panel tile and `residual_mean` lose their
meaning, and everything else scores the same.

The shipped network predicts the residual to subtract, not the surface image:

```
input   (B, 6, H, W) = cat([distorted, light])  in [-1, 1]
output  (B, 3, H, W) = residual
restored = (distorted - residual).clamp(-1, 1)
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
