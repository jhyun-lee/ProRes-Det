# ProRes-Det

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22153821-1682D4)](https://doi.org/10.5281/zenodo.22153821)

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
| [COMMANDS.md](COMMANDS.md) | Every command and flag in one lookup table, plus the interactive keys |
| [README_running.md](README_running.md) | `demo.py` · `evaluate.py` · `train.py` — every option, input, output |
| [projector_distortion/README_code.md](src/projector_distortion/README_code.md) | What each module owns, the call graph, the public API |
| [datasets/README_data.md](datasets/README_data.md) | The three splits, layout, filename rules, both label formats, every `Data.py` stage |
| [weights/README_weights.md](weights/README_weights.md) | The four checkpoints, model selection, checkpoint format, the residual convention |

---

## 1. Install

Python ≥ 3.9.

```bash
git clone https://github.com/jhyun-lee/ProRes-Det.git
cd ProRes-Det
conda create -n prores-det python=3.10 -y && conda activate prores-det

# Default installation (SSD backend; permissive dependencies, no Ultralytics)
pip install -e .                           # Default installation
pip install -e . -r requirements-cuda.txt  # Windows CUDA PyTorch wheel

# Optional: YOLO11 backend (opts into AGPL-3.0)
pip install -e ".[yolo]"

# All extras (Warning: installs all dependencies including AGPL-3.0 Ultralytics)
pip install -e ".[all]" -r requirements-cuda.txt
```

Quote the brackets (`".[yolo]"`, `".[all]"`) — they are glob characters in zsh.

> **License Notice**: The default installation (`pip install -e .`) touches only permissive licenses (MIT, BSD, Apache) and runs the BSD-3-Clause `ssd` detector. `ultralytics` (YOLO11) is licensed under **AGPL-3.0** and is installed only when explicitly opting in via `.[yolo]` or `.[all]`. See [THIRD_PARTY.md](THIRD_PARTY.md) for full licensing boundaries.

Narrower extras exist: `[yolo]` for the YOLO detector (AGPL-3.0), `[train]` for `train.py`,
`[live]` for `--live` on non-Windows, `[test]` for pytest.

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
python src/demo.py                    # restore + detect 200 bundled pairs → output/<timestamp>/
python src/evaluate.py                # before/after mAP + PSNR/SSIM     → output/Eval_<dataset>/
python src/train.py --epochs 30       # retrain the restorer             → runs/<tag>/
python src/demo.py --live             # webcam + projector rig           → output/<timestamp>/
python src/Data.py capture_warp       # collect your own data (see below) → datasets/Create_Data/warp_<MMDD>/
python src/Data.py record --screen 2  # project + record, no models      → datasets/recordings/
python src/demo.py --video-pair datasets/recordings/rec_<t>.mp4   # replay that recording → result.mp4
```

The first three need no arguments. Sample data is under `datasets/SampleData/`, and
`weights/` contains two restoration and two detection checkpoints. An editable install
also puts `pdf-demo`, `pdf-evaluate` and `pdf-train` on the PATH; they are thin wrappers
around the same three scripts.

The bundled dataset is split three ways under `datasets/SampleData/`, and each entry point
defaults to its own split:

| Split | Contents | Default for |
|---|---|---|
| `sample_train` | 3,000 pairs, 20 scenes, no labels | `train.py` |
| `sample_eval` | 200 pairs, 10 scenes, 2,140 GT instances | `demo.py`, `evaluate.py` |
| `sample_test` | 1,000 pairs, 10 scenes, 10,468 GT instances | held out — pass `--input`/`--gt` to reach it |

The scenes do not overlap, so a model trained on `sample_train` is scored on frames it has
never seen. Details and the two label formats: [datasets/README_data.md](datasets/README_data.md).

**Run from the repo root.** Config paths resolve against the project root, but the
`train.data` globs are read against the working directory.

Swapping a model leaves every path unchanged — it is a one-line edit in
[configs/detection.yaml](src/projector_distortion/configs/detection.yaml), not a flag:

```yaml
detector:
  backend: ssd      # no ultralytics needed
  # backend: none   # restoration only
```

```bash
python src/demo.py --limit 5          # first 5 pairs
```

`--live` needs a webcam and a projector. Which monitor it projects on, which webcam
watches, and the projector→camera latency are properties of the machine, so they live
in [configs/live.yaml](src/projector_distortion/configs/live.yaml) rather than in
flags; `--screen` and `--camera` override the first two for one run. The monitor table
prints at startup, so a wrong guess costs one aborted run rather than a file edit.

The rig and the models do not have to be on the same machine, or in the same week.
`Data.py record` projects a clip and records the camera's view as one raw mp4 with no
models loaded; `demo.py --video-pair` turns that file back into a run — rectifying it,
pairing every camera frame with the clip frame that caused it, and writing the restored
video. One argument is enough, because the `.json` written beside the mp4 already names
the clip and the rate the camera actually managed:

```bash
python src/Data.py record --screen 2 --seconds 30               # on the rig
python src/demo.py --video-pair datasets/recordings/rec_<t>.mp4 # anywhere, later
```

<p align="center">
  <img src="Image/BeamPro.jpg" width="240" alt="The capture rig: a short-throw projector overhead, a camera on the same arm, and the objects laid out on the table under the projected pattern — the (distorted, light) pairs come from exactly this setup.">
</p>

Collecting your own data is `Data.py`, which dispatches to the stage scripts under
`datasets/` — `check`, `make_light`, `capture`, `warp`, `record`. Everything lands
under `datasets/Create_Data/`, ready for `--input` and `--data-root`:
[datasets/README_data.md](datasets/README_data.md).

Every flag and its default: [COMMANDS.md](COMMANDS.md). Why each entry point behaves
the way it does: [README_running.md](README_running.md).

![Pipeline: step 1 collects a surface shot and a projected frame, then rectifies the ROI into a distorted image; step 2 feeds the projected and distorted images to the 3-level U-Net, which predicts a residual to subtract; step 3 runs the detector on the distorted and the restored image and compares the two.](Image/Pipeline.png)

Step 1 is `Data.py`, step 2 is `train.py` and the restorer, step 3 is `demo.py` and
`evaluate.py`. The bundled sample data already carries step 1's output, so a clone starts
at step 2.

---

## 3. Reference results

These results use the current `datasets/SampleData` splits and the two restoration
checkpoints listed in [weights/README_weights.md](weights/README_weights.md).
These tables replace the historical results from the smaller dataset release.

Both restorers were trained for 30 epochs on the 3,000 `sample_train` pairs.
After each epoch, all 200 `sample_eval` pairs were restored without gradients;
the checkpoint with the highest mean validation PSNR was selected: epoch 28 for
NAFSE-UNet and epoch 17 for Basic CNN. `sample_test` was reserved for final scoring
and has **5× as many pairs** as `sample_eval`.

### Detection, before vs after restoration

D means distorted input; R means restored output. Both passes use the same detector
weights, confidence threshold of 0.25, and box-size gate (minimum width/height
20 pixels, minimum area 500 pixels²), with restoration input size 640×360.
mAP@0.5 and mAP@0.9 are VOC-style interpolated PR areas at separate IoU thresholds,
not COCO's IoU-averaged mAP. F1 is reported at IoU 0.5.

**`sample_test` — 1,000 pairs, 10 scenes, 10,468 GT instances**

| Restorer | Detector | mAP@0.5 (D → R) | mAP@0.9 (D → R) | F1@0.5 (D → R) |
|---|---|---|---|---|
| NAFSE-UNet | SSDLite | 0.7101 → **0.9303** | 0.2977 → **0.4867** | 0.6257 → **0.8586** |
| NAFSE-UNet | YOLO11 | 0.7446 → **0.9239** | 0.3694 → **0.5457** | 0.7787 → **0.9152** |
| Basic CNN | SSDLite | 0.7101 → **0.8016** | 0.2977 → **0.3392** | 0.6257 → **0.7107** |
| Basic CNN | YOLO11 | 0.7446 → **0.8122** | 0.3694 → **0.4368** | 0.7787 → **0.8218** |

**`sample_eval` — 200 pairs, 10 scenes, 2,140 GT instances**

| Restorer | Detector | mAP@0.5 (D → R) | mAP@0.9 (D → R) | F1@0.5 (D → R) |
|---|---|---|---|---|
| NAFSE-UNet | SSDLite | 0.8720 → **0.9889** | 0.5264 → **0.7669** | 0.7873 → **0.9668** |
| NAFSE-UNet | YOLO11 | 0.9010 → **0.9859** | 0.7420 → **0.8668** | 0.8918 → **0.9784** |
| Basic CNN | SSDLite | 0.8720 → **0.9123** | 0.5264 → **0.5649** | 0.7873 → **0.8275** |
| Basic CNN | YOLO11 | 0.9010 → **0.9269** | 0.7420 → **0.7680** | 0.8918 → **0.9113** |

For NAFSE-UNet on `sample_test`, SSDLite gains 0.2202 mAP@0.5 and YOLO11 gains
0.1793. YOLO11 false negatives at IoU 0.5 decrease from 2,634 to 750.
The two restorers have different capacities (4,184,259 vs 30,371 parameters);
this compares complete backends under the same training budget, rather than
isolating the effect of an individual architectural component.

### Restoration quality

PSNR and SSIM are computed against the projection-free surface, then averaged over
the split. These metrics do not depend on the detector.

| Split | Restorer | Mean PSNR (D → R), dB | Mean SSIM (D → R) |
|---|---|---|---|
| `sample_test` | NAFSE-UNet | 16.6488 → **21.9406** | 0.6928 → **0.8590** |
| `sample_test` | Basic CNN | 16.6488 → **18.8042** | 0.6928 → **0.7860** |
| `sample_eval` | NAFSE-UNet | 16.3458 → **23.5281** | 0.7667 → **0.8844** |
| `sample_eval` | Basic CNN | 16.3458 → **19.3022** | 0.7667 → **0.8192** |

### Reproducing

The shipped defaults are NAFSE-UNet restoration and SSDLite detection. Run from the
project root. These commands reproduce the default restorer's SSD rows at both IoUs
without overwriting the reports from the other threshold:

```bash
for split in sample_eval sample_test; do
  for iou in 0.5 0.9; do
    tag="${iou/./}"
    python src/evaluate.py \
      --input "datasets/SampleData/$split" \
      --gt "datasets/SampleData/$split" \
      --iou "$iou" \
      --output "output/reproduce_naf_se_unet/iou$tag"
  done
done
```

Each output directory contains `Eval_sample_eval/report.json` or
`Eval_sample_test/report.json`, plus per-image and per-class CSVs.

To include both detector rows, install the optional YOLO dependencies with
`pip install -e ".[yolo]"`, then set `detector.backend: [ssd, yolo]` in
[detection.yaml](src/projector_distortion/configs/detection.yaml) and rerun.
This opts into the YOLO license described in [License](#8-license).
The default scalar `ssd` runs only SSDLite; `demo.py` uses only the first entry
when a detector list is configured.

To reproduce the Basic CNN rows, replace the `model` selection in
[restoration.yaml](src/projector_distortion/configs/restoration.yaml) with:

```yaml
model:
  backend: plain_cnn
  weights: weights/restorer_CNN_best.pt
  input_size: [640, 360]
```

Keep `plain_cnn.width: 32` and `plain_cnn.depth: 5`, then rerun the loop with the
output prefix changed to `output/reproduce_plain_cnn`. Both `model.backend` and
`model.weights` must match; changing the output directory does not select a model.
To return to the default, use `backend: naf_se_unet` and
`weights: weights/restorer_unet_best.pt`.

The reference reports were produced with CUDA. Numerical differences between
devices and library versions may affect the last printed digits. The detector's
historical training-pool disclosure is documented in
[weights/README_weights.md](weights/README_weights.md#detector-training-provenance).

---

## 4. Layout

```
ProRes-Det/
├── README.md  LICENSE.txt  COMMANDS.md  README_running.md
├── setup.py  requirements.txt  requirements-cuda.txt
│
├── src/                      all source lives here
│   ├── demo.py               restore → detect, end to end (offline / --live / --video-pair)
│   ├── evaluate.py           detection before/after + restoration quality
│   ├── train.py              fine-tune the restoration model
│   ├── Data.py               dataset collection: one entry point for every stage
│   ├── projector_distortion/ the library → README_code.md
│   │   ├── configs/          default settings (YAML)
│   │   ├── models/           restoration + detection models
│   │   ├── pipeline/         offline / live / replay run loops
│   │   └── utils/            image, visualization, recording and display helpers
│   └── tests/                pytest suite, 151 tests
│
├── datasets/                 SampleData/ (train · eval · test), live/ clips, and the
│                             collection scripts Data.py drives
├── weights/                  4 checkpoints: 2 restorers and 2 detectors
├── Image/                    the figures above
└── output/                   run artefacts (git-ignored)
```

What each module owns and how they call each other:
[projector_distortion/README_code.md](src/projector_distortion/README_code.md).

---

## 5. Configuration

```
configs/*.yaml  <  CLI flags
```

| File | Holds |
|---|---|
| [restoration.yaml](src/projector_distortion/configs/restoration.yaml) | Restoration backend, weights, input size, the ablation toggles, training hyperparameters and loss weights |
| [detection.yaml](src/projector_distortion/configs/detection.yaml) | Detector backend, per-backend weights, confidence floor, box size gate, the 17 class names |
| [live.yaml](src/projector_distortion/configs/live.yaml) | The `--live` rig: monitor, webcam, projector→camera latency, calibration review |
| [collect.yaml](src/projector_distortion/configs/collect.yaml) | Everything `Data.py` reads: session folders, light extraction, capture rig, warp geometry, recording |

No model setting has a CLI flag. What remains on the command line is about *this* run —
which data, where the output goes, how much to process — so a checkpoint or a threshold
is edited in place rather than remembered as an argument:

```yaml
# projector_distortion/configs/detection.yaml
detector:
  backend: ssd
  conf: 0.4
```

Relative paths inside a config resolve against the project root — except the three
`train.data` entries, which are globbed as written and so read against the working
directory. Setting-by-setting:
[COMMANDS.md](COMMANDS.md#settings-that-have-no-flag).

---

## 6. Swapping modules

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
[configs/detection.yaml](src/projector_distortion/configs/detection.yaml) works
immediately, and the pipeline, recording and evaluation code stay exactly as they were.

### Add a restorer

The same shape, one registry along:

```python
from projector_distortion.models import BaseRestorer, register_restorer

@register_restorer("myrest")
class MyRestorer(BaseRestorer):
    name = "myrest"

    def restore(self, distorted_bgr, light_bgr):
        return self.net(distorted_bgr), residual_or_zeros
```

`model.backend: myrest` in
[configs/restoration.yaml](src/projector_distortion/configs/restoration.yaml) is the
whole change. Both the backend and its checkpoint come from the config; there is no
flag for either.

`light` — the frame the projector emitted at that moment — is passed to every restorer
because it is the one signal a ProCam rig has and an ordinary restoration setting does
not. A single-image backend simply ignores it.

Interfaces, the call graph and the public API:
[README_code.md](src/projector_distortion/README_code.md).

---

## 7. Tests

```bash
python -m pytest -q
```

---

## 8. License

MIT for this project's own source, data and restoration weights — see
[LICENSE](LICENSE.txt).

The repository also redistributes third-party artefacts under their own licences. One
of them, `weights/detector_yolo11s.pt`, is **AGPL-3.0**, because it is a fine-tune of
an Ultralytics YOLO11 checkpoint. The default configuration does not use it: the
shipped `detector.backend` is `ssd`, torchvision's BSD-licensed SSDLite, so an
out-of-the-box run touches only MIT/BSD/Apache code. Selecting `yolo` opts into
AGPL-3.0.

Upstream: [ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0) and
[torchvision](https://github.com/pytorch/vision) (BSD-3-Clause). Every other
dependency is MIT, BSD or Apache; see [THIRD_PARTY.md](THIRD_PARTY.md) for the
complete component-by-component breakdown and license boundary guarantees.

---

## 9. Citation

If this software is useful in your research, please cite it. Zenodo issues two kinds of
DOI: **10.5281/zenodo.22153821** always resolves to the newest release and is what the
badge above points at, and each release also gets its own version DOI, listed on that
Zenodo page. Cite the version DOI of the release you used — a result is reproducible
against a version, not against "latest".

```bibtex
@software{lee_prores_det,
  author  = {Lee, JeongHyeon},
  title   = {{ProRes-Det: Projector Restoration and Detection}},
  year    = {2026},
  version = {1.1.0},
  doi     = {10.5281/zenodo.22153821},
  url     = {https://github.com/jhyun-lee/ProRes-Det}
}
```

---

## 10. Contact

Questions, bug reports and attribution corrections:
[ljhyun0821@gnu.ac.kr](mailto:ljhyun0821@gnu.ac.kr), or open an issue on the
repository.
