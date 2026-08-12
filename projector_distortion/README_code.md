# projector_distortion/

What each module owns, how a run flows through them, and the public API.

Script options are in [README_running.md](../README_running.md). The extension recipe for a
new detector or restorer is in [README.md](../README.md#5-swapping-modules).

- [Call graph](#call-graph)
- [Files](#files)
- [The contracts](#the-contracts)
- [Public API](#public-api)
- [Tests](#tests)

---

## Call graph

The three root scripts are thin. They parse args, then hand off.

```
demo.py / evaluate.py / train.py
   │
   ├─ cli.add_common_args        shared flags
   ├─ cli.build_models           YAML + flags → (restorer, detector, info)
   │     ├─ config.load_config   configs/*.yaml, merged
   │     ├─ models.build_restorer
   │     └─ models.build_detector
   │
   ├─ data.find_samples          distorted ↔ light ↔ surface ↔ label, by filename
   │  data.index_triplets        training only: complete triplets
   │
   ├─ pipeline.run_offline       ─┐
   │  pipeline.run_live          ─┤ loop
   │     └─ pipeline.process_sample
   │            ├─ restorer.restore_full
   │            ├─ detector(distorted) · detector(restored)
   │            ├─ models.filter_detections
   │            └─ utils.visualize.draw_detections / grid_2x2
   │
   └─ utils.recording.RunRecorder   everything written to output/<run>/
```

`evaluate.py` skips `run_offline` and drives `process_sample` itself, because it scores
per sample instead of recording frames.

`pipeline/__init__.py` exposes `run_live` as a lazy proxy, so an offline run never loads
the webcam and Win32 window plumbing (`utils/display.py`) it has no use for.

---

## Files

### Top level

| File | Owns |
|---|---|
| [`__init__.py`](__init__.py) | Public API re-exports, `__version__` |
| [`cli.py`](cli.py) | Shared argparse groups, config precedence, `build_models`, device resolution, `run_dir`, and the `pdf-*` console entry points |
| [`config.py`](config.py) | `load_config` (YAML + recursive merge), `resolve_path` (against `PROJECT_ROOT`), `pick` (CLI over YAML) |
| [`data.py`](data.py) | Filename → id parsing, layout detection, `find_samples`, `index_triplets`, label loading (YOLO txt and LabelMe json), `TripletPatchDataset` |

### `configs/`

| File | Holds |
|---|---|
| [`configs/restoration.yaml`](configs/restoration.yaml) | `model` (backend, weights, input_size) · `ablation` (structure + capacity) · `train` (data paths, hyperparameters, loss weights) |
| [`configs/detection.yaml`](configs/detection.yaml) | `detector` (backend, conf, imgsz, box gate) · `weights` per backend · `names` (17 classes) |

### `models/`

| File | Owns |
|---|---|
| [`models/base.py`](models/base.py) | `BaseRestorer`, `BaseDetector`, `Detection`, `NullDetector`, and the `@register_restorer` / `@register_detector` registries |
| [`models/restoration.py`](models/restoration.py) | `RestorationConfig` + the 10 `TOGGLES`, the network (`LayerNorm2d` → `SimpleGate` → `CALayer` → `NAFBlock` → `NAFSEBlock` → `RestorationNet`), checkpoint save/load, `NAFSEUNetRestorer` |
| [`models/detection.py`](models/detection.py) | `YoloDetector` (ultralytics), `SsdDetector` (torchvision), `build_detector`, `filter_detections`, the default `CLASS_NAMES` |

The network is a 3-level U-Net of NAFNet blocks with squeeze-excite channel attention —
`naf_se_unet`, which is what the name says. There is no MDTA attention and no
depthwise-gated FFN, so it is not Restormer, whatever the pre-rename `restormer_like`
tag on older checkpoints suggests. It is fully convolutional, which is why the 180×320
training patch and the 640×360 inference size behave identically.

`ssd` normalises its labels: torchvision heads emit the COCO id (1..N, 0 = background), so
`detect()` subtracts 1. ultralytics is already 0-based.

### `pipeline/`

| File | Owns |
|---|---|
| [`pipeline/offline.py`](pipeline/offline.py) | `FrameResult`, `process_sample` (one pair end to end), `build_panel`, `run_offline`. No hardware |
| [`pipeline/live.py`](pipeline/live.py) | Webcam opening, black/white-flash calibration, homography + warp, the restore/detect worker thread, the writer thread, stride auto-tuning |

`live.py` runs three threads: the main loop projects and captures, a worker restores and
detects, a writer encodes. Queue depths are `MAX_IN_FLIGHT = 3` and
`MAX_PENDING_WRITES = 8`. The analysis stride is fixed once from `STRIDE_WARMUP = 12`
frames with the first `STRIDE_DISCARD = 2` thrown away, because CUDA autotuning makes
frame 1 roughly 50× the steady-state cost.

Window titles: `Projector_Display`, `Combined_View`, `PreWarp_Debug`, `Warp_FirstFrame`.

### `utils/`

| File | Owns |
|---|---|
| [`utils/image.py`](utils/image.py) | `read_bgr`, `resize`, `bgr_to_tensor` / `tensor_to_bgr` / `residual_to_bgr`, `psnr`, `ssim`, `iou`, `IMAGE_EXT` |
| [`utils/visualize.py`](utils/visualize.py) | `draw_detections`, `caption`, `grid_2x2`, `panel_size`, `draw_quad`, `warp_before_after` |
| [`utils/recording.py`](utils/recording.py) | `RunRecorder`, `FRAME_KINDS`, `KIND_DIRS`, `parse_kinds` |
| [`utils/display.py`](utils/display.py) | `Monitor`, `list_monitors`, `place_fullscreen` — Win32 monitor enumeration and borderless placement |

`display.py` is separate because getting a window onto the projector is not a pipeline
concern, and `collect.py` and `record.py` need it too. On Windows,
`cv2.setWindowProperty(WND_PROP_FULLSCREEN)` snaps the window back to the primary display
and silently undoes `cv2.moveWindow()`, which is why a naive `--screen` has no effect;
geometry goes through the Win32 API instead. `live.place_window` is a one-line wrapper
that supplies the projector window's title.

`ssim` is implemented here (gaussian-windowed, 11×11, sigma 1.5) so metrics never drag in
scikit-image or pytorch-msssim. `residual_to_bgr` colourmaps mean |residual| as a JET
heatmap, which is why the scalar cannot be recovered from the image and
`restore_full()` returns it separately.

`RunRecorder` decides what lands where. `FRAME_KINDS` is
`("distorted", "restored", "panel")`; `distorted` and `restored` go to `captures/`,
`panel` goes to `frames_all/`, and calibration images go to `calib/`.

---

## The contracts

Four small types carry everything between modules.

```python
# models/base.py
class BaseRestorer:
    input_size: tuple[int, int] = (640, 360)
    def restore(self, distorted_bgr, light_bgr) -> tuple[restored_bgr, residual_bgr]: ...
    def restore_full(self, distorted_bgr, light_bgr) -> tuple[restored, residual, mean_abs]: ...

class BaseDetector:
    def detect(self, bgr) -> list[Detection]: ...

@dataclass(frozen=True)
class Detection:
    cls_id: int; name: str; conf: float; box: Sequence[int]   # (x1, y1, x2, y2), pixels
```

```python
# data.py
@dataclass(frozen=True)
class Sample:
    name_id: str; distorted: str; light: str
    surface: str | None = None      # optional: needed to score
    label: str | None = None        # .txt (YOLO) or .json (LabelMe)

def load_labels(path, img_w, img_h, class_names=None) -> [(cls_id, (x1,y1,x2,y2)), ...]
```

`load_labels` dispatches on the extension, so the pipeline never learns which annotator
produced a split. `class_names` is only consulted for LabelMe, which stores class names
rather than ids; passing the detector's own list keeps ground truth and predictions on one
set of ids.

```python
# pipeline/offline.py
@dataclass
class FrameResult:
    frame_id, name_id
    light, distorted, distorted_det, restored, restored_det, residual   # BGR arrays
    residual_mean, det_distorted, det_restored, t_restore, t_detect
    surface, gt_boxes                                                    # None / [] without GT
    def metrics(self) -> dict   # psnr/ssim, distorted and restored, None without surface
```

`restore_full` exists so a caller gets mean |residual| from one forward pass instead of
two. The base implementation returns `0.0` rather than a wrong number, and subclasses that
know the value override it.

Field names on `FrameResult` are what `RunRecorder` reads, so renaming one breaks
recording.

---

## Public API

```python
from projector_distortion import build_restorer, build_detector
from projector_distortion.data import find_samples
from projector_distortion.pipeline import process_sample

restorer = build_restorer("weights/restorer_nafse_unet.pt")
detector = build_detector("ssd", "weights/detector_ssdlite.pth")
root = "data/SampleData/sample_eval"     # distorted/ light/ + surface/ labels/ for scoring
for i, s in enumerate(find_samples(root, root)):
    r = process_sample(s, restorer, detector, frame_id=i)
    print(s.name_id, len(r.det_distorted), "->", len(r.det_restored))
```

`__init__.py` re-exports `BaseRestorer`, `BaseDetector`, `Detection`,
`RestorationConfig`, `build_restorer`, `build_detector`, `detector_names`,
`filter_detections`, `CLASS_NAMES`, `load_config`, `resolve_path`, `PROJECT_ROOT`.

Submodule `__init__.py` files re-export more: `models` exposes the network internals and
checkpoint helpers, `utils` exposes every helper listed above.

---

## Tests

114 tests, no hardware needed.

| File | Covers |
|---|---|
| [`../tests/conftest.py`](../tests/conftest.py) | Fixtures `root` / `bgr_image` / `pro_beam`, and the skips for a missing checkpoint or optional module |
| [`../tests/test_pipeline.py`](../tests/test_pipeline.py) | Filename ids, sample discovery, PSNR/SSIM/IoU, `RunRecorder` behaviour, triplet indexing, `average_precision`, the argparse defaults, unknown-backend handling, the `live._worker` queue contract driven with stubs, `device_note`, the `requirements-cuda.txt` pins |
| [`../tests/test_restoration.py`](../tests/test_restoration.py) | `RestorationConfig` and its tags, every toggle building and training one step, forward shapes, checkpoint round trip, the shipped weights, the restorer registry with a third-party backend |
| [`../tests/test_detection.py`](../tests/test_detection.py) | Registry, label normalisation, the box size gate, both backends against the real checkpoints |
| [`../tests/test_collect.py`](../tests/test_collect.py) | `collect.py` without a rig: corner ordering, boundary resampling, `boundary` vs `homography` equivalence, the `warp` and `light` stage outputs |

```bash
python -m pytest -q
```

---

[한국어](README_code.ko.md) · [← README](../README.md)
