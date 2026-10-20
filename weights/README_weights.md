# weights/

Two restoration checkpoints and two detector checkpoints, their selection settings,
and the shared residual convention.

- [Files](#files)
- [Selecting a restorer and detector](#selecting-a-restorer-and-detector)
- [Restoration training and selection](#restoration-training-and-selection)
- [Detector training provenance](#detector-training-provenance)
- [Checkpoint format](#checkpoint-format)
- [The residual convention](#the-residual-convention)

## Files

The current directory contains four checkpoints (51.5 MiB total).
Paths in the configuration resolve relative to the project root.

| File | Role / backend | Architecture | Parameters | Size |
|---|---|---|---:|---:|
| `restorer_unet_best.pt` | Default restoration / `naf_se_unet` | Three-level NAFSE-UNet, FULL configuration | 4,184,259 | 16.09 MiB |
| `restorer_CNN_best.pt` | Basic restoration baseline / `plain_cnn` | Five 3×3 convolutions, width 32 | 30,371 | 0.12 MiB |
| `detector_ssdlite.pth` | Default detection / `ssd` | SSDLite320-MobileNetV3-Large, 17 classes | 4,393,592 | 17.02 MiB |
| `detector_yolo11s.pt` | Optional detection / `yolo` | YOLO11s, 17 classes | 9,434,371 | 18.28 MiB |

The detector classes are listed in
[detection.yaml](../src/projector_distortion/configs/detection.yaml).
YOLO uses the names embedded in its checkpoint; SSDLite uses the configured list.
The restoration models use image pairs and surface targets, with no detection
annotations required for restoration training.

## Selecting a restorer and detector

Edit `model.backend` and `model.weights` together in
[restoration.yaml](../src/projector_distortion/configs/restoration.yaml).
The default NAFSE-UNet selection is:

```yaml
model:
  backend: naf_se_unet
  weights: weights/restorer_unet_best.pt
  input_size: [640, 360]
```

For the Basic CNN baseline, replace that selection with:

```yaml
model:
  backend: plain_cnn
  weights: weights/restorer_CNN_best.pt
  input_size: [640, 360]

plain_cnn:
  width: 32
  depth: 5
```

Keep one `model` block in the configuration. The current checkpoints have matching
architecture metadata; selecting CNN with NAFSE-UNet weights, or vice versa, raises
a backend mismatch error. No model-selection CLI flag is needed. Both `demo.py`
and `evaluate.py` read this configuration; output folder names do not select models.

The shipped [detection.yaml](../src/projector_distortion/configs/detection.yaml)
runs SSDLite only:

```yaml
detector:
  backend: ssd
  conf: 0.25

weights:
  ssd: weights/detector_ssdlite.pth
  yolo: weights/detector_yolo11s.pt
```

YOLO is optional: install `pip install -e ".[yolo]"` and select `backend: yolo`.
For evaluation with both detectors, use `backend: [ssd, yolo]`. `evaluate.py`
scores each entry; `demo.py` uses the first one. See
[THIRD_PARTY.md](../THIRD_PARTY.md) for the YOLO license terms.

From the project root, evaluate the currently selected restorer and detector:

```bash
python src/evaluate.py \
  --input datasets/SampleData/sample_test \
  --gt datasets/SampleData/sample_test \
  --iou 0.5 \
  --output output/reproduce_selected/iou05
```

Use distinct output directories for different restorers and IoU thresholds.
Latest detection and restoration results, including both backends, are in
[Reference results](../README.md#3-reference-results).

## Restoration training and selection

Both models were trained for 30 epochs using the same 3,000 `sample_train` pairs,
seed 42, batch size 4, gradient accumulation of 4, learning rate 0.0002, and
restoration loss weights. Only training samples update the weights.
After every epoch, all 200 `sample_eval` pairs are restored at 640×360 without
gradients or random crops. The checkpoint maximizing mean validation PSNR is
copied to `weights/`; the 1,000-pair, 10-scene `sample_test` split is reserved
for final evaluation.

| File | Selected epoch / training budget | Mean validation PSNR | Mean validation SSIM |
|---|---:|---:|---:|
| `restorer_unet_best.pt` | 28 / 30 | 23.5281 dB | 0.8844 |
| `restorer_CNN_best.pt` | 17 / 30 | 19.3022 dB | 0.8192 |

The NAFSE-UNet checkpoint has about 138× as many parameters as the CNN baseline.
These runs compare complete backends; they are one training run per model,
not repeated-seed statistics or a capacity-controlled block ablation.

## Detector training provenance

The YOLO checkpoint records a COCO-pretrained `yolo11s.pt` base, 50 fine-tuning
epochs, image size 640, batch size 16, seed 0, and Ultralytics 8.4.61
(2026-06-18). Its recorded validation metrics belong to its own detector-training
split and are separate from the current restoration evaluation.
`detector_ssdlite.pth` is a bare state dictionary with no embedded training record.

The detectors were fine-tuned before the released restoration splits were formed.
Their annotated training pool is not distributed here, so overlap with
`sample_eval` scenes cannot be established from this repository alone.
`sample_test` annotations were prepared separately in LabelMe format; this
annotation history alone does not establish disjointness from the detector's
original training pool. The reported comparisons hold detector weights fixed
and measure the effect of restoration on the same captured inputs.

## Checkpoint format

Both current restoration checkpoints use format 2 with embedded architecture
configuration and validation-selection metadata. A schematic checkpoint is:

```python
{
    "format": 2,
    "arch": "naf_se_unet",  # "plain_cnn" for the Basic CNN checkpoint
    "cfg": {...},
    "state_dict": {...},
    "epoch": 28,
    "selection_metric": "val_psnr",
    "selection_mode": "max",
    "best_epoch": 28,
    "best_val_psnr": 23.528064169042526,
    # Training settings and validation results are stored alongside these fields.
}
```

`load_checkpoint()` rebuilds the matching network from `arch` and `cfg`.
Keep architecture settings at their defaults to use the embedded configuration;
a non-default `ablation` or `plain_cnn` block explicitly overrides configuration
within the selected model family and must match the stored weights.

| `cfg_source` | Meaning |
|---|---|
| `embedded` | Architecture settings came from the checkpoint; normal for both current files |
| `explicit` | Non-default architecture settings were supplied explicitly |
| `legacy-raw` | A bare restoration state dictionary, reconstructed from the selected family/defaults |
| `defaults` | A structured checkpoint without embedded `cfg` |

## The residual convention

Both restorers accept the distorted capture and its projected light image,
concatenated into six channels, and predict a three-channel residual:

```text
input    (B, 6, H, W) = cat([distorted, light]) in [-1, 1]
residual (B, 3, H, W) = net(input)
restored = (distorted - residual).clamp(-1, 1)
```

NAFSE-UNet uses a U-Net of NAFSE blocks; Basic CNN uses five 3×3 convolutions
with channels 6 → 32 → 32 → 32 → 32 → 3, ReLU after the first four layers,
and Tanh at the output. Both current checkpoints bound the residual with Tanh.

Training compares the restored image against its projection-free surface target;
the residual has no separate target. The configured loss is:

```python
loss = (0.9262 * L1(restored, surface)
      + 2.0369 * Perceptual(restored, surface)
      + 0.5319 * (1 - SSIM(restored, surface))
      + 0.8977 * WaveletHF(restored, surface))
```

The perceptual term uses VGG19 features; the wavelet term compares Haar
high-frequency bands. See [train.py](../src/train.py) and
[restoration.yaml](../src/projector_distortion/configs/restoration.yaml).
In the training split, 20 surface targets are shared across 3,000 captures with
different projected illumination. Predicting a residual provides an identity path
from the capture, but does not by itself guarantee that a model cannot memorize
scenes or generate incorrect content. Scene-disjoint evaluation remains necessary.

A custom restorer that emits the surface directly can still implement
`BaseRestorer`; see [Swapping modules](../README.md#6-swapping-modules).

---

[← README](../README.md)
