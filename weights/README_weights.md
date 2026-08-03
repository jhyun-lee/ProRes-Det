# weights/

These three checkpoints are committed to git (51 MiB in total), so a clone runs with
no extra download. To use your own instead, point at them with `--restorer-weights` /
`--det-weights` rather than overwriting these.

## Files

| File | Used for | Architecture | Params | Size |
|---|---|---|---|---|
| `restorer_restormerlike.pt` | restoration | 3-level U-Net of RestormerLikeBlock | 4,184,259 | 16.1 MiB |
| `detector_yolo11s.pt` | detection (`--detector yolo`) | YOLO11s | 9,434,371 | 18.3 MiB |
| `detector_ssdlite.pth` | detection (`--detector ssd`) | SSDLite320-MobileNetV3-Large | 4,393,592 | 17.0 MiB |

All three are fine-tuned on the same 17-class projector-distortion dataset (11 fruits +
6 animals). The class list lives in
[configs/detection.yaml](../projector_distortion/configs/detection.yaml); the YOLO
checkpoint carries the same 17 names internally, and those win unless `--classes` is
passed.

Default paths come from the configs, so no path is hard-coded in the source:

```yaml
# projector_distortion/configs/restoration.yaml
model:
  weights: weights/restorer_restormerlike.pt

# projector_distortion/configs/detection.yaml
weights:
  yolo: weights/detector_yolo11s.pt
  ssd:  weights/detector_ssdlite.pth
```

Paths resolve against the project root, not the working directory.

## Using other weights

```bash
python demo.py --restorer-weights path/to/other.pt
python demo.py --detector ssd --det-weights path/to/other.pth
```

## Checkpoint format

Anything `train.py` produces stores the architecture config next to the weights:

```python
{"format": 2, "arch": "restormer_like",
 "cfg": {...RestorationConfig...}, "state_dict": {...},
 "epoch": 30, "loss": 0.1234, ...}
```

`load_checkpoint()` reads `cfg` and rebuilds the matching architecture automatically,
so an ablated variant never needs its flags repeated at inference time.

```bash
python train.py --no-ca --epochs 30
python demo.py --restorer-weights runs/0730_1948_30ep_NoCA/restorer_NoCA_best.pt
#                                  ↑ no --no-ca needed
```

`restorer_restormerlike.pt` is a bare state_dict with no embedded config, so it is
rebuilt from the defaults (every toggle on, tag `FULL`). A strict load succeeds, which
confirms the file really is the FULL variant. Runs report this as `cfg from legacy-raw`.

## The residual convention

The restoration network does not draw the restored image. It emits the light to
subtract from `pro`:

```
input   (B, 6, H, W) = cat([pro, beam])  in [-1, 1]
output  (B, 3, H, W) = residual
restored = (pro - residual).clamp(-1, 1)
```

### Why the residual, not the clean image

1) Preserving the original becomes the default.
Where no projector light lands, residual ≈ 0 is enough and the `pro` pixel passes
through untouched. "Do nothing" is the identity, so the network only has to learn
what must change. Regressing clean directly forces it to regenerate perfectly good
background too, which smears regions that should have been left alone.

2) It blocks the "paint the objects in" overfit.
If the network outputs clean directly, the fastest way to drive the loss down is to
largely ignore the input and reproduce a screen memorised from the training set.
The bundled data makes that especially tempting: 10 clean images back 22 `pro`
captures, so clean repeats and memorising the target per `oriId` pays off. A model
trained that way makes the detector see objects that were never there, which
destroys the whole point of the evaluation. `restored = pro − residual` forces the
output to always derive from real camera pixels, closing that shortcut.

3) Values cannot blow up.
The output `tanh` bounds the residual to [-1, 1], and `clamp(-1, 1)` bounds the result
after the subtraction — two layers of range control. (`--no-tanh` removes the first;
that is one of the ablation switches.)

### How it is enforced — the loss is on `restored`, not on the residual

The key is that the subtraction lives inside the graph. The residual is never given
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
| `WaveletHF` (Haar LH/HL/HH, LL excluded) | Edges and texture only | Blurring everything to lower the loss |

Dropping the low-frequency (LL) band is the point of `WaveletHF`. Blurring the whole
image still lowers L1, but not the high-frequency term. That is what makes the residual
follow the actual boundaries of the projected light.

The weights live under `train.loss` in
[configs/restoration.yaml](../projector_distortion/configs/restoration.yaml) and come
from an Optuna sweep on this dataset. Implementation: [train.py](../train.py).

> A custom restorer that emits clean directly still satisfies the `BaseRestorer`
> interface. In that case the `residual` visualisation and the `residual_mean` metric
> lose their meaning, and both advantages above are gone. See
> [Swapping modules](../README.md#5-swapping-modules).

---

[한국어](README_weights.ko.md) · [← README](../README.md)
