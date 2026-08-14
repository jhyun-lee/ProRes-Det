# weights/

The three checkpoints, the checkpoint format, and why the restoration network predicts a
residual instead of the surface image.

- [Files](#files)
- [Where the paths come from](#where-the-paths-come-from)
- [Checkpoint format](#checkpoint-format)
- [The residual convention](#the-residual-convention)

---

## Files

All three are committed to git — 51 MiB in total — so a clone runs with no extra download.
To use your own, point the configs below at them rather than overwriting these.

| File | Used for | Architecture | Params | Size |
|---|---|---|---|---|
| `restorer_nafse_unet.pt` | restoration | 3-level U-Net of NAFSEBlock | 4,184,259 | 16.1 MiB |
| `detector_yolo11s.pt` | the `yolo` detector | YOLO11s | 9,434,371 | 18.3 MiB |
| `detector_ssdlite.pth` | the `ssd` detector | SSDLite320-MobileNetV3-Large | 4,393,592 | 17.0 MiB |

All three are fine-tuned on the same 17-class projector-distortion dataset — 11 fruits and
6 animals. The class list lives in
[configs/detection.yaml](../projector_distortion/configs/detection.yaml). The YOLO
checkpoint carries the same 17 names internally, and those win over the config's list.
torchvision has no names, so `ssd` always uses the config's list.

---

## Where the paths come from

The configs, and only the configs — there is no CLI flag for any of these paths, so a
checkpoint swap is a one-line edit that every entry point picks up at once:

```yaml
# projector_distortion/configs/restoration.yaml
model:
  backend: naf_se_unet
  weights: weights/restorer_nafse_unet.pt

# projector_distortion/configs/detection.yaml
weights:
  yolo: weights/detector_yolo11s.pt
  ssd:  weights/detector_ssdlite.pth
```

These resolve against the project root, not the working directory.

---

## Checkpoint format

Anything `train.py` produces stores the architecture config next to the weights:

```python
{"format": 2, "arch": "naf_se_unet",
 "cfg": {...RestorationConfig...}, "state_dict": {...},
 "epoch": 30, "loss": 0.1234, ...}
```

`load_checkpoint()` reads `cfg` and rebuilds the matching architecture, so an ablated
variant never needs its config restated at inference time.

```bash
# ablation: { use_ca: false } in projector_distortion/configs/restoration.yaml
python train.py --epochs 30
# then point model.weights at runs/0730_1948_30ep_NoCA/restorer_NoCA_best.pt
python demo.py
#   the ablation block does not have to stay set: the config travels in the checkpoint
```

Runs print where the config came from:

| `cfg from` | Meaning |
|---|---|
| `embedded` | The checkpoint carried its own `cfg`. Normal for anything `train.py` wrote |
| `legacy-raw` | A bare `state_dict`, rebuilt from the defaults |
| `defaults` | Format 2 with no `cfg` key |

`restorer_nafse_unet.pt` is `legacy-raw`: a bare state_dict with no embedded config, so
it is rebuilt from the defaults — every toggle on, tag `FULL`. A strict load succeeds,
which confirms the file really is the FULL variant.

---

## The residual convention

The restoration network does not draw the restored image. It emits the light to subtract
from `distorted`:

```
input   (B, 6, H, W) = cat([distorted, light])  in [-1, 1]
output  (B, 3, H, W) = residual
restored = (distorted - residual).clamp(-1, 1)
```

### Why the residual, not the surface image

**1) Preserving the original becomes the default.**
Where no projector light lands, residual ≈ 0 is enough and the `distorted` pixel passes through
untouched. "Do nothing" is the identity, so the network only has to learn what must
change. Regressing surface directly forces it to regenerate perfectly good background too,
which smears regions that should have been left alone.

**2) It blocks the "paint the objects in" overfit.**
If the network outputs surface directly, the fastest way to drive the loss down is to largely
ignore the input and reproduce a screen memorised from the training set. The bundled data
makes that especially tempting: 10 surface images back 22 `distorted` captures, so surface repeats
and memorising the target per `surfaceId` pays off. A model trained that way makes the detector
see objects that were never there, which destroys the whole point of the evaluation.
`restored = distorted − residual` forces the output to always derive from real camera pixels.

**3) Values cannot blow up.**
The output `tanh` bounds the residual to [-1, 1], and `clamp(-1, 1)` bounds the result
after the subtraction — two layers of range control. `use_tanh: false` removes the first;
that is one of the ablation switches.

### How it is enforced — the loss is on `restored`, not on the residual

The subtraction lives inside the graph. The residual is never given a target of its own;
only the subtracted result is compared against surface. What to subtract is left for the
network to discover.

```python
residual = net(torch.cat([distorted, light], dim=1))     # network output
restored = (distorted - residual).clamp(-1, 1)          # subtraction inside the graph
loss = (0.93 * L1(restored, surface)
      + 2.04 * Perceptual(restored, surface)
      + 0.53 * (1 - SSIM(restored, surface))
      + 0.90 * WaveletHF(restored, surface))        # all four measure `restored`
```

| Loss term | What it measures | What it penalises |
|---|---|---|
| `L1` | Absolute pixel error | Global colour / brightness drift |
| `Perceptual` (VGG19 relu3_3) | Feature-map distance | Pixel-close results whose structure is broken |
| `1 − SSIM` | Local luminance, contrast, structure | Flat output that only matched the mean |
| `WaveletHF` (Haar LH/HL/HH, LL excluded) | Edges and texture only | Blurring everything to lower the loss |

Dropping the low-frequency LL band is the point of `WaveletHF`. Blurring the whole image
still lowers L1, but not the high-frequency term. That is what makes the residual follow
the actual boundaries of the projected light.

The weights live under `train.loss` in
[configs/restoration.yaml](../projector_distortion/configs/restoration.yaml) and come from
an Optuna sweep on this dataset. Implementation: [train.py](../train.py).

> A custom restorer that emits surface directly still satisfies the `BaseRestorer` interface.
> In that case the `residual` visualisation and the `residual_mean` metric lose their
> meaning, and both advantages above are gone. See
> [Swapping modules](../README.md#5-swapping-modules).

---

[한국어](README_weights.ko.md) · [← README](../README.md)
