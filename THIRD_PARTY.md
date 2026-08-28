# Third-party components

ProRes-Det's own source is MIT ([LICENSE](LICENSE)). It also *ships* and *depends on*
work by others, not all of it under MIT, and this file is the accounting of that.

Two different obligations are separated below, because they are not the same thing:

- **Redistributed artefacts** — files physically present in this repository that
  someone else holds copyright in. Their licence governs them, not this project's.
- **Dependencies** — installed by pip, never redistributed here.

Architectures that were *reimplemented* from a paper rather than copied are listed
last: those carry an academic citation obligation, not a licence one.

---

## Redistributed in this repository

| File | Derived from | Licence |
|---|---|---|
| `weights/detector_yolo11s.pt` | [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) — fine-tuned from `yolo11s.pt` | **AGPL-3.0** |
| `weights/detector_ssdlite.pth` | [torchvision](https://github.com/pytorch/vision) `ssdlite320_mobilenet_v3_large` | BSD-3-Clause |
| `weights/restorer_nafse_unet.pt` | Trained from scratch in this project | MIT |
| `data/SampleData/**`, `data/live/**` | Collected for this project | MIT |

### The AGPL-3.0 file, specifically

`weights/detector_yolo11s.pt` is a fine-tune of an Ultralytics checkpoint. Ultralytics
releases its code *and its models* under AGPL-3.0, and the file records that itself —
`torch.load(...)["license"]` reads `AGPL-3.0 (https://ultralytics.com/license)`.

It is therefore **not** covered by this project's MIT licence, and MIT is not being
claimed over it.

What that means in practice:

- The default configuration does **not** use it. `detector.backend` is `ssd`, so
  `python demo.py` and `python evaluate.py` out of the box run torchvision's
  BSD-licensed detector and never load this file.
- Selecting `yolo` — in [configs/detection.yaml](projector_distortion/configs/detection.yaml)
  or by installing the `[yolo]` extra — opts into AGPL-3.0 for whatever you build on
  top. If that is a problem for your use, stay on `ssd` or supply your own weights.
- Ultralytics sells an Enterprise licence for use outside AGPL terms. That is between
  you and them; this project takes no position on it.

The reference table in the README reports both backends precisely so the result does
not depend on the AGPL one.

---

## Dependencies (installed, not redistributed)

| Package | Licence | Needed for |
|---|---|---|
| [torch](https://github.com/pytorch/pytorch) | BSD-3-Clause | everything |
| [torchvision](https://github.com/pytorch/vision) | BSD-3-Clause | the `ssd` detector, VGG19 perceptual loss |
| [opencv-python](https://github.com/opencv/opencv-python) | Apache-2.0 | all image and video I/O |
| [numpy](https://github.com/numpy/numpy) | BSD-3-Clause | everything |
| [PyYAML](https://github.com/yaml/pyyaml) | MIT | reading `configs/*.yaml` |
| [tqdm](https://github.com/tqdm/tqdm) | MPL-2.0 AND MIT | progress bars |
| [ultralytics](https://github.com/ultralytics/ultralytics) | **AGPL-3.0** | the `yolo` detector — optional extra `[yolo]` |
| [pytorch-msssim](https://github.com/VainF/pytorch-msssim) | MIT | SSIM loss term — optional extra `[train]` |
| [pandas](https://github.com/pandas-dev/pandas) | BSD-3-Clause | training logs — optional extra `[train]` |
| [matplotlib](https://github.com/matplotlib/matplotlib) | PSF-based | training plots — optional extra `[train]` |
| [screeninfo](https://github.com/rr-/screeninfo) | MIT | `--live` on non-Windows — optional extra `[live]` |
| [pytest](https://github.com/pytest-dev/pytest) | MIT | tests — optional extra `[test]` |

`ultralytics` is the only AGPL entry and it is the only one behind an optional extra
that the default configuration never reaches.

### Pretrained weights pulled at runtime

`train.py`'s perceptual loss builds a VGG19 with `VGG19_Weights.IMAGENET1K_V1`, which
torchvision downloads on first use. Those weights are trained on ImageNet and carry
ImageNet's own terms of use; they are not redistributed here. Training without that
term is a matter of setting `loss.perceptual: 0` in
[configs/restoration.yaml](projector_distortion/configs/restoration.yaml).

---

## Reimplemented, not copied

No source was taken from these. They are cited because the architecture follows them,
which is an academic obligation rather than a licensing one.

| Idea | Where it comes from |
|---|---|
| NAFBlock — SimpleGate, the LayerNorm placement, the learnable residual scales | Chen et al., *Simple Baselines for Image Restoration*, ECCV 2022 — [megvii-research/NAFNet](https://github.com/megvii-research/NAFNet) |
| Squeeze-and-excitation channel attention | Hu et al., *Squeeze-and-Excitation Networks*, CVPR 2018 |
| U-Net encoder/decoder skeleton | Ronneberger et al., *U-Net*, MICCAI 2015 |

`projector_distortion/models/restoration.py` names the network `naf_se_unet` for
exactly this reason, and its module docstring records what it deliberately does *not*
take from Restormer.

---

## If you find something missing here

Attribution errors are bugs. Mail
[ljhyun0821@gnu.ac.kr](mailto:ljhyun0821@gnu.ac.kr) or open an issue and it will be
corrected.

> This file is a good-faith accounting by the authors, not legal advice. If your use
> has licensing consequences you cannot absorb — commercial redistribution in
> particular — check with someone qualified rather than relying on this page.
