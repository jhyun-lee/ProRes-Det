# Third-Party Licenses and Intellectual Property Notices

This document details the licenses and intellectual property notices for third-party software and model weights incorporated into, depended upon, or redistributed with **ProRes-Det**.

---

## 1. Project License Overview

| Component | Role in ProRes-Det | License | Upstream Source / Reference |
|---|---|---|---|
| **ProRes-Det Source Code** | Core framework, pipeline, CLI, evaluation | **MIT License** | [LICENSE.txt](LICENSE.txt) |
| **ProRes-Det Dataset** (`SampleData/`) | Bundled image pairs and benchmark ground truth | **MIT License** | Self-collected |
| **Restoration Weights** (`restorer_unet_best.pt`) | Default restoration checkpoint (NAFSE-UNet) | **MIT License** | Fitted from scratch by ProRes-Det on `sample_train` |
| **Restoration Weights** (`restorer_CNN_best.pt`) | Basic CNN restoration baseline checkpoint | **MIT License** | Fitted from scratch by ProRes-Det on `sample_train` |
| **SSDLite Weights** (`detector_ssdlite.pth`) | Default detection model checkpoint | **BSD-3-Clause** | PyTorch / torchvision |
| **YOLO11s Weights** (`detector_yolo11s.pt`) | Optional alternative detection checkpoint | **AGPL-3.0** | Ultralytics YOLO11 fine-tune |

---

## 2. License Boundaries and Isolation Principle

ProRes-Det is structured to ensure a **strict boundary** between permissive open-source components (MIT/BSD/Apache) and copyleft components (AGPL-3.0):

1. **Default Out-of-the-Box Mode (Permissive Only)**:
   - When installed using the default command (`pip install -e .` or `pip install -e . -r requirements-cuda.txt`), the framework installs **zero AGPL-licensed dependencies**.
   - The default detector backend configured in `configs/detection.yaml` is `ssd` ([SSDLite320-MobileNetV3-Large](https://github.com/pytorch/vision)), licensed under **BSD-3-Clause**.
   - Running default commands (`pdf-demo`, `pdf-evaluate`, `pdf-train`) touches exclusively **MIT, BSD-3-Clause, and Apache-2.0** code and models.

2. **Optional YOLO Backend (AGPL-3.0 Opt-in)**:
   - The `ultralytics` package and the checkpoint `weights/detector_yolo11s.pt` are licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.
   - Users opt into the AGPL-3.0 license **only** if they explicitly:
     - Install the YOLO extra via `pip install -e ".[yolo]"` or `pip install -e ".[all]"`.
     - Select `backend: yolo` in `configs/detection.yaml`.
   - The framework accesses YOLO exclusively through dynamic modular dispatch ([BaseDetector](src/projector_distortion/models/base.py) interface). If `ultralytics` is not installed, the framework runs normally using `ssd` and only raises an informative warning if the user deliberately requests the `yolo` backend.

---

## 3. Third-Party Dependencies and Attributions

### Permissive Dependencies (Default Install)
- **PyTorch & torchvision** ([BSD-3-Clause](https://github.com/pytorch/pytorch/blob/main/LICENSE))
  - Deep learning framework and MobileNetV3 / SSDLite models.
- **OpenCV Python (`opencv-python`)** ([Apache-2.0](https://github.com/opencv/opencv-python/blob/master/LICENSE.txt))
  - Image I/O, resizing, matrix operations, and drawing utilities.
- **NumPy** ([BSD-3-Clause](https://github.com/numpy/numpy/blob/main/LICENSE.txt))
  - Numerical arrays and mathematical operations.
- **PyYAML** ([MIT](https://github.com/yaml/pyyaml/blob/master/LICENSE))
  - YAML configuration parsing and merging.
- **tqdm** ([MPL-2.0 / MIT](https://github.com/tqdm/tqdm/blob/master/LICENCE))
  - Progress bar displays for CLI execution.

### Optional Extras
- **Ultralytics YOLO** ([AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE))
  - Object detection architecture and weights.
  - Required only for `detector.backend: yolo`.
- **pytorch-msssim** ([MIT](https://github.com/VainF/pytorch-msssim/blob/master/LICENSE))
  - Fast MS-SSIM loss calculation used during restorer training (`train.py`).
- **screeninfo** ([MIT](https://github.com/rr-/screeninfo/blob/master/LICENSE))
  - Multi-monitor resolution discovery for live capture on non-Windows platforms.

### Model Architecture References
- **NAFNet** (Nonlinear Activation Free Network):
  - Chen et al., *Simple Baselines for Image Restoration*, ECCV 2022.
  - Reference implementation: [megvii-research/NAFNet](https://github.com/megvii-research/NAFNet) (MIT License).
  - The `NAFBlock`, `SimpleGate`, and LayerNorm placements used in `RestorationNet` are re-implemented in accordance with the published paper.
- **Squeeze-and-Excitation Networks**:
  - Hu et al., *Squeeze-and-Excitation Networks*, CVPR 2018.
- **U-Net**:
  - Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015.

---

## 4. Permissively Licensed Configuration

Users who wish to avoid the AGPL-3.0-licensed Ultralytics runtime should install the base package without the `[yolo]` or `[all]` extras and select the default SSDLite backend (`backend: ssd`).

The optional Ultralytics YOLO11 backend and its bundled model weights remain subject to AGPL-3.0. Users who enable or redistribute these components must review the applicable licensing obligations.
