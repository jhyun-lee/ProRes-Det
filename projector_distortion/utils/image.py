# projector_distortion/utils/image.py
"""Image <-> tensor conversion, resizing, and quality metrics."""

from typing import Optional, Tuple

import cv2
import numpy as np
import torch

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def read_bgr(path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return img


def resize(img, size: Tuple[int, int]) -> np.ndarray:
    """size is (width, height). Shrinks with INTER_AREA, grows with INTER_LINEAR."""
    if (img.shape[1], img.shape[0]) == tuple(size):
        return img
    shrinking = size[0] * size[1] < img.shape[1] * img.shape[0]
    return cv2.resize(img, tuple(size),
                      interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR)


def bgr_to_tensor(bgr, device="cpu", size: Optional[Tuple[int, int]] = None):
    """BGR uint8 HWC -> RGB float CHW in [-1, 1], batched."""
    if size is not None:
        bgr = resize(bgr, size)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    arr = (rgb.astype(np.float32) / 127.5) - 1.0
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)


def tensor_to_bgr(t, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Batched RGB float CHW in [-1, 1] -> BGR uint8 HWC."""
    img = t.detach().float().squeeze(0).cpu().numpy().transpose(1, 2, 0)
    img = ((img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return resize(bgr, size) if size is not None else bgr


def residual_to_bgr(residual, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Mean |residual| across channels as a JET heatmap - what the model removed."""
    mag = residual.detach().float().squeeze(0).abs().mean(0).cpu().numpy()
    mag = (mag / 2.0 * 255.0).clip(0, 255).astype(np.uint8)   # residual is in [-1,1]
    heat = cv2.applyColorMap(mag, cv2.COLORMAP_JET)
    return resize(heat, size) if size is not None else heat


# --- quality metrics ----------------------------------------------------------
# Only meaningful against an un-annotated image; the pipeline keeps clean copies
# precisely so these stay computable after a run.

def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 255.0) -> float:
    """Peak signal-to-noise ratio in dB. inf when the images are identical."""
    a, b = _match(a, b)
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(data_range ** 2 / mse))


def ssim(a: np.ndarray, b: np.ndarray, data_range: float = 255.0) -> float:
    """
    Mean SSIM over channels, gaussian-windowed (11x11, sigma 1.5).

    Self-contained so metrics do not drag in scikit-image or pytorch-msssim.
    """
    a, b = _match(a, b)
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if a.ndim == 2:
        a, b = a[..., None], b[..., None]

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    win = cv2.getGaussianKernel(11, 1.5)
    scores = []
    for ch in range(a.shape[2]):
        x, y = a[:, :, ch], b[:, :, ch]
        mu_x = cv2.sepFilter2D(x, -1, win, win)
        mu_y = cv2.sepFilter2D(y, -1, win, win)
        xx = cv2.sepFilter2D(x * x, -1, win, win) - mu_x * mu_x
        yy = cv2.sepFilter2D(y * y, -1, win, win) - mu_y * mu_y
        xy = cv2.sepFilter2D(x * y, -1, win, win) - mu_x * mu_y
        num = (2 * mu_x * mu_y + c1) * (2 * xy + c2)
        den = (mu_x ** 2 + mu_y ** 2 + c1) * (xx + yy + c2)
        scores.append(float(np.mean(num / den)))
    return float(np.mean(scores))


def _match(a, b):
    """Resize b onto a when they differ, so metrics never silently compare mismatches."""
    if a.shape[:2] != b.shape[:2]:
        b = resize(b, (a.shape[1], a.shape[0]))
    if a.ndim != b.ndim:
        raise ValueError(f"channel mismatch: {a.shape} vs {b.shape}")
    return a, b


def iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in box_a)
    bx1, by1, bx2, by2 = (float(v) for v in box_b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
