"""
Helpers that keep the pipeline code readable.

`display` is deliberately not re-exported here: it imports ctypes, and an offline run
has no windows to place. Import it as `projector_distortion.utils.display`.
"""

from .image import (
    IMAGE_EXT,
    bgr_to_tensor,
    iou,
    psnr,
    read_bgr,
    residual_to_bgr,
    resize,
    ssim,
    tensor_to_bgr,
)
from .recording import FRAME_KINDS, RunRecorder
from .visualize import (
    caption,
    draw_detections,
    draw_quad,
    grid_2x2,
    panel_size,
    warp_before_after,
)

__all__ = [
    "read_bgr", "resize", "bgr_to_tensor", "tensor_to_bgr", "residual_to_bgr",
    "psnr", "ssim", "iou", "IMAGE_EXT",
    "draw_detections", "caption", "grid_2x2",
    "panel_size", "draw_quad", "warp_before_after",
    "RunRecorder", "FRAME_KINDS",
]
