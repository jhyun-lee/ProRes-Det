# projector_distortion/utils/visualize.py
"""Box drawing, captions, and the before/after comparison layouts."""

from typing import Sequence

import cv2
import numpy as np

BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 0, 255)
GT_COLOR = (0, 200, 255)

CORNER_TAGS = ("TL", "TR", "BR", "BL")
_CORNER_COLORS = ((0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 128, 0))


def draw_detections(img, detections, color=BOX_COLOR, text_color=TEXT_COLOR,
                    thickness=2) -> np.ndarray:
    """Draw in place and return the image. Pass a copy to keep a clean original."""
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(img, f"{d.name} {d.conf:.2f}", (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    return img


def draw_ground_truth(img, boxes, names=None, color=GT_COLOR) -> np.ndarray:
    """Dashed-looking GT boxes in a distinct colour, for evaluate.py overlays."""
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        if names is not None and i < len(names):
            cv2.putText(img, f"GT:{names[i]}", (x1, min(y2 + 14, img.shape[0] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return img


def caption(img, text, org=(8, 22), scale=0.6) -> np.ndarray:
    """White text with a black outline, readable over any content."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3,
                cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1,
                cv2.LINE_AA)
    return img


def side_by_side(left, right, left_label="before", right_label="after") -> np.ndarray:
    """Horizontal before/after pair with captions."""
    if left.shape[:2] != right.shape[:2]:
        right = cv2.resize(right, (left.shape[1], left.shape[0]))
    return cv2.hconcat([caption(left.copy(), left_label),
                        caption(right.copy(), right_label)])


def grid_2x2(top_left, top_right, bottom_left, bottom_right, labels=None) -> np.ndarray:
    """
    2x2 comparison panel. All four tiles are resized onto the first one.

    Default use: beam | captured+det / restored+det | residual heatmap.
    """
    h, w = top_left.shape[:2]
    tiles = []
    for i, img in enumerate((top_left, top_right, bottom_left, bottom_right)):
        t = img if img.shape[:2] == (h, w) else cv2.resize(img, (w, h))
        t = t.copy()
        if labels and i < len(labels) and labels[i]:
            caption(t, labels[i])
        tiles.append(t)
    return cv2.vconcat([cv2.hconcat(tiles[:2]), cv2.hconcat(tiles[2:])])


def draw_quad(frame, points, title=None, thickness=2) -> np.ndarray:
    """
    Calibration quad on a copy of `frame`, one colour per corner plus coordinates.

    Drawn on the *raw* pre-warp camera view, which is what makes a mis-ordered or
    drifted calibration obvious at a glance.
    """
    canvas = frame.copy()
    pts = [tuple(int(v) for v in p) for p in points]
    cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], True, (255, 0, 0), thickness)
    for tag, p, color in zip(CORNER_TAGS, pts, _CORNER_COLORS):
        cv2.circle(canvas, p, 6, color, -1)
        cv2.circle(canvas, p, 7, (255, 255, 255), 1)
        text = f"{tag} ({p[0]},{p[1]})"
        cv2.putText(canvas, text, (p[0] + 10, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, text, (p[0] + 10, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 1, cv2.LINE_AA)
    if title:
        caption(canvas, title, scale=0.7)
    return canvas
