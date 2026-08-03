"""Box drawing, captions, and the before/after comparison layouts."""

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX

BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 0, 255)
GT_COLOR = (0, 200, 255)

CORNER_TAGS = ("TL", "TR", "BR", "BL")
_CORNER_COLORS = ((0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 128, 0))

# 2x2 panel: captions sit on the background below each tile, never over the image.
SUBFIG_TAGS = ("(a)", "(b)", "(c)", "(d)")
PANEL_BG = (255, 255, 255)
PANEL_TEXT = (32, 32, 32)
PANEL_GAP = 12
PANEL_LABEL_H = 34
PANEL_FONT_SCALE = 0.55

# Width of the arrow column between the two tiles of the before/after warp figure.
WARP_ARROW_W = 80


def draw_detections(img, detections, color=BOX_COLOR, text_color=TEXT_COLOR,
                    thickness=2) -> np.ndarray:
    """Draws in place and returns the image; pass a copy to keep a clean original."""
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(img, f"{d.name} {d.conf:.2f}", (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    return img


def draw_ground_truth(img, boxes, names=None, color=GT_COLOR) -> np.ndarray:
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
    if left.shape[:2] != right.shape[:2]:
        right = cv2.resize(right, (left.shape[1], left.shape[0]))
    return cv2.hconcat([caption(left.copy(), left_label),
                        caption(right.copy(), right_label)])


def panel_size(tile_w, tile_h):
    """Canvas size grid_2x2 produces for tiles of this size, as (width, height)."""
    return (tile_w * 2 + PANEL_GAP * 3,
            (tile_h + PANEL_LABEL_H) * 2 + PANEL_GAP * 3)


def grid_2x2(top_left, top_right, bottom_left, bottom_right, labels=None) -> np.ndarray:
    """
    2x2 comparison panel, laid out like a paper figure.

    Tiles are resized onto the first one and captioned "(a) …" on the background
    strip below each, so no text ever covers image content.
    """
    h, w = top_left.shape[:2]
    canvas_w, canvas_h = panel_size(w, h)
    canvas = np.full((canvas_h, canvas_w, 3), PANEL_BG, np.uint8)

    for i, img in enumerate((top_left, top_right, bottom_left, bottom_right)):
        tile = img if img.shape[:2] == (h, w) else cv2.resize(img, (w, h))
        x = PANEL_GAP + (i % 2) * (w + PANEL_GAP)
        y = PANEL_GAP + (i // 2) * (h + PANEL_LABEL_H + PANEL_GAP)
        canvas[y:y + h, x:x + w] = tile

        text = SUBFIG_TAGS[i]
        if labels and i < len(labels) and labels[i]:
            text = f"{text} {labels[i]}"
        (text_w, text_h), _ = cv2.getTextSize(text, FONT, PANEL_FONT_SCALE, 1)
        cv2.putText(canvas, text,
                    (x + max(0, (w - text_w) // 2),
                     y + h + (PANEL_LABEL_H + text_h) // 2),
                    FONT, PANEL_FONT_SCALE, PANEL_TEXT, 1, cv2.LINE_AA)
    return canvas


def draw_quad(frame, points, title=None, thickness=2) -> np.ndarray:
    """
    Calibration quad on a copy of `frame`, one colour per corner plus coordinates.

    Drawn on the raw pre-warp camera view, which is what makes a mis-ordered or
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


def warp_before_after(pre, post, points=None, tile_w=640, labels=None) -> np.ndarray:
    """
    The raw camera view next to the rectified one, with the arrow between them.

    The two rarely share a resolution - one is the camera sensor, the other the
    model input - so each tile keeps its own aspect ratio and the sizes go into the
    captions. `points` draws the calibration quad on the left tile, which is what
    makes a mis-ordered or drifted warp obvious.
    """
    left = draw_quad(pre, points) if points else pre.copy()
    right = post.copy()
    if labels is None:
        labels = (f"(a) pre-warp camera {pre.shape[1]}x{pre.shape[0]}",
                  f"(b) post-warp {post.shape[1]}x{post.shape[0]}")

    tiles = []
    for img in (left, right):
        scale = tile_w / img.shape[1]
        tiles.append(cv2.resize(img, (tile_w, max(1, round(img.shape[0] * scale)))))
    tile_h = max(t.shape[0] for t in tiles)

    canvas_w = PANEL_GAP * 2 + tile_w * 2 + WARP_ARROW_W
    canvas_h = PANEL_GAP * 2 + tile_h + PANEL_LABEL_H
    canvas = np.full((canvas_h, canvas_w, 3), PANEL_BG, np.uint8)

    for i, tile in enumerate(tiles):
        x = PANEL_GAP + i * (tile_w + WARP_ARROW_W)
        y = PANEL_GAP + (tile_h - tile.shape[0]) // 2
        canvas[y:y + tile.shape[0], x:x + tile_w] = tile

        text = labels[i] if i < len(labels) else ""
        (text_w, text_h), _ = cv2.getTextSize(text, FONT, PANEL_FONT_SCALE, 1)
        cv2.putText(canvas, text,
                    (x + max(0, (tile_w - text_w) // 2),
                     PANEL_GAP + tile_h + (PANEL_LABEL_H + text_h) // 2),
                    FONT, PANEL_FONT_SCALE, PANEL_TEXT, 1, cv2.LINE_AA)

    mid_x, mid_y = PANEL_GAP + tile_w, PANEL_GAP + tile_h // 2
    cv2.arrowedLine(canvas, (mid_x + 14, mid_y), (mid_x + WARP_ARROW_W - 14, mid_y),
                    PANEL_TEXT, 3, cv2.LINE_AA, tipLength=0.4)
    cv2.putText(canvas, "warp", (mid_x + 14, mid_y - 14), FONT, 0.5, PANEL_TEXT, 1,
                cv2.LINE_AA)
    return canvas
