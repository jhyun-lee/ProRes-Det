"""Box drawing, captions, and the before/after comparison layouts."""

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Detections: a red box with the class name on a filled white tab above it. The tab
# is what keeps the name readable over a bright projection, where plain text is not.
BOX_COLOR = (0, 0, 255)
LABEL_BG = (255, 255, 255)
LABEL_TEXT = (0, 0, 0)
LABEL_SCALE = 0.5
LABEL_WEIGHT = 1
LABEL_PAD = 3

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

# Bottom bar carrying the key bindings, on the figures that are also a prompt.
HINT_BAR_H = 30
HINT_TEXT = (120, 60, 0)


def draw_detections(img, detections, color=BOX_COLOR, thickness=2,
                    scale=LABEL_SCALE) -> np.ndarray:
    """
    One box per detection, class name on a filled tab centred above it.

    Only what the detector found is drawn; ground truth is never overlaid, so the
    annotated view stays a picture of the prediction alone. Confidence is left out
    of the label to keep it short - every score is in detections.csv.

    Draws in place and returns the image; pass a copy to keep a clean original.
    """
    h, w = img.shape[:2]
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        (text_w, text_h), baseline = cv2.getTextSize(d.name, FONT, scale, LABEL_WEIGHT)
        tab_w = text_w + LABEL_PAD * 2
        tab_h = text_h + baseline + LABEL_PAD * 2

        # Centred on the box, then pulled back inside the frame. A name wider than
        # its own box is normal, and a box against an edge must not push its label
        # off-image; a box near the top gets its label below the edge instead.
        x = min(max(x1 + (x2 - x1 - tab_w) // 2, 0), max(w - tab_w, 0))
        y = y1 - tab_h if y1 - tab_h >= 0 else y1
        y = min(y, max(h - tab_h, 0))

        cv2.rectangle(img, (x, y), (x + tab_w, y + tab_h), LABEL_BG, -1)
        cv2.putText(img, d.name, (x + LABEL_PAD, y + LABEL_PAD + text_h),
                    FONT, scale, LABEL_TEXT, LABEL_WEIGHT, cv2.LINE_AA)
    return img


def caption(img, text, org=(8, 22), scale=0.6) -> np.ndarray:
    """White text with a black outline, readable over any content."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3,
                cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1,
                cv2.LINE_AA)
    return img


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


def warp_before_after(pre, post, points=None, tile_w=640, labels=None,
                      hint=None) -> np.ndarray:
    """
    The raw camera view next to the rectified one, with the arrow between them.

    The two rarely share a resolution - one is the camera sensor, the other the
    model input - so each tile keeps its own aspect ratio and the sizes go into the
    captions. `points` draws the calibration quad on the left tile, which is what
    makes a mis-ordered or drifted warp obvious.

    `hint` adds a bar of key bindings along the bottom. Prompts that block the run
    need it: the operator is at the rig looking at this window, and the console line
    saying what to press is behind the fullscreen projector.
    """
    # Length, not truthiness: `points` is routinely a numpy (4, 2), and testing an
    # array for truth raises rather than being falsy.
    left = draw_quad(pre, points) if points is not None and len(points) else pre.copy()
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
    canvas_h = PANEL_GAP * 2 + tile_h + PANEL_LABEL_H + (HINT_BAR_H if hint else 0)
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

    if hint:
        cv2.putText(canvas, hint, (PANEL_GAP, canvas_h - HINT_BAR_H // 3), FONT, 0.6,
                    HINT_TEXT, 1, cv2.LINE_AA)
    return canvas
