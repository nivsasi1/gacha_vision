"""Locate the individual cards inside a spawn screenshot.

Two strategies:
  * ``auto``    -- contour detection: find tall portrait rectangles that look
                   like cards. Works on tight crops and full Discord grabs.
  * ``columns`` -- assume the image is a tight row of ``expected`` cards and
                   split it into equal columns. Dumb but reliable; a good
                   fallback and the right choice for synthetic fixtures.
"""

from __future__ import annotations

import cv2
import numpy as np

Box = tuple[int, int, int, int]  # x, y, w, h


def _overlap_ratio(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix, iy = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, ix2 - ix) * max(0, iy2 - iy)
    smaller = min(aw * ah, bw * bh)
    return inter / smaller if smaller else 0.0


def _dedupe(boxes: list[Box], thresh: float = 0.55) -> list[Box]:
    """Keep larger boxes; drop those largely covered by an already-kept one."""
    kept: list[Box] = []
    for b in sorted(boxes, key=lambda x: x[2] * x[3], reverse=True):
        if all(_overlap_ratio(b, k) < thresh for k in kept):
            kept.append(b)
    return kept


def column_split(width: int, height: int, n: int, pad_frac: float = 0.0) -> list[Box]:
    n = max(1, n)
    pad = int(width * pad_frac)
    inner = width - pad * (n + 1)
    cw = inner // n
    return [(pad + i * (cw + pad), 0, cw, height) for i in range(n)]


def find_cards(
    bgr: np.ndarray,
    expected: int | None = None,
    layout: str = "auto",
) -> list[Box]:
    h, w = bgr.shape[:2]

    if layout == "columns":
        return column_split(w, h, expected or 2)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 130)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: list[Box] = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if ch < 0.30 * h or cw < 0.06 * w:      # cards are tall and not slivers
            continue
        if cw * ch < 0.03 * w * h:               # and a real chunk of the frame
            continue
        aspect = cw / ch
        if not (0.45 <= aspect <= 1.05):         # portrait-ish
            continue
        boxes.append((x, y, cw, ch))

    boxes = _dedupe(boxes)
    boxes.sort(key=lambda b: b[0])               # left -> right == slot order

    # If detection disagrees with a known count, trust the count.
    if expected and len(boxes) != expected:
        return column_split(w, h, expected)
    if not boxes:
        return column_split(w, h, expected or 2)
    return boxes
