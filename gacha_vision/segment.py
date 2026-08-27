"""Locate the individual cards inside a spawn screenshot.

Strategies, tried in order by ``auto``:

  * **projection** -- cards sit in a row separated by flat background, so the
    per-column pixel variance drops to near zero in the gutters. Splitting on
    those gutters is far more reliable than looking for card outlines: a
    common (grey, low-contrast) frame produces no closed contour at all, and
    Canny returns only fragments of the artwork inside it.
  * **contour** -- bounding boxes of tall portrait rectangles. Kept as a
    fallback for busy backgrounds where no clean gutter exists.
  * **columns** -- assume a tight row of ``expected`` cards and split evenly.
    Dumb but total; the last resort, and correct for synthetic fixtures.
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


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous [start, end) runs of True in a 1-D boolean mask."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def _span(energy: np.ndarray, frac: float) -> tuple[int, int]:
    """Extent of the non-background part of a 1-D energy profile."""
    thr = energy.min() + frac * (energy.max() - energy.min())
    idx = np.flatnonzero(energy > thr)
    return (int(idx[0]), int(idx[-1]) + 1) if idx.size else (0, len(energy))


def projection_split(
    bgr: np.ndarray,
    frac: float = 0.15,
    min_width: float = 0.08,
) -> list[Box]:
    """Split a row of cards on the low-variance gutters between them.

    Returns [] when the image has no clean gutter structure, so the caller
    can fall back rather than trusting a bad split.
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    col = gray.std(axis=0)
    if col.max() - col.min() < 1e-3:
        return []
    thr = col.min() + frac * (col.max() - col.min())
    spans = [(a, b) for a, b in _runs(col > thr) if (b - a) >= min_width * w]
    if not spans:
        return []

    # Cards in one spawn are the same size; wildly uneven spans mean this is
    # not a clean row and the split should not be trusted.
    widths = sorted(b - a for a, b in spans)
    median = widths[len(widths) // 2]
    if any(abs((b - a) - median) > 0.35 * median for a, b in spans):
        return []

    boxes: list[Box] = []
    for x0, x1 in spans:
        row = gray[:, x0:x1].std(axis=1)
        y0, y1 = _span(row, frac)
        boxes.append((x0, y0, x1 - x0, y1 - y0))
    return boxes


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

    if layout in ("auto", "projection"):
        proj = projection_split(bgr)
        if proj and (expected is None or len(proj) == expected):
            return proj
        if layout == "projection":
            return proj or column_split(w, h, expected or 2)

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
