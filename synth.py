"""Generate synthetic spawn images.

Real screenshots are the ground truth, but they arrive slowly and can't be
committed to a repo. Synthetic spawns let the whole pipeline -- segmentation,
badge OCR, frame classification, ranking -- be exercised deterministically in
tests and demos. The frames imitate the *chromatic* character of each tier
(flat and grey vs. vivid multi-hue), which is exactly what frame.py measures.
"""

from __future__ import annotations

import cv2
import numpy as np

from .models import FrameTier

CARD_W, CARD_H = 260, 400
GUTTER = 24


def _border_color(tier: FrameTier, t: float) -> tuple[int, int, int]:
    """BGR colour for border ring at depth t in 0..1."""
    if tier == FrameTier.HOLO:                       # full rainbow sweep
        hue = int((t * 180 * 2) % 180)
        hsv = np.uint8([[[hue, 235, 245]]])
    elif tier == FrameTier.RARE:                     # vivid two-hue sweep
        hue = int(95 + 45 * np.sin(t * np.pi * 2))
        hsv = np.uint8([[[hue, 215, 230]]])
    elif tier == FrameTier.UNCOMMON:                 # single saturated hue
        hsv = np.uint8([[[105, 170, 205]]])
    else:                                            # common: desaturated grey
        v = int(90 + 25 * t)
        return (v, v, v)
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0])


def draw_card(
    tier: FrameTier = FrameTier.COMMON,
    badge: str = "42",
    character: str = "CHARACTER",
    series: str = "SERIES",
    art_hue: int = 15,
    w: int = CARD_W,
    h: int = CARD_H,
) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Artwork: a soft vertical gradient so the interior isn't flat.
    art = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        hsv = np.uint8([[[art_hue % 180, 120, int(70 + 110 * y / h)]]])
        art[y, :] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    img[:] = art

    # Border ring, drawn as nested rectangles.
    thickness = max(6, int(min(w, h) * 0.11))
    for i in range(thickness):
        t = i / max(1, thickness - 1)
        cv2.rectangle(img, (i, i), (w - 1 - i, h - 1 - i), _border_color(tier, t), 1)

    # Badge: dark plate near the top-right with the print number (or E).
    bw, bh = 74, 40
    bx, by = w - bw - thickness - 8, thickness + 8
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (28, 24, 30), -1)
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (210, 210, 215), 2)
    scale = 1.15 if len(badge) <= 3 else 0.85
    (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.putText(img, badge, (bx + (bw - tw) // 2, by + (bh + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (245, 245, 245), 2, cv2.LINE_AA)

    # Name block near the bottom.
    yb = int(h * 0.80)
    overlay = img.copy()
    cv2.rectangle(overlay, (thickness, yb - 6), (w - thickness, yb + 54), (20, 18, 22), -1)
    img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
    for text, dy, sc in ((character, 18, 0.60), (series, 42, 0.48)):
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, sc, 1)
        cv2.putText(img, text, (max(thickness + 4, (w - tw) // 2), yb + dy),
                    cv2.FONT_HERSHEY_SIMPLEX, sc, (250, 250, 250), 1, cv2.LINE_AA)
    return img


def draw_spawn(specs: list[dict], bg: tuple[int, int, int] = (49, 51, 56)) -> np.ndarray:
    """Compose cards side by side on a Discord-ish dark background."""
    cards = [draw_card(**s) for s in specs]
    h = max(c.shape[0] for c in cards)
    w = sum(c.shape[1] for c in cards) + GUTTER * (len(cards) + 1)
    canvas = np.full((h + GUTTER * 2, w, 3), bg, dtype=np.uint8)
    x = GUTTER
    for c in cards:
        canvas[GUTTER:GUTTER + c.shape[0], x:x + c.shape[1]] = c
        x += c.shape[1] + GUTTER
    return canvas
