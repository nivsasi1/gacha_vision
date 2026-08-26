"""Generate synthetic spawn images.

Real screenshots are the ground truth, but they arrive slowly and cannot be
committed to a repo. Synthetic spawns let the whole pipeline -- segmentation,
badge OCR, frame corroboration, ranking -- be exercised deterministically in
tests and demos.

The two frames imitate the real ones: ``NORMAL`` is a thin pale border with a
light bar along the bottom, ``E`` is a thick gold/bronze one with rainbow
corner orbs. Note which is which -- the *ornate* frame is the one carrying no
print number, which is the opposite of what decoration usually signals.
"""

from __future__ import annotations

import cv2
import numpy as np

from .models import FrameTier

CARD_W, CARD_H = 260, 400
GUTTER = 24


def _border_color(tier: FrameTier, t: float) -> tuple[int, int, int]:
    """BGR colour for the border ring at depth t in 0..1."""
    if tier == FrameTier.E:                # ornate gold sweeping to bronze
        hsv = np.uint8([[[int(14 + 12 * t), 205, int(150 + 85 * t)]]])
    elif tier == FrameTier.OTHER:          # an unfamiliar frame: full rainbow
        hsv = np.uint8([[[int((t * 360) % 180), 235, 245]]])
    else:                                  # NORMAL: thin, pale, low saturation
        hsv = np.uint8([[[int(120 + 30 * t), 60, 225]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0])


def _thickness(tier: FrameTier, w: int, h: int) -> int:
    """The E frame is visibly heavier than the plain one."""
    frac = 0.11 if tier in (FrameTier.E, FrameTier.OTHER) else 0.045
    return max(4, int(min(w, h) * frac))


def draw_card(
    tier: FrameTier = FrameTier.NORMAL,
    badge: str = "42",
    character: str = "CHARACTER",
    series: str = "SERIES",
    art_hue: int = 15,
    w: int = CARD_W,
    h: int = CARD_H,
    badge_h: int = 40,
) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Artwork: a soft vertical gradient so the interior isn't flat.
    art = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        hsv = np.uint8([[[art_hue % 180, 120, int(70 + 110 * y / h)]]])
        art[y, :] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    img[:] = art

    # Border ring, drawn as nested rectangles.
    thickness = _thickness(tier, w, h)
    for i in range(thickness):
        t = i / max(1, thickness - 1)
        cv2.rectangle(img, (i, i), (w - 1 - i, h - 1 - i), _border_color(tier, t), 1)

    if tier in (FrameTier.E, FrameTier.OTHER):
        # Rainbow corner orbs, the E frame's signature.
        for (cx, cy) in ((thickness, thickness), (w - thickness, thickness),
                         (thickness, h - thickness), (w - thickness, h - thickness)):
            for r in range(int(thickness * 1.3), 0, -1):
                hsv = np.uint8([[[int((r * 10) % 180), 245, 250]]])
                cv2.circle(img, (cx, cy), r,
                           tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]), 1)
    else:
        # NORMAL wears a pale bar along the bottom edge.
        cv2.rectangle(img, (thickness, h - thickness - 14),
                      (w - thickness, h - thickness), (238, 226, 240), -1)

    # The hook icon that sits immediately left of the badge on every real
    # card, sharing its row. Present in the fixtures because it is the single
    # thing that broke badge reading in the field: it is fixed art, so it
    # OCRs to the same glyph every time, and being on the badge's row it was
    # swept into the same crop as the digits.
    hook_h = max(6, int(badge_h * 0.62))
    hx = w - int(badge_h * 1.85) - int(min(w, h) * 0.11) - 8 - int(hook_h * 1.5)
    hy = int(min(w, h) * 0.11) + 8 + (badge_h - hook_h) // 2
    cv2.line(img, (hx + hook_h // 2, hy), (hx + hook_h // 2, hy + hook_h),
             (235, 235, 240), max(1, hook_h // 7), cv2.LINE_AA)
    cv2.ellipse(img, (hx + hook_h // 2, hy + max(2, hook_h // 5)),
                (max(2, hook_h // 3), max(2, hook_h // 4)), 0, 180, 360,
                (235, 235, 240), max(1, hook_h // 7), cv2.LINE_AA)

    # Badge: dark plate near the top-right with the print number (or E).
    # badge_h is a knob because the real cards wear a much smaller badge than
    # the first synthetic ones did, and that difference turned out to matter.
    bh = max(8, badge_h)
    bw = int(bh * 1.85)
    bx, by = w - bw - thickness - 8, thickness + 8
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (28, 24, 30), -1)
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (210, 210, 215), 2)
    scale = (1.15 if len(badge) <= 3 else 0.85) * (bh / 40.0)
    thick = max(1, int(round(2 * bh / 40.0)))
    (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.putText(img, badge, (bx + (bw - tw) // 2, by + (bh + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (245, 245, 245), thick, cv2.LINE_AA)

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
