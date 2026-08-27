"""Read the character and series names off a card.

Same shape of problem as the badge, and the same answer. The text is crisp
white ink in a fixed font; what defeats a general OCR engine is that the ink
sits over arbitrary artwork at a brightness that moves card to card. So the
threshold is searched rather than assumed, glyphs are cut out by component,
and each one is matched against an atlas built from labelled real cards.

Unlike the badge, the line count is not fixed: the character name takes one
line and the series wraps over one to three more.
"""

from __future__ import annotations

import cv2
import numpy as np

SCALE = 4
# Unlike the badge plate, the name band sits directly over arbitrary artwork,
# so its brightness swings far more from card to card -- one card's 90th
# percentile is already at 255, another's median is 85. A fixed absolute
# grayscale cut can't follow that; cutting at percentiles of each band's own
# distribution does.
THRESHOLD_PERCENTILES = (80, 84, 88, 91, 93, 95, 96.5, 98, 99)
MIN_GLYPH_AREA = 18
# A stray fleck of artwork can form a component that looks like a glyph in
# isolation. A real line of text is never a single letter -- the shortest
# strings in the corpus are two characters ("DC") -- so a "line" with only
# one box is noise, not text, and is dropped before scoring.
MIN_GLYPHS_PER_LINE = 2
# A genuine text line spans a good fraction of the band's width; a cluster of
# noise fused together by coincidence stays narrow.
MIN_LINE_SPAN = 0.18


def prepare_band(band_gray: np.ndarray) -> np.ndarray:
    """Upscale a name band to the size the thresholds are tuned for."""
    return cv2.resize(band_gray, None, fx=SCALE, fy=SCALE,
                      interpolation=cv2.INTER_CUBIC)


def _components(work: np.ndarray, thr: float) -> list[tuple[int, int, int, int]]:
    """Candidate glyph boxes visible at one threshold."""
    h, w = work.shape
    ink = (work >= thr).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[i][:5])
        if area < MIN_GLYPH_AREA or bh < 0.05 * h or bh > 0.30 * h:
            continue
        if bw > 0.40 * w:
            continue
        out.append((x, y, bw, bh))
    return out


def _split_fused(work: np.ndarray, box: tuple[int, int, int, int],
                  max_aspect: float = 0.85) -> list[tuple[int, int, int, int]]:
    """Split a component still holding two or more glyphs, at the ink valley."""
    x, y, bw, bh = box
    if bh <= 0 or bw / bh <= max_aspect:
        return [box]
    k = max(2, int(round(bw / (0.62 * bh))))
    sub = work[y:y + bh, x:x + bw]
    ink = (sub >= np.percentile(sub, 55)).astype(np.float64)
    columns = ink.sum(axis=0)
    cuts = []
    for j in range(1, k):
        target = j * bw / k
        lo = int(max(1, target - 0.30 * bw / k))
        hi = int(min(bw - 1, target + 0.30 * bw / k))
        if hi > lo:
            cuts.append(lo + int(np.argmin(columns[lo:hi])))
    bounds = [0] + sorted(set(cuts)) + [bw]
    out = []
    for a, b in zip(bounds, bounds[1:]):
        if b - a >= 2:
            out.append((x + a, y, b - a, bh))
    return out if out else [box]


def _group_rows(boxes, tol: float = 0.45):
    """Cluster boxes into text lines by vertical overlap."""
    lines: list[list[tuple[int, int, int, int]]] = []
    for b in sorted(boxes, key=lambda c: c[1] + c[3] / 2):
        cy, bh = b[1] + b[3] / 2, b[3]
        for ln in lines:
            ry = sum(o[1] + o[3] / 2 for o in ln) / len(ln)
            rh = sum(o[3] for o in ln) / len(ln)
            if abs(cy - ry) <= tol * max(rh, bh):
                ln.append(b)
                break
        else:
            lines.append([b])
    for ln in lines:
        ln.sort(key=lambda c: c[0])
    lines.sort(key=lambda ln: sum(o[1] for o in ln) / len(ln))
    return lines


def _looks_like_text(ln, w: int) -> bool:
    """Reject a row cluster that is just coincidentally-aligned artwork."""
    if len(ln) < MIN_GLYPHS_PER_LINE:
        return False
    span = (ln[-1][0] + ln[-1][2]) - ln[0][0]
    return span >= MIN_LINE_SPAN * w


def _line_score(lines, w: int) -> float:
    """Prefer a threshold giving many glyphs of consistent height per line."""
    if not lines:
        return -1e9
    total = sum(len(ln) for ln in lines)
    hs = np.array([b[3] for ln in lines for b in ln], dtype=float)
    height_penalty = 30.0 * float(hs.std() / max(hs.mean(), 1e-6)) if len(hs) >= 2 else 0.0
    # A cluttered threshold fragments each real line into several noisy
    # slivers; penalise line count beyond what a name block actually has
    # (one for the character, up to three for a wrapped series).
    excess_lines = max(0, len(lines) - 4)
    return total - height_penalty - 14.0 * excess_lines


def segment_lines(work_gray: np.ndarray) -> list[list[tuple[int, int, int, int]]]:
    """Glyph-component boxes of the name block, grouped into text lines.

    Lines are ordered top to bottom, each line's boxes left to right.
    """
    h, w = work_gray.shape
    flat = work_gray.reshape(-1).astype(np.float64)
    best, best_score = [], -1e18
    for pct in THRESHOLD_PERCENTILES:
        thr = float(np.percentile(flat, pct))
        raw = _components(work_gray, thr)
        split = [g for box in raw for g in _split_fused(work_gray, box)]
        lines = [ln for ln in _group_rows(split) if _looks_like_text(ln, w)]
        sc = _line_score(lines, w)
        if sc > best_score:
            best_score, best = sc, lines
    return best
