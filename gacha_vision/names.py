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
        # A component touching the very top row is clipped by the band crop,
        # not a self-contained glyph -- it's the character artwork bleeding
        # in above the text (hair, clothing, highlights). Real glyphs always
        # sit with margin below the crop edge; this is what actually
        # distinguishes the artwork-noise "line" that outranks the real
        # name from the real one, which plain span/count filters below let
        # through since noise can span most of the band's width too.
        if y == 0:
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


# Real letters in this font sit close enough to touch; even a wide word gap
# is nowhere near this. A row where every consecutive pair is farther apart
# than its own glyph height is never actual text -- it's isolated flecks of
# artwork (hair strands, cloth folds) that only coincidentally share a row
# with each other and pass the span check below by spreading wide.
DENSE_GAP_MULT = 1.0


def _has_dense_run(ln) -> bool:
    if len(ln) < 2:
        return True
    boxes = sorted(ln, key=lambda c: c[0])
    for a, b in zip(boxes, boxes[1:]):
        gap = b[0] - (a[0] + a[2])
        h = min(a[3], b[3])
        if h > 0 and gap <= DENSE_GAP_MULT * h:
            return True
    return False


# A handful of small, individually-dense clusters (each a few touching
# flecks of artwork) can still be scattered wide enough apart from each
# other to slip past `_has_dense_run` -- that only asks whether *a* tight
# pair exists anywhere in the row, not whether the whole row is one piece.
# A real line of text never has a gap this wide between any two neighbours
# (the widest genuine word gap measured in the corpus is ~2.8x); this is a
# straight reject, not a split -- splitting a line changes how many lines
# compete in `_line_score`'s threshold search and was measured to distort
# which threshold wins for *other*, unrelated cards.
MAX_INTERNAL_GAP = 4.0


def _no_huge_gap(ln) -> bool:
    boxes = sorted(ln, key=lambda c: c[0])
    if len(boxes) < 2:
        return True
    med_h = float(np.median([b[3] for b in boxes]))
    if med_h <= 0:
        return True
    return all(b[0] - (a[0] + a[2]) <= MAX_INTERNAL_GAP * med_h
               for a, b in zip(boxes, boxes[1:]))


def _looks_like_text(ln, w: int) -> bool:
    """Reject a row cluster that is just coincidentally-aligned artwork."""
    if len(ln) < MIN_GLYPHS_PER_LINE:
        return False
    span = (ln[-1][0] + ln[-1][2]) - ln[0][0]
    if span < MIN_LINE_SPAN * w:
        return False
    return _has_dense_run(ln) and _no_huge_gap(ln)


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


# segment_lines picks one threshold for the *whole* name block (scored on
# total glyphs across every line at once); measured directly, that is
# rarely the threshold that cuts any *one* line into its cleanest glyphs.
# Re-deriving glyphs from a threshold search scoped to just this line's own
# crop -- same search-and-score shape as segment_lines, just narrower --
# raised the corpus's exact-letter-count match rate several-fold over
# reusing segment_lines' boxes directly. `line` therefore only fixes the
# region to search; the boxes themselves come from `work_gray`.
LOCAL_THRESHOLD_PERCENTILES = (55, 60, 65, 70, 75, 80, 84, 88, 91, 93, 95, 96.5, 98, 99)
LINE_CROP_PAD = 6
# The winning threshold's glyphs must cover most of the line's own width --
# without this gate, a threshold lighting up only a couple of glyphs in the
# middle can still score as "clean" (perfectly even height) and win.
SPAN_COVERAGE_MIN = 0.5
# Weight on height-inconsistency when scoring a candidate threshold: glyphs
# on one line share a font size, so the threshold lighting up the line most
# *evenly* is preferred over the one lighting up the most glyphs.
HEIGHT_CV_WEIGHT = 15.0
# Column runs, not connected components: two ink blobs of one letter that
# don't touch 8-connected (this font's rounder lowercase letters can render
# that way once upscaled) still share every column between them, so a
# vertical ink projection keeps them one glyph where
# connectedComponentsWithStats would wrongly split them into two. No glyph
# is wider than this multiple of the line's median run width; a fused pair
# (touching letters with zero gap, which this font allows by design) runs
# well above it.
MAX_GLYPH_ASPECT = 2.0


def _line_region(work_gray: np.ndarray, line: list[tuple[int, int, int, int]]):
    """The line's bounding box, padded, as (x0, y0, cropped array)."""
    xs = [b[0] for b in line]
    xe = [b[0] + b[2] for b in line]
    ys = [b[1] for b in line]
    ye = [b[1] + b[3] for b in line]
    x0 = max(0, min(xs) - LINE_CROP_PAD)
    x1 = max(xe) + LINE_CROP_PAD
    y0 = max(0, min(ys) - LINE_CROP_PAD)
    y1 = max(ye) + LINE_CROP_PAD
    return x0, y0, work_gray[y0:y1, x0:x1]


def _run_height(ink: np.ndarray, x0: int, x1: int):
    """(y-offset, height) of the ink actually present in one column run."""
    rows = np.where(ink[:, x0:x1].any(axis=1))[0]
    if len(rows) == 0:
        return 0, 0
    return int(rows.min()), int(rows.max() - rows.min() + 1)


def _column_runs(region: np.ndarray, thr: float):
    """Ink-column runs at one threshold: (x0, x1) pairs, plus the ink mask.

    A run too small in area to be a real glyph -- upscaled dust, a stray
    antialiasing pixel -- is dropped here, before it can drag the line's
    median run width down and make `_split_wide_run` shred every
    genuinely wide glyph into a dozen slivers (measured: one card without
    this filter came back 97 "glyphs" for an 8-letter name).
    """
    ink = (region >= thr).astype(np.uint8)
    colsum = ink.sum(axis=0)
    is_gap = colsum <= 0
    runs = []
    x, w = 0, len(colsum)
    while x < w:
        if is_gap[x]:
            x += 1
            continue
        start = x
        while x < w and not is_gap[x]:
            x += 1
        _, h = _run_height(ink, start, x)
        if (x - start) * h >= MIN_GLYPH_AREA:
            runs.append((start, x))
    return runs, ink


def _split_wide_run(ink: np.ndarray, x0: int, x1: int, med_w: float):
    """Split a column run still holding two or more glyphs, at the ink valley."""
    bw = x1 - x0
    if med_w <= 0 or bw / med_w <= MAX_GLYPH_ASPECT:
        return [(x0, x1)]
    k = max(2, int(round(bw / med_w)))
    cols = ink[:, x0:x1].astype(np.float64).sum(axis=0)
    cuts = []
    for j in range(1, k):
        target = j * bw / k
        lo = int(max(1, target - 0.30 * bw / k))
        hi = int(min(bw - 1, target + 0.30 * bw / k))
        if hi > lo:
            cuts.append(lo + int(np.argmin(cols[lo:hi])))
    bounds = [0] + sorted(set(cuts)) + [bw]
    return [(x0 + a, x0 + b) for a, b in zip(bounds, bounds[1:]) if b - a >= 2]


def _pick_line_threshold(region: np.ndarray):
    """Search this line's own thresholds for the one giving the most evenly
    sized glyphs, gated on covering most of the line's width."""
    ref_span = region.shape[1]
    flat = region.reshape(-1).astype(np.float64)
    best_runs, best_ink, best_score = None, None, -1e18
    for pct in LOCAL_THRESHOLD_PERCENTILES:
        thr = float(np.percentile(flat, pct))
        runs, ink = _column_runs(region, thr)
        if len(runs) < 2:
            continue
        span = runs[-1][1] - runs[0][0]
        if span < SPAN_COVERAGE_MIN * ref_span:
            continue
        heights = [h for a, b in runs for h in [_run_height(ink, a, b)[1]] if h > 0]
        if len(heights) < 2:
            continue
        hs = np.array(heights, dtype=float)
        h_cv = hs.std() / max(hs.mean(), 1e-6)
        score = len(runs) - HEIGHT_CV_WEIGHT * h_cv
        if score > best_score:
            best_score, best_runs, best_ink = score, runs, ink
    return best_runs, best_ink


# A word space is a clear outlier against the letter spacing of the same
# line, so the cut point is derived per line rather than fixed. Both terms
# below are floors measured against the corpus, not the plan's originals:
# the multiple-of-median-gap term alone false-triggers within a word when
# letters sit naturally far apart (e.g. after an "i" or "l"), so it is
# combined with an absolute floor relative to glyph height.
GAP_GAP_MULTIPLE = 2.0
GAP_HEIGHT_FRACTION = 0.22


def split_line(work_gray: np.ndarray,
               line: list[tuple[int, int, int, int]]
               ) -> list[tuple[int, int, int, int] | None]:
    """Glyph boxes for one line, left to right, with None marking a space.

    `line`'s boxes only fix the region to search: segment_lines picks one
    threshold for the whole name block, which is rarely the threshold that
    cuts *this* line into its cleanest glyphs, so the glyphs themselves are
    re-derived from `work_gray` within that region.
    """
    if not line:
        return []
    x0, y0, region = _line_region(work_gray, line)
    runs, ink = _pick_line_threshold(region)
    if runs is None:
        # No local threshold cleared the coverage/height bar -- fall back to
        # segment_lines' own boxes rather than returning nothing.
        glyphs = [tuple(b) for b in sorted(line, key=lambda c: c[0])]
    else:
        widths = [b - a for a, b in runs]
        med_w = float(np.median(widths))
        glyphs = []
        for a, b in runs:
            for ga, gb in _split_wide_run(ink, a, b, med_w):
                yoff, h = _run_height(ink, ga, gb)
                if h > 0:
                    glyphs.append((x0 + ga, y0 + yoff, gb - ga, h))
        glyphs.sort(key=lambda g: g[0])

    if len(glyphs) < 2:
        return glyphs

    med_h = float(np.median([g[3] for g in glyphs]))
    gaps = [glyphs[i + 1][0] - (glyphs[i][0] + glyphs[i][2])
            for i in range(len(glyphs) - 1)]
    inter = float(np.median(gaps))
    cut = max(inter * GAP_GAP_MULTIPLE, GAP_HEIGHT_FRACTION * med_h)

    out = [glyphs[0]]
    for g, gap in zip(glyphs[1:], gaps):
        if gap >= cut:
            out.append(None)
        out.append(g)
    return out


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
