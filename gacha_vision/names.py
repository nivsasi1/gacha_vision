"""Read the character and series names off a card.

Same shape of problem as the badge, and the same answer: the text is crisp
white ink in a fixed font, and what defeats a general OCR engine is that the
ink sits over arbitrary artwork at a brightness that moves card to card. So
the threshold is searched rather than assumed, and each glyph is matched
against an atlas built from labelled real cards.

Where this font actually differs from the badge -- and from names.py's own
first attempt -- is that cutting a line into per-glyph boxes before
recognising them has a 55-62% ceiling here (see
docs/superpowers/plans/2026-08-27-name-reader.md's 2026-08-28 revision):
some letters are only single components at a threshold that fuses others on
the same line, so no per-line threshold exists. `segment_lines`/`split_line`
below still find where each *line* is and seed the atlas
(tools/build_glyph_atlas.py), but a glyph's *identity* is decided by
`names_align.free_align`: dynamic programming that chooses the character
sequence itself, never committing to a cut point ahead of time. See
`read_line_free` below.

Unlike the badge, the line count is not fixed: the character name takes one
line and the series wraps over one to three more.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .names_align import GAP_MIN_FRAC, LINE_HEIGHT, free_align, normalise_line

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


# --------------------------------------------------------------------------
# reading -- Task R2
#
# free_align decides *where* the glyphs are; classifying each one -- what it
# actually is -- is kept separate and happens here, against the full,
# unblurred atlas by nearest-neighbour (free_align only sees per-class
# *means*, blurry by design so the width sweep stays affordable -- see its
# docstring). Same split digits.py doesn't need (its glyphs are already
# isolated by component before classification) but this font does, since
# nothing here is told in advance how many glyphs a line has or where they
# start.

_ATLAS_PATH = Path(__file__).parent / "data" / "glyph_atlas.npz"

# Same crop tools/build_name_fixture.py used to build the name_bands.npz
# fixture read_names_from_band is measured against -- must match exactly, or
# read_names would be validated against a different region than the one it
# actually reads at inference time. (tools/name_montage.py's labelling
# helper independently uses 0.55; that mismatch predates this task -- see
# .superpowers/sdd/progress.md's Task 1 notes -- and is out of scope here,
# since name_montage.py never produced a fixture anything is measured
# against.)
BAND_TOP = 0.60

# Below this, a read is more likely wrong than right -- return nothing
# rather than a guess, same MIN_TRUSTED_MATCH precedent as digits.py
# (read_print_number_from_window).
#
# digits.py calibrates this by finding the gap between correct reads'
# confidence and incorrect ones'. That method doesn't apply here: the
# leave-one-card-out run this was calibrated against (task-R2-report.md)
# has zero character-name exact matches out of 181 cards, so there is no
# "correct" population to measure a floor against, and confidence barely
# correlates with actual accuracy in the wrong population either (r=0.16
# against CER; the single highest-confidence read in the whole corpus,
# 0.994, was one of the worst -- CER 9.0 on a 3-letter name). Confidence
# here mostly reflects the free decoder having found many short spans that
# each individually look confident, not whether the read is right.
#
# With no signal to calibrate against, 0.995 -- just above every confidence
# observed in that run (max 0.994) -- is a deliberate "trust nothing"
# floor, not a fitted separator: it reflects the measured reality that this
# reader is not accurate enough yet for any of its own confidence scores to
# be worth acting on. See task-R2-report.md before ever lowering this.
MIN_TRUSTED_MATCH = 0.995


@dataclass
class NameRead:
    """A card's character and series names, plus how much to trust them."""
    character: str = ""
    series: str = ""
    confidence: float = 0.0


def atlas_samples() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw atlas templates, their labels, and the card each came from --
    same shape and meaning as digits.atlas_samples()."""
    with np.load(_ATLAS_PATH) as z:
        return z["templates"], z["labels"], z["card"]


def _normalise_rows(tpl: np.ndarray) -> np.ndarray:
    X = tpl.reshape(len(tpl), -1).astype(np.float32)
    X -= X.mean(axis=1, keepdims=True)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-6)
    return X


@functools.lru_cache(maxsize=1)
def _full_atlas():
    tpl, labels, card = atlas_samples()
    return tpl, labels, card, _normalise_rows(tpl)


def _atlas_view(exclude_card: str | None):
    """`(templates, labels, X)` with one card's own samples dropped -- the
    leave-one-card-out testing seam. `exclude_card=None`, the production
    default, uses every sample in the atlas."""
    tpl, labels, card, X = _full_atlas()
    if not exclude_card:
        return tpl, labels, X
    keep = card != exclude_card
    return tpl[keep], labels[keep], X[keep]


def _class_mean_templates(tpl: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    """One mean bitmap per class, for free_align's segmentation search."""
    return {str(c): tpl[labels == c].astype(np.float32).mean(axis=0)
            for c in np.unique(labels)}


def _glyph_vector(line_img: np.ndarray, x0: int, x1: int, shape: tuple[int, int]
                   ) -> np.ndarray | None:
    """Contrast-normalise and resize one decided glyph span to the atlas's
    own storage shape -- the same recipe digits._glyph_vector uses."""
    gh, gw = shape
    sub = line_img[:, x0:x1]
    if sub.size == 0:
        return None
    sub = cv2.normalize(sub, None, 0, 255, cv2.NORM_MINMAX)
    sub = cv2.resize(sub, (gw, gh), interpolation=cv2.INTER_AREA)
    v = sub.reshape(-1).astype(np.float32)
    v -= v.mean()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else None


def read_line_free(line_img: np.ndarray, *, exclude_card: str | None = None
                    ) -> tuple[str, list[float]]:
    """Free-running read of one `normalise_line`-shaped line: segment and
    classify in a single pass, with no known text to check against.

    Returns the line's text -- a space inserted wherever a decided gap is
    wide enough to be a word break rather than ordinary letter spacing,
    reusing names_align's own measured floor for that -- and the per-glyph
    nearest-neighbour scores behind it, for the caller to average into an
    overall read confidence.
    """
    tpl, labels, X = _atlas_view(exclude_card)
    if len(tpl) == 0:
        return "", []
    shape = (tpl.shape[1], tpl.shape[2])
    spans = free_align(line_img, _class_mean_templates(tpl, labels))
    if not spans:
        return "", []

    gap_min_px = GAP_MIN_FRAC * LINE_HEIGHT
    chars: list[str] = []
    scores: list[float] = []
    for i, (x0, x1) in enumerate(spans):
        if i > 0 and x0 - spans[i - 1][1] >= gap_min_px:
            chars.append(" ")
        v = _glyph_vector(line_img, x0, x1, shape)
        if v is None:
            continue
        sims = X @ v
        j = int(np.argmax(sims))
        chars.append(str(labels[j]))
        scores.append(float(sims[j]))
    return "".join(chars).strip(), scores


def _read_names_raw(band_gray: np.ndarray, *, exclude_card: str | None = None) -> NameRead:
    """The reader's actual output, with no trust floor applied -- what
    test_names.py's leave-one-card-out accuracy measurement scores. Kept
    separate from `read_names_from_band` because gating a read to "" can
    only ever match or worsen its edit distance to the truth (an empty
    guess costs a full-length edit; a wrong-but-close guess usually costs
    less), so measuring accuracy *through* the trust floor would reward
    lowering it to nothing -- exactly backwards from what the floor is for.
    `read_names_from_band` is the one callers should use; this one is what
    the corpus measurement calls.
    """
    work = prepare_band(band_gray)
    lines = segment_lines(work)
    if not lines:
        return NameRead()

    character, char_scores = read_line_free(
        normalise_line(work, lines[0]), exclude_card=exclude_card)
    parts: list[str] = []
    series_scores: list[float] = []
    for ln in lines[1:]:
        norm = normalise_line(work, ln)
        if norm.size == 0:
            continue
        text, scores = read_line_free(norm, exclude_card=exclude_card)
        if text:
            parts.append(text)
            series_scores.extend(scores)

    all_scores = char_scores + series_scores
    confidence = round(float(np.mean(all_scores)), 3) if all_scores else 0.0
    return NameRead(character, " ".join(parts), confidence)


def read_names_from_band(band_gray: np.ndarray, *, exclude_card: str | None = None
                          ) -> NameRead:
    """Read the character name (line 0) and series (the remaining lines,
    joined) off an already-cropped name band.

    `exclude_card` drops one card's own templates from the atlas before
    matching -- the leave-one-card-out testing seam threaded down from here
    into `read_line_free`; production callers never pass it.
    """
    got = _read_names_raw(band_gray, exclude_card=exclude_card)
    if got.confidence < MIN_TRUSTED_MATCH:
        return NameRead(confidence=got.confidence)
    return got


def read_names(card_bgr: np.ndarray, *, exclude_card: str | None = None) -> NameRead:
    """Read the character and series names off a whole card image."""
    h = card_bgr.shape[0]
    band = cv2.cvtColor(card_bgr[int(BAND_TOP * h):, :], cv2.COLOR_BGR2GRAY)
    return read_names_from_band(band, exclude_card=exclude_card)
