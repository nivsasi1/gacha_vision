"""Read the print number off a card's badge.

Tesseract does not do this job well -- 38 of 81 real badges -- and the reason
is not the glyphs, which are crisp white numerals in a fixed game font. It is
that a general OCR engine has to be handed a clean crop, and producing one is
the hard part here: the badge plate is semi-transparent over the card artwork,
so its brightness moves from card to card, and any single global threshold
either fuses neighbouring digits into one blob or dissolves the thin ones.

So this module does not threshold once. It searches, and scores each result
against what a badge must look like:

    one to four glyphs, equal height, on a common baseline, evenly spaced

which is a strong enough shape that the right cut point identifies itself. A
blob still holding two glyphs is then split -- no digit in this font is wider
than about 0.72 of its height, so anything wider is two -- and each glyph is
matched against an atlas of examples taken from labelled real cards.

Ten rigid classes and a few hundred examples make nearest-neighbour the right
classifier, and it reads every glyph in the corpus correctly when the card it
came from is held out. It is also about a hundred times faster than shelling
out to tesseract per candidate crop.
"""

from __future__ import annotations

import functools
from pathlib import Path

import cv2
import numpy as np

# The badge always sits in the card's top-right. Generous on both axes: this
# only has to contain the plate, the search below finds it inside.
WIN_TOP = 0.22
WIN_LEFT = 0.45
# Work upscaled, so a 12-pixel-tall glyph has enough rows to split and match.
SCALE = 5

# Cut points to try. Below this range the artwork floods in, above it the
# strokes dissolve; the score picks the one that lands on the digits.
THRESHOLDS = tuple(range(228, 254, 2))

MAX_DIGITS = 4
# No digit in this font is wider than ~0.72 of its height. A fused pair runs
# 0.9 and up. Width alone cannot tell them apart -- a '1' is half the width of
# a '0', so "15" fused is barely wider than a lone '0' -- but the ratio can.
MAX_DIGIT_ASPECT = 0.78

# Above this the atlas is answering about the font it was built from. Real
# cards read at 0.95 and up with their own card held out; the one corpus
# misread sat at 0.84, and a font the atlas has never seen scores 0.69 or
# less. Below the floor the caller should ask tesseract instead.
MIN_TRUSTED_MATCH = 0.90

GLYPH_W, GLYPH_H = 16, 24
_ATLAS_PATH = Path(__file__).parent / "data" / "digit_atlas.npz"


# --------------------------------------------------------------------------
# segmentation


def _rows_at(gray: np.ndarray, thr: int) -> list[list[tuple]]:
    """Candidate glyph rows visible at one threshold."""
    h, w = gray.shape
    ink = (gray >= thr).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, 8)

    parts = []
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[i][:5])
        if area < 25 or bh < 0.07 * h or bh > 0.55 * h:
            continue
        if bw > 4.5 * bh:       # the plate's specular top edge
            continue
        if bh > 9.0 * bw:       # a sliver of card frame ('1' is nearly this thin)
            continue
        parts.append((x, y, bw, bh, area))

    rows = []
    for seed in parts:
        cy = seed[1] + seed[3] / 2
        row = [c for c in parts
               if abs((c[1] + c[3] / 2) - cy) <= 0.35 * seed[3]
               and 0.62 * seed[3] <= c[3] <= 1.55 * seed[3]]
        row.sort(key=lambda c: c[0])
        if row and row not in rows:
            rows.append(row)
    return rows


def _score(row: list[tuple], h: int, w: int) -> float:
    """How much does this row look like a print number?"""
    k = len(row)
    if k == 0 or k > MAX_DIGITS:
        return -1e9
    heights = np.array([c[3] for c in row], dtype=float)
    widths = np.array([c[2] for c in row], dtype=float)

    s = 0.0
    s -= 40 * (heights.std() / heights.mean())                  # one type size
    aspect = heights.mean() / max(1.0, widths.mean())
    s -= 18 * abs(np.log(max(aspect, 1e-3) / 1.7))              # digits are ~1.7:1
    if k > 1:
        centres = np.array([c[0] + c[2] / 2 for c in row], dtype=float)
        gaps = np.diff(centres)
        if gaps.min() <= 0:
            return -1e9
        s -= 22 * (gaps.std() / gaps.mean())                    # evenly set
        s -= 14 * abs(np.log(gaps.mean() / max(1.0, widths.mean() * 1.25)))
    s += 9 * k                                                  # prefer a longer read
    s += 5.0 * (heights.mean() / (0.30 * h))                    # prefer the big text
    s += 3.0 * (np.mean([c[0] + c[2] / 2 for c in row]) / w)    # badge sits right
    return s


def _split_fused(gray: np.ndarray, row: list[tuple]) -> list[tuple]:
    """Split any component still holding two glyphs, at the ink valley."""
    out: list[tuple] = []
    for (x, y, bw, bh) in row:
        if bh <= 0 or bw / bh <= MAX_DIGIT_ASPECT:
            out.append((x, y, bw, bh))
            continue
        k = max(2, int(round(bw / (0.62 * bh))))
        sub = gray[y:y + bh, x:x + bw]
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
        for a, b in zip(bounds, bounds[1:]):
            if b - a >= 3:
                out.append((x + a, y, b - a, bh))
    out.sort(key=lambda c: c[0])
    return out


def prepare_window(window_gray: np.ndarray) -> np.ndarray:
    """Upscale a badge window to the size the thresholds are tuned for."""
    return cv2.resize(window_gray, None, fx=SCALE, fy=SCALE,
                      interpolation=cv2.INTER_CUBIC)


def segment_digits(work_gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Glyph boxes of the print number, left to right, in `work_gray` coords."""
    h, w = work_gray.shape
    best, best_score = [], -1e18
    for thr in THRESHOLDS:
        for row in _rows_at(work_gray, thr):
            sc = _score(row, h, w)
            if sc > best_score:
                best_score = sc
                best = [(c[0], c[1], c[2], c[3]) for c in row]
    return _split_fused(work_gray, best) if best else []


# --------------------------------------------------------------------------
# classification


@functools.lru_cache(maxsize=1)
def _atlas() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(_ATLAS_PATH) as z:
        tpl, labels, card = z["templates"], z["labels"], z["card"]
    X = tpl.reshape(len(tpl), -1).astype(np.float32)
    X -= X.mean(axis=1, keepdims=True)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-6)
    return X, labels, card


def atlas_samples() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw atlas templates, their labels, and the card each came from."""
    with np.load(_ATLAS_PATH) as z:
        return z["templates"], z["labels"], z["card"]


def _glyph_vector(work_gray: np.ndarray, box) -> np.ndarray | None:
    x, y, w, h = box
    sub = work_gray[y:y + h, x:x + w]
    if sub.size == 0:
        return None
    sub = cv2.normalize(sub, None, 0, 255, cv2.NORM_MINMAX)
    sub = cv2.resize(sub, (GLYPH_W, GLYPH_H), interpolation=cv2.INTER_AREA)
    v = sub.reshape(-1).astype(np.float32)
    v -= v.mean()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else None


def read_print_number_from_window(window_gray: np.ndarray) -> tuple[int | None, float]:
    """Read the print number from a badge window. Returns (value, confidence).

    Confidence is the mean cosine similarity to the matched templates, which
    separates a clean glyph from a smeared one far better than a general OCR
    engine's own score did -- that one ran 0.31 on correct digits against 0.27
    on wrong ones, which is no signal at all.
    """
    work = prepare_window(window_gray)
    boxes = segment_digits(work)
    if not boxes:
        return None, 0.0
    X, labels, _ = _atlas()
    digits, sims = [], []
    for box in boxes:
        v = _glyph_vector(work, box)
        if v is None:
            continue
        s = X @ v
        i = int(np.argmax(s))
        digits.append(str(labels[i]))
        sims.append(float(s[i]))
    if not digits:
        return None, 0.0
    text = "".join(digits)
    if len(text) > MAX_DIGITS or not text.isdigit():
        return None, 0.0
    return int(text), round(float(np.mean(sims)), 3)


def read_print_number(card_bgr: np.ndarray) -> tuple[int | None, float]:
    """Read the print number from a whole card image."""
    h, w = card_bgr.shape[:2]
    window = card_bgr[0:max(8, int(WIN_TOP * h)), int(WIN_LEFT * w):]
    if window.size == 0:
        return None, 0.0
    gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
    return read_print_number_from_window(gray)
