"""Place a card's already-known name text onto its pixels.

`names.py` cuts a line into glyphs before recognising them, and that cannot
work on this font: an oracle allowed to pick the best threshold per card
*using the known answer* still only reaches 55-62% exact glyph-count match
(see docs/superpowers/plans/2026-08-27-name-reader.md, the 2026-08-28
revision). Some letters are only single components at a threshold that
fuses other letters on the same line, so no per-line threshold exists --
that is a ceiling on cutting-then-classifying, not slack in its constants.

This module never cuts. Every card's correct text is already known (Task 1's
hand-read labels), so instead of guessing where letters start and stop, a
template for each letter is slid along the line and dynamic programming
finds the best non-overlapping placement *for that known string*. That turns
"segment correctly" into "find the best positions for a known answer", which
always produces an answer -- the DP cannot fail to place all of it, the way
free segmentation can fail to find the right number of components.

Pipeline (driven by `tools/build_glyph_atlas.py`):

    normalise_line   -- crop a text line to its ink and rescale to a fixed
                         height, so a template harvested from one card's
                         band is comparable to another's.
    forced_align      -- DP-place a known string onto a normalised line.
    harvest            -- cut glyph bitmaps out at the positions found.
    mean_templates     -- collapse harvested bitmaps into one template per
                         class, ready for the next round of forced_align.

Repeating align -> harvest -> mean_templates is the EM loop: alignment is
the E-step (best positions given the current templates), harvesting +
averaging is the M-step (best templates given the current positions).
"""

from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np

# Every line is rescaled to this height before anything is matched. The
# corpus's own band shapes vary (112-120px tall pre-upscale, one outlier at
# 163), and the character-name line is measurably taller than the wrapped
# series lines beneath it -- neither variation is font size, so leaving it
# in would make a template harvested from one card/line meaningless on
# another. 40 sits close to the corpus median top-line height after
# prepare_band's 4x upscale (43px, measured across all 182 bands), so most
# lines are only lightly resampled rather than manufacturing detail that
# was never there.
LINE_HEIGHT = 40

# Padding around a line's component-box union, as a fraction of that
# union's own height. segment_lines groups components by vertical-centre
# proximity (names.py's `_group_rows`), which can leave a small
# high-riding mark -- an accent, a tall ascender's tip -- just outside the
# box union if its centre falls past the clustering tolerance. The pad
# gives that ink room to still land inside the crop; applied on all four
# sides since nothing in the corpus suggests descenders overhang more than
# ascenders in this font.
LINE_PAD_FRAC = 0.20

# A word space's width, as a fraction of the line's own (padded) height.
# Measured from split_line's own gap detections across the 45 two-word
# character names in the corpus (gap width / local median glyph height):
# min 0.20, median 0.43, max 0.67. The search range below is padded past
# both ends of that spread rather than clamped to it, since a card outside
# this fixture could legitimately fall just past either edge.
GAP_MIN_FRAC = 0.15
GAP_MAX_FRAC = 0.85

# A class with no template yet cannot be scored -- there is nothing to
# compare the pixels against. It is given a neutral (zero) score at every
# position instead of being skipped, so the DP still places it: its
# position is decided entirely by its scored neighbours to its left and
# right, which is exactly what lets an unseeded class bootstrap a first,
# crude position on the very first alignment pass (see `harvest`'s
# docstring and Task R1's report for how that first position becomes next
# round's seed).
DEFAULT_GLYPH_WIDTH_FRAC = 0.55


def line_ink_bounds(line_boxes, work_shape, pad_frac: float = LINE_PAD_FRAC):
    """Padded pixel bounds `(x0, y0, x1, y1)` of a line's ink, clipped to
    `work_shape`. Shared by `normalise_line` and by callers (seeding,
    `tools/build_glyph_atlas.py`) that need to map a box from `segment_lines`
    / `split_line`'s coordinate frame into `normalise_line`'s output frame.
    """
    x0 = min(b[0] for b in line_boxes)
    x1 = max(b[0] + b[2] for b in line_boxes)
    y0 = min(b[1] for b in line_boxes)
    y1 = max(b[1] + b[3] for b in line_boxes)
    pad = round(pad_frac * max(1, y1 - y0))
    h, w = work_shape
    return (max(0, x0 - pad), max(0, y0 - pad),
            min(w, x1 + pad), min(h, y1 + pad))


def normalise_line(work_gray: np.ndarray, line_boxes) -> np.ndarray:
    """Crop a text line to its (padded) ink bounds and rescale to
    `LINE_HEIGHT`, preserving aspect ratio so a letter's width stays
    proportional to its height.

    `line_boxes` is one line from `names.segment_lines(work_gray)` -- only
    used to find the crop region; the pixels themselves come straight from
    `work_gray`, same division of labour as `names.split_line`.
    """
    x0, y0, x1, y1 = line_ink_bounds(line_boxes, work_gray.shape)
    crop = work_gray[y0:y1, x0:x1]
    if crop.size == 0:
        return crop
    crop = cv2.normalize(crop, None, 0, 255, cv2.NORM_MINMAX)
    scale = LINE_HEIGHT / crop.shape[0]
    new_w = max(1, round(crop.shape[1] * scale))
    return cv2.resize(crop, (new_w, LINE_HEIGHT), interpolation=cv2.INTER_CUBIC)


def match_score(patch: np.ndarray, template: np.ndarray) -> float:
    """Mean-centred, unit-norm correlation of two equal-shaped patches --
    the same representation `digits.py`'s atlas match uses, so a card's
    band being globally brighter or dimmer than another's doesn't move the
    score.
    """
    a = patch.reshape(-1).astype(np.float64)
    b = template.reshape(-1).astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _tokenize(text: str):
    """`text` split into `("char", c)` / `("gap", None)` tokens. Runs of
    whitespace collapse to a single gap token; a leading/trailing space
    would otherwise produce a gap with no letter on one side, which the DP
    below has no placement rule for.
    """
    words = text.split()
    tokens: list[tuple[str, str | None]] = []
    for i, word in enumerate(words):
        if i > 0:
            tokens.append(("gap", None))
        tokens.extend(("char", c) for c in word)
    return tokens


def forced_align(line_img: np.ndarray, text: str, templates: dict[str, np.ndarray]
                  ) -> list[tuple[str, int, int]]:
    """DP-place the known string `text` onto `line_img`, sliding each
    class's own template. Returns `(char, x_start, x_end)` for every
    non-space character in `text`, left to right, in `line_img`'s
    coordinate frame (the frame `normalise_line` produces).

    A space becomes a single variable-width gap token, not a template slide
    -- there is no glyph shape to look for, only the expectation that the
    gap sits over low-ink background. Both an unseeded letter class and a
    space are scored the same practical way: neither rewards nor punishes
    any particular position, so the DP places them using only the pull of
    their scored neighbours (see `DEFAULT_GLYPH_WIDTH_FRAC`'s docstring).

    The DP never fails to return a full placement -- it is constrained to
    emit exactly `text`'s letters, in order, non-overlapping. What can be
    poor is the placement's *quality* when `line_img` doesn't actually
    contain the real text (e.g. `segment_lines` picked up decorative
    artwork instead of a name); `tools/build_glyph_atlas.py` filters on
    match score before harvesting to keep that from poisoning the atlas.
    """
    tokens = _tokenize(text)
    if not tokens:
        return []

    H, W = line_img.shape
    widths = [t.shape[1] for t in templates.values()] if templates else []
    fallback_w = max(1, int(round(np.median(widths)))) if widths \
        else max(1, round(DEFAULT_GLYPH_WIDTH_FRAC * H))

    gap_min = max(1, round(GAP_MIN_FRAC * H))
    gap_max = max(gap_min, round(GAP_MAX_FRAC * H))

    # `segment_lines` is only ~86% reliable at isolating just the character
    # line (Task 2's measurement) -- the rest fuse in a neighbouring line or
    # border artwork, which can make a line's ink bounds narrower than the
    # text's templates would naturally need. Without a rescue, the DP would
    # have no feasible placement at all and fail to return the one span per
    # letter it is supposed to always produce. Shrinking every width by the
    # same factor keeps the placement's *proportions* faithful to the
    # templates -- and, on the ~86% of cards where the line is clean, this
    # is a no-op (shrink stays 1.0) since the natural widths already fit.
    natural_total = sum(min(templates[c].shape[1], W) if c in templates else fallback_w
                        for kind, c in tokens if kind == "char")
    natural_total += gap_min * sum(1 for kind, _ in tokens if kind == "gap")
    # A little extra headroom past the exact ratio: every width below is
    # rounded independently, so the rounded sum can land a few pixels above
    # the unrounded target.
    shrink = min(1.0, 0.97 * W / natural_total) if natural_total > 0 else 1.0

    fallback_w = max(1, int(round(fallback_w * shrink)))
    gap_min = max(1, int(round(gap_min * shrink)))
    gap_max = max(gap_min, int(round(gap_max * shrink)))

    line_f = line_img.astype(np.float32)
    # Per-column brightness as a 0..1 "how much ink lives here" proxy, used
    # only to score candidate gap widths -- normalise_line already stretches
    # each line to the full 0-255 range, so this is comparable line to line.
    ink = line_img.astype(np.float64).mean(axis=0) / 255.0
    gap_signal = 1.0 - 2.0 * ink            # blank column -> +1, solid ink -> -1
    gap_prefix = np.concatenate(([0.0], np.cumsum(gap_signal)))

    gap_max = min(gap_max, W)

    curve_cache: dict[str, tuple[int, np.ndarray]] = {}

    def char_curve(c: str) -> tuple[int, np.ndarray]:
        if c in curve_cache:
            return curve_cache[c]
        tpl = templates.get(c)
        if tpl is None:
            w = max(1, min(fallback_w, W))
            curve = np.zeros(max(0, W - w + 1))
        else:
            w = max(1, min(int(round(tpl.shape[1] * shrink)), W))
            tpl_f = tpl.astype(np.float32)
            if shrink < 1.0 or tpl_f.shape[1] != w:
                tpl_f = cv2.resize(tpl_f, (w, H), interpolation=cv2.INTER_AREA)
            if w < 1 or tpl_f.shape[0] != H:
                curve = np.zeros(max(0, W - w + 1))
            else:
                res = cv2.matchTemplate(line_f, tpl_f, cv2.TM_CCOEFF_NORMED)
                curve = res.reshape(-1).astype(np.float64)
        curve_cache[c] = (w, curve)
        return w, curve

    NEG = -1e18
    # dp[x]: best cumulative score of a boundary landing at column x, with
    # every token processed so far placed to its left. Free at x=0 tokens
    # placed (score 0 everywhere) -- the line is already cropped tight to
    # its own ink, so nothing anchors the first letter to x=0 more than any
    # other column; letting the DP choose costs nothing and tolerates the
    # crop's own padding slack.
    dp = np.zeros(W + 1)
    # backptr[i]: for a char token, its fixed width (predecessor is x - w);
    # for a gap token, an array of the chosen gap width per end-column x.
    backptr: list[int | np.ndarray] = [0] * len(tokens)

    for i, (kind, c) in enumerate(tokens):
        new_dp = np.full(W + 1, NEG)
        if kind == "char":
            w, curve = char_curve(c)
            n = len(curve)
            if n > 0:
                new_dp[w:w + n] = dp[:n] + curve
            backptr[i] = w
        else:
            best = np.full(W + 1, NEG)
            choice = np.zeros(W + 1, dtype=np.int32)
            hi = min(gap_max, W)
            for gw in range(gap_min, hi + 1):
                n = W - gw + 1
                if n <= 0:
                    break
                x0 = np.arange(n)
                x1 = x0 + gw
                seg_score = (gap_prefix[x1] - gap_prefix[x0]) / gw
                cand = dp[x0] + seg_score
                better = cand > best[x1]
                idx = x1[better]
                best[idx] = cand[better]
                choice[idx] = gw
            new_dp = best
            backptr[i] = choice
        dp = new_dp

    end_x = int(np.argmax(dp))
    if dp[end_x] <= NEG / 2:
        return []

    spans: list[tuple[str, int, int] | None] = [None] * len(tokens)
    x = end_x
    for i in range(len(tokens) - 1, -1, -1):
        kind, c = tokens[i]
        if kind == "char":
            w = backptr[i]
            x0 = x - w
            spans[i] = (c, x0, x)
        else:
            gw = int(backptr[i][x])
            x0 = x - gw
        x = x0

    return [s for s in spans if s is not None]


def harvest(line_img: np.ndarray, alignment) -> dict[str, list[np.ndarray]]:
    """Cut glyph bitmaps out of `line_img` at each `(char, x_start, x_end)`
    in `alignment` (`forced_align`'s return value), grouped by class.

    Every bitmap keeps the line's full height rather than being cropped
    tight to that one glyph's own ink -- a letter's vertical position
    within the line (x-height vs ascender vs descender) is itself part of
    what distinguishes it from another class, and cropping it away would
    make e.g. 'o' and 'p' look more alike than they are.
    """
    out: dict[str, list[np.ndarray]] = defaultdict(list)
    H, W = line_img.shape
    for char, x0, x1 in alignment:
        x0c, x1c = max(0, int(x0)), min(W, int(x1))
        if x1c <= x0c:
            continue
        out[char].append(line_img[:, x0c:x1c].copy())
    return dict(out)


def mean_templates(pool: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    """Collapse each class's harvested bitmaps into one representative
    template, for the next round of `forced_align`.

    Samples of the same class differ by a few pixels of width -- kerning,
    threshold jitter at the seeding stage, a slightly different alignment
    cut -- so each is resized to that class's own median width before
    averaging; averaging mismatched widths directly would blur the shape
    rather than sharpen it.
    """
    out: dict[str, np.ndarray] = {}
    for c, samples in pool.items():
        if not samples:
            continue
        h = samples[0].shape[0]
        med_w = max(1, int(round(np.median([s.shape[1] for s in samples]))))
        resized = [s if s.shape[1] == med_w
                   else cv2.resize(s, (med_w, h), interpolation=cv2.INTER_AREA)
                   for s in samples]
        stack = np.stack(resized).astype(np.float64)
        out[c] = stack.mean(axis=0).astype(np.uint8)
    return out
