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


# --------------------------------------------------------------------------
# free-running (unconstrained) decoding -- Task R2
#
# forced_align above places a *known* string: every token's class is given,
# so the only thing the DP searches for is position. Reading an unlabelled
# card has no known string -- the DP has to search class *and* position (and
# how many glyphs there even are) at once. That needs two changes to the
# same shift-and-score idea:
#
#   1. At every column, every atlas class is a candidate, not just the next
#      token in a fixed script -- so the DP takes an elementwise max over
#      classes instead of following a predetermined sequence.
#   2. There is no "the rest of the string still has to go somewhere" -- so
#      a column that no class explains well can simply cost nothing and be
#      skipped, rather than being forced to absorb some letter's template.
#      That is what lets a free read step over the border-art contamination
#      documented in task-R1-report.md instead of being forced to mislabel
#      it (forced_align has no such option because every token in the known
#      string must be placed *somewhere*).

# Candidate glyph widths, in the LINE_HEIGHT-normalised frame free_align
# operates in. Measured from split_line's own widths on the ~52 cards whose
# naive segmentation already agrees with the character-name label (the same
# cards Task R1's seeding uses), converted to this frame via each line's own
# ink-bounds scale: 1st/50th/99th percentile 3.8 / 13.6 / 30.1px (n=534,
# min 0.8, max 37.5). The bucket list below spans that range at ~2px
# resolution through the dense middle and coarser at the tails, since the
# widest letters ('m', 'M', 'W') are rare enough that precision there
# matters less than keeping the per-line sweep (width x class) small.
FREE_WIDTH_CANDIDATES: tuple[int, ...] = (5, 7, 9, 11, 13, 16, 19, 22, 26, 30, 34)

# The cost of declaring one run of columns "not a glyph", regardless of how
# wide that run is -- a flat per-*token* cost, not a per-column one. A
# per-column cost would make a wide skip cost proportionally more than a
# narrow one, which has no basis here: matchTemplate's normalised
# correlation doesn't get systematically weaker for wider templates, so a
# wide letter shouldn't need a disproportionately better score just to beat
# background over its own width. (An early version of this DP charged
# background per column instead and had a second, worse bug as a result:
# because a background-ending dp[x] could itself feed the *next* column's
# background option, the flat cost compounded every single column instead of
# being charged once per run, so a handful of unmatched columns was enough
# to make background unbeatable for the rest of the line. free_align's
# bg_widths below avoids that: background is just more entries in the same
# width-indexed transition table characters use, so two background tokens
# placed back to back always cost background_score twice, never once.)
#
# The value itself is 0.0 -- neutral, not a reward -- and that is load
# bearing, found the hard way: a first attempt set this to +0.40 (see
# FREE_CHAR_PENALTY below for where that number came from) on the reasoning
# that background should score about what a spurious match typically
# achieves. That made background strictly profitable to chain: since two
# adjacent background tokens both still fire independently (previous
# paragraph), and each contributes a *positive* +0.40 regardless of the
# pixels underneath it, the DP's optimal move was always to tile the entire
# line with as many minimum-width background tokens as it could fit --
# caught by the ordering/coverage unit test below scoring 0 spans on a line
# with two unmistakable synthetic glyphs on it, and confirmed by hand: dp
# climbed by exactly 0.40 every 5 columns regardless of content, reaching
# 9.6 over a 120px line versus ~0.6 for the one real glyph actually in it.
# The fix is this constant at 0.0 (a background run of any length
# contributes nothing, so chaining more of them is neutral, never a gain)
# with the *character* side of the comparison doing 100% of the
# discrimination work instead -- see FREE_CHAR_PENALTY.
FREE_BACKGROUND_SCORE: float = 0.0

# A flat cost subtracted from every *character* placement (background is
# unaffected, see above). With background scoring a flat 0.0, this
# constant alone decides how good a match has to be to beat "nothing is
# here": a class scoring `correlation` at some position is placed only if
# `correlation - FREE_CHAR_PENALTY` clears whatever the rest of the DP is
# doing, which -- against a background of flat 0.0 over the same span --
# means only positions genuinely correlating above ~FREE_CHAR_PENALTY are
# ever placed at all.
#
# The value is measured, and the measurement is a real finding, not just a
# tuning note. Matching a glyph-sized patch against 63 classes' worth of
# templates and keeping the best score is prone to a false positive by
# construction -- more candidates means a higher expected best-of-many even
# against pixels that aren't a letter at all. Measured directly: 172 cards'
# forced-aligned true-glyph positions ("ON") versus random same-line
# positions at least 6px from any true glyph ("OFF"), both scored the same
# way free_align scores a candidate (best-over-63-classes correlation,
# converged atlas class means):
#
#   percentile        1     5    10    25    50    75    90    95    99
#   ON  score       0.11  0.22  0.27  0.40  0.55  0.68  0.77  0.82  0.88
#   OFF score       0.12  0.17  0.21  0.32  0.47  0.65  0.75  0.79  0.84
#
# The two distributions overlap heavily -- a single glyph-sized patch of
# card artwork routinely correlates with *some* letter template about as
# well as a real letter does, because at this resolution both are just
# blobs of bright ink on a dark ground. No threshold cleanly separates them
# (see task-R2-report.md); the sweep actually run found 0.40 as the value
# maximising P(ON > thr) - P(OFF > thr) (0.759 vs 0.611, a 15-point gap --
# the best on offer, not a confident separator).
#
# That per-glyph number is a starting point, not the final value, because a
# *sequence* DP has a second failure mode a single-position threshold
# doesn't: a self-similar glyph shape (e.g. two horizontal strokes repeated
# down a tall letter) can correlate almost as well one-third at a time as it
# does whole, and nothing about a per-glyph threshold stops the DP from
# using three cheap-looking tokens where one correct one belongs -- 0.40
# measured +16.4 mean excess glyphs per line against the true letter count
# (100 cards, class means from the converged atlas, no LOO). Sweeping this
# constant directly against that same span-count-vs-truth measurement (the
# only lever now, since background is fixed at neutral) finds the bias
# crossing zero around 0.70 (+0.8 mean excess, -3.0 median, at 150 cards).
# Note for the record: even there only ~10-17% of individual cards land
# within +-2 of their true letter count -- the mean crossing zero reflects
# errors cancelling in both directions, not most cards being close, which is
# the per-glyph ON/OFF overlap above showing up again at the sequence level.
# See task-R2-report.md for how this caps the final leave-one-card-out
# accuracy.
FREE_CHAR_PENALTY: float = 0.70


def free_align(line_img: np.ndarray, class_templates: dict[str, np.ndarray],
                widths: tuple[int, ...] = FREE_WIDTH_CANDIDATES,
                background_score: float = FREE_BACKGROUND_SCORE,
                char_penalty: float = FREE_CHAR_PENALTY
                ) -> list[tuple[int, int]]:
    """DP-place an unknown number of unknown-class glyphs onto `line_img`.

    `class_templates` (one representative bitmap per class, all the same
    canonical shape -- names.py builds this from the atlas's own storage
    shape, not each class's natural width, since the persisted atlas
    doesn't keep that) is tried at every width in `widths`; the DP chooses
    whichever (width, class, position) combination -- or background --
    maximises the cumulative score. Returns non-overlapping `(x_start,
    x_end)` spans in left-to-right order; *which* class won at each span is
    deliberately not returned here, because a per-class mean is a blurry
    classifier by design (it has to be, to keep the width sweep affordable)
    -- names.read_line_free reclassifies each returned span against the
    full, unblurred atlas by nearest-neighbour, the same way digits.py
    classifies.
    """
    H, W = line_img.shape
    if W == 0 or not class_templates:
        return []
    line_f = line_img.astype(np.float32)

    # best_by_width[w][i]: the best score, over every class, of placing a
    # w-wide glyph starting at column i, net of char_penalty. Computed once
    # per width (not once per class per position) by collapsing
    # cv2.matchTemplate's per-class curves with a single elementwise max.
    best_by_width: dict[int, np.ndarray] = {}
    for w in widths:
        if w < 1 or w > W:
            continue
        curves = []
        for tpl in class_templates.values():
            rtpl = tpl.astype(np.float32)
            if rtpl.shape[1] != w:
                interp = cv2.INTER_AREA if w < rtpl.shape[1] else cv2.INTER_LINEAR
                rtpl = cv2.resize(rtpl, (w, H), interpolation=interp)
            curves.append(cv2.matchTemplate(line_f, rtpl, cv2.TM_CCOEFF_NORMED).reshape(-1))
        if curves:
            best_by_width[w] = np.max(np.stack(curves), axis=0) - char_penalty

    # A background run costs the same flat `background_score` regardless of
    # which of these widths it uses -- so given a genuinely blank span, the
    # DP always prefers the widest one that fits (more columns explained per
    # unit cost), and a big contaminated block collapses to one or two
    # background tokens rather than needing a whole chain of small ones.
    # Reusing `widths` covers ordinary inter-letter and word gaps; the
    # coarser buckets past its top (up to 130px) exist only so a large
    # contaminated region -- e.g. a fused decorative line, task-R1-report.md
    # -- doesn't have to pay `background_score` many times over to get past.
    # No real glyph is anywhere near this wide (99th percentile measured at
    # 30px, see FREE_WIDTH_CANDIDATES above), so these buckets are
    # unambiguously background-only.
    bg_widths = sorted(set(w for w in widths if 1 <= w <= W)
                        | set(w for w in (45, 60, 90, 130) if w <= W))

    # dp[x]: best cumulative score of a placement (glyphs and background
    # runs) covering columns [0, x). Both transition kinds end exactly at x
    # and start at x - w for their own w, so -- unlike a per-column
    # background cost -- two background runs placed back to back always
    # cost 2 x background_score, never less: nothing here lets adjacent
    # background tokens merge into one and silently keep re-collecting the
    # flat cost, which is what a naive "cost accumulates while you stay in
    # a background state" formulation would do. Free start comes for free --
    # a background run beginning at column 0 is just another x - w = 0
    # transition, so nothing needs to anchor the first glyph to x=0 the way
    # forced_align's dp0 = 0 initialisation does explicitly.
    NEG = -1e18
    dp = np.zeros(W + 1)
    # choice[x]: (True, w) for a glyph of width w ending at x; (False, w)
    # for a background run of width w ending at x.
    choice: list[tuple[bool, int]] = [(False, 1)] * (W + 1)
    for x in range(1, W + 1):
        best_val, best_choice = NEG, (False, 1)
        for w, arr in best_by_width.items():
            if w > x:
                continue
            val = dp[x - w] + arr[x - w]
            if val > best_val:
                best_val, best_choice = val, (True, w)
        for w in bg_widths:
            if w > x:
                continue
            val = dp[x - w] + background_score
            if val > best_val:
                best_val, best_choice = val, (False, w)
        dp[x] = best_val
        choice[x] = best_choice

    if dp[W] <= NEG / 2:
        # W can't be reached by any combination of the candidate widths (all
        # >= 5) -- only possible on a pathologically narrow crop, since real
        # lines are hundreds of pixels wide and 5 & 7 alone already span
        # every width beyond 23. Same defensive floor forced_align uses.
        return []

    spans: list[tuple[int, int]] = []
    x = W
    while x > 0:
        is_char, w = choice[x]
        if is_char:
            spans.append((x - w, x))
        x -= w
    spans.reverse()
    return spans
