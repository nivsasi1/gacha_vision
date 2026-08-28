"""Forced alignment and template harvesting, against the real corpus.

`forced_align` is given the correct text up front (Task 1's hand-read
labels), so unlike free segmentation it can never "fail to find" glyphs --
it always places exactly len(text.replace(" ", "")) spans. The property
worth testing is whether those placements are *any good*: ordered,
non-overlapping, and covering the ink that's actually there, rather than
some fixed-shape answer that ignores the pixels entirely.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from gacha_vision.names import prepare_band, segment_lines, split_line
from gacha_vision.names_align import (
    LINE_HEIGHT,
    GAP_MIN_FRAC,
    line_ink_bounds,
    normalise_line,
    forced_align,
    harvest,
    mean_templates,
)

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def corpus():
    bands = np.load(DATA / "name_bands.npz")
    with (DATA / "name_truth.csv").open(encoding="utf-8") as f:
        truth = {r["card"]: (r["true_character"], r["true_series"])
                 for r in csv.DictReader(f)}
    return bands, truth


def _seed_pool(bands, truth):
    """A crude first-round template per class, from the ~52 cards where
    Task 2/3's naive per-glyph split already agrees with the label's letter
    count. This is Task R1's prescribed seeding step -- kept local to the
    test so the test doesn't depend on `tools/build_glyph_atlas.py` having
    already been run.
    """
    pool = defaultdict(list)
    for card, (character, _series) in truth.items():
        if not character:
            continue
        work = prepare_band(bands[card])
        lines = segment_lines(work)
        if not lines:
            continue
        boxes = split_line(work, lines[0])
        glyphs = [g for g in boxes if g is not None]
        letters = [c for c in character if c != " "]
        if len(glyphs) != len(letters):
            continue
        x0, y0, x1, y1 = line_ink_bounds(lines[0], work.shape)
        scale = LINE_HEIGHT / (y1 - y0)
        norm = normalise_line(work, lines[0])
        for (gx, gy, gw, gh), ch in zip(glyphs, letters):
            nx0 = round((gx - x0) * scale)
            nx1 = round((gx + gw - x0) * scale)
            nx0, nx1 = max(0, nx0), min(norm.shape[1], nx1)
            if nx1 > nx0:
                pool[ch].append(norm[:, nx0:nx1].copy())
    return dict(pool)


@pytest.fixture(scope="module")
def seed_templates(corpus):
    bands, truth = corpus
    pool = _seed_pool(bands, truth)
    assert len(pool) >= 20, f"seeding should cover most of the alphabet, got {len(pool)} classes"
    return mean_templates(pool)


@pytest.fixture(scope="module")
def named_cards(corpus):
    """Every card with real name text, paired with its normalised top line."""
    bands, truth = corpus
    out = []
    for card, (character, _series) in truth.items():
        if not character:
            continue
        work = prepare_band(bands[card])
        lines = segment_lines(work)
        if not lines:
            continue
        norm = normalise_line(work, lines[0])
        if norm.size == 0:
            continue
        out.append((card, character, norm))
    assert len(out) >= 150
    return out


def test_normalise_line_produces_a_fixed_height_real_image(named_cards):
    for card, _character, norm in named_cards[:20]:
        assert norm.shape[0] == LINE_HEIGHT, card
        assert norm.shape[1] > 0, card
        # A stub could return the right shape filled with a constant; the
        # real crop has actual ink variation in it.
        assert float(norm.std()) > 5.0, f"{card}: suspiciously flat crop"


def _ink_coverage(norm: np.ndarray, spans) -> float:
    """Fraction of `norm`'s own ink (its brightest 30% of pixels, the same
    percentile-relative idea `names.py` uses throughout rather than a fixed
    grey level) that falls inside the union of aligned spans.

    Deliberately not raw pixel-width coverage: `segment_lines`'s line boxes
    are sometimes wider than the real text (a short name's box can pull in
    neighbouring border art to clear `MIN_LINE_SPAN`), so a short, correctly
    -placed name can legitimately occupy a small fraction of a wide, noisy
    crop's *width*. Measuring against the crop's own ink instead of its
    width controls for that.
    """
    ink_mask = norm >= np.percentile(norm, 70)
    total = int(ink_mask.sum())
    if total == 0:
        return 1.0
    covered = np.zeros(norm.shape[1], dtype=bool)
    for _, x0, x1 in spans:
        covered[x0:x1] = True
    return float(ink_mask[:, covered].sum()) / total


def test_forced_align_is_ordered_non_overlapping_and_covers_most_of_the_line(
        named_cards, seed_templates):
    """The core alignment property. A stub returning fixed positions --
    e.g. always [(c, 0, 10) for c in text] -- would be ordered and
    non-overlapping too, but would not come close to spanning a real
    line's own ink, so the coverage check is what actually exercises the
    real pixel data.

    Ordering/non-overlap hold on every card -- they're structural
    guarantees of the DP, not a data-quality question. Ink coverage isn't:
    ~14% of cards have `segment_lines` picking up decorative artwork
    instead of (or alongside) the real name (measured directly -- see
    `docs/superpowers/plans/2026-08-27-name-reader.md`'s pivot notes and
    task-R1-report.md), and forced_align cannot recover text that was
    never in the crop it was handed. 0.5 ink coverage / 20% "thin" is
    measured against this exact implementation (14.4% fall below 0.5) with
    headroom -- `tools/build_glyph_atlas.py` is responsible for not
    training the atlas from the thin tail, not this test.
    """
    thin = []
    checked = 0
    for card, character, norm in named_cards:
        letters = [c for c in character if c != " "]
        spans = forced_align(norm, character, seed_templates)
        assert [c for c, _, _ in spans] == letters, card

        for c, x0, x1 in spans:
            assert x0 < x1, (card, c)
        for (_, _, x1), (_, nx0, _) in zip(spans, spans[1:]):
            assert x1 <= nx0, f"{card}: overlapping spans"

        checked += 1
        coverage = _ink_coverage(norm, spans)
        if coverage < 0.5:
            thin.append((card, round(coverage, 3)))

    assert checked >= 150
    assert len(thin) <= 0.20 * checked, (
        f"{len(thin)}/{checked} cards' alignment covered under 50% of the "
        f"line's own ink: {thin[:10]}")


def test_forced_align_marks_a_real_gap_for_two_word_names(corpus, seed_templates):
    bands, truth = corpus
    gap_min_px = GAP_MIN_FRAC * LINE_HEIGHT
    two_word = [(c, ch) for c, (ch, _) in truth.items() if ch.count(" ") == 1]
    assert two_word
    found = 0
    for card, character in two_word:
        work = prepare_band(bands[card])
        lines = segment_lines(work)
        if not lines:
            continue
        norm = normalise_line(work, lines[0])
        w1, w2 = character.split(" ")
        spans = forced_align(norm, character, seed_templates)
        if len(spans) != len(w1) + len(w2):
            continue
        gap = spans[len(w1)][1] - spans[len(w1) - 1][2]
        if gap >= gap_min_px:
            found += 1
    assert found >= 0.70 * len(two_word), (
        f"only {found}/{len(two_word)} two-word names got a real gap between the words")


def test_harvest_bitmaps_match_the_alignment_spans(named_cards, seed_templates):
    card, character, norm = named_cards[0]
    spans = forced_align(norm, character, seed_templates)
    glyphs = harvest(norm, spans)

    letters = [c for c, _, _ in spans]
    assert set(glyphs) == set(letters)
    for (c, x0, x1) in spans:
        matches = [g for g in glyphs[c] if g.shape[1] == x1 - x0]
        assert matches, f"{card}: no harvested bitmap of the aligned width for {c!r}"
        assert matches[0].shape[0] == LINE_HEIGHT
        assert matches[0].dtype == np.uint8


def test_mean_templates_collapses_a_pool_to_one_bitmap_per_class():
    rng = np.random.default_rng(0)
    pool = {
        "a": [rng.integers(0, 255, size=(LINE_HEIGHT, 10), dtype=np.uint8),
              rng.integers(0, 255, size=(LINE_HEIGHT, 12), dtype=np.uint8)],
        "b": [rng.integers(0, 255, size=(LINE_HEIGHT, 20), dtype=np.uint8)],
    }
    templates = mean_templates(pool)
    assert set(templates) == {"a", "b"}
    for c, t in templates.items():
        assert t.shape[0] == LINE_HEIGHT
        assert t.dtype == np.uint8
