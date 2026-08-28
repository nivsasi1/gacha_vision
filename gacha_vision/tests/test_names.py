"""The name reader, against the real name bands of all 182 corpus cards."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from gacha_vision.names import prepare_band, segment_lines

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def corpus():
    bands = np.load(DATA / "name_bands.npz")
    with (DATA / "name_truth.csv").open(encoding="utf-8") as f:
        truth = {r["card"]: (r["true_character"], r["true_series"])
                 for r in csv.DictReader(f)}
    assert len(truth) == 182
    # Task 1 review finding: nothing guarded that the npz and csv key sets
    # actually agree. They must, or a card silently drops out of one side.
    assert set(bands.files) == set(truth.keys())
    return bands, truth


def test_finds_at_least_one_line_on_every_card(corpus):
    bands, truth = corpus
    # cards33.png#1 genuinely carries no name text (empty truth), so it
    # cannot be expected to yield a line -- exclude it from this assertion.
    named = [c for c, (ch, se) in truth.items() if ch]
    empty = [c for c in named if not segment_lines(prepare_band(bands[c]))]
    assert empty == [], f"no text found on {len(empty)} cards: {empty[:10]}"


def test_line_count_covers_the_wrapped_series_names(corpus):
    """A series like 'I've Been Killing Slimes for 300 Years...' wraps to
    three lines, so a two-line assumption drops most of it."""
    bands, truth = corpus
    wrapped = [c for c, (ch, se) in truth.items() if len(se) > 40]
    assert wrapped, "corpus should contain long wrapped series names"
    thin = [c for c in wrapped if len(segment_lines(prepare_band(bands[c]))) < 3]
    assert len(thin) <= len(wrapped) * 0.2, (
        f"{len(thin)}/{len(wrapped)} long series found on fewer than 3 lines")


def test_top_line_component_count_tracks_the_character_name(corpus):
    """Line *counts* alone don't prove segmentation is looking at the real
    text -- a stub that returns 18 fake single-box lines for every card,
    ignoring its input, passes both tests above. Bind the topmost line (the
    character name) to the label: its component count should scale with the
    number of letters in `true_character`. Glyphs aren't split individually
    at this stage (that's Task 3), so touching letters fuse into fewer
    components -- the count is a lower bound, not an exact match. 0.3x the
    letter count is a threshold real segmentation clears on ~90% of the
    corpus (measured directly against name_truth.csv); a fixed-shape stub
    that ignores the image clears it on essentially none."""
    bands, truth = corpus
    named = [c for c, (ch, se) in truth.items() if ch]
    thin = []
    for c in named:
        ch, _ = truth[c]
        n_letters = len(ch.replace(" ", ""))
        lines = segment_lines(prepare_band(bands[c]))
        top_count = len(lines[0]) if lines else 0
        if top_count < 0.3 * n_letters:
            thin.append(c)
    assert len(thin) <= len(named) * 0.15, (
        f"{len(thin)}/{len(named)} cards' topmost line component count "
        f"doesn't track the character name's letter count: {thin[:10]}")


def test_word_gap_falls_between_the_two_words(corpus):
    """Counting *how many* gaps a line has isn't enough -- a stub that
    always inserts exactly one fixed gap into a fixed-shape output, ignoring
    `work_gray`/`line` entirely, would clear that on every two-word card.
    Bind the gap's *position*: the glyph count on each side must match each
    word's own letter count, not just sum to the total. 0.20x is measured
    directly against the real implementation (0.299 on the corpus); a
    shape-only stub cannot clear it since its gap position never tracks the
    actual word split."""
    from gacha_vision.names import split_line
    bands, truth = corpus
    two_word = [c for c, (ch, _) in truth.items() if ch.count(" ") == 1]
    ok = 0
    for card in two_word:
        character, _ = truth[card]
        w1, w2 = character.split(" ")
        work = prepare_band(bands[card])
        lines = segment_lines(work)
        if not lines:
            continue
        out = split_line(work, lines[0])
        none_idx = [i for i, g in enumerate(out) if g is None]
        if len(none_idx) != 1:
            continue
        before = sum(1 for g in out[:none_idx[0]] if g is not None)
        after = sum(1 for g in out[none_idx[0] + 1:] if g is not None)
        if before == len(w1) and after == len(w2):
            ok += 1
    assert ok >= 0.20 * len(two_word), (
        f"only {ok}/{len(two_word)} two-word names split at the right glyph "
        f"counts on each side of the gap")
