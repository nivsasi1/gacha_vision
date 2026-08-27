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
