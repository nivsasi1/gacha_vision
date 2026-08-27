"""The badge digit reader, against the badge windows of 81 real numbered cards.

These are real pixels, not renders: `badge_windows.npz` holds the grayscale
top-right corner of every numbered card in the labelled corpus, and
`badge_truth.csv` holds what a human read off each one.

Why this file exists. Tesseract managed 38 of these 81. The digits are crisp
white glyphs in a fixed game font -- the failure was never recognition, it was
that the reader binarised at one global cut point, split the result into
connected components, and threw away anything too wide. A leading `1` sits a
hair away from its neighbour, so `1550` fused into `15`+`5`+`0`, `1770` lost
its `1` entirely, and the surviving fragment lost the vote to noise from the
artwork. Every "digits off" read in the corpus was that bug, not a hard glyph.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from gacha_vision.digits import atlas_samples, read_print_number_from_window

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def corpus():
    windows = np.load(DATA / "badge_windows.npz")
    with (DATA / "badge_truth.csv").open(encoding="utf-8") as f:
        truth = {r["card"]: r["true_print"] for r in csv.DictReader(f)}
    assert len(truth) == 81, f"expected 81 numbered cards, got {len(truth)}"
    return windows, truth


def _read_all(corpus):
    windows, truth = corpus
    got = {}
    for card, want in truth.items():
        value, _ = read_print_number_from_window(windows[card])
        got[card] = (want, "" if value is None else str(value))
    return got


def test_reads_the_print_number_on_almost_every_real_badge(corpus):
    got = _read_all(corpus)
    wrong = {c: v for c, v in got.items() if v[0] != v[1]}
    assert len(wrong) <= 2, (
        f"{len(wrong)} of {len(got)} badges misread: "
        + ", ".join(f"{c} want {w} got {g or '-'}" for c, (w, g) in sorted(wrong.items()))
    )


def test_a_leading_one_is_not_dropped(corpus):
    """The exact shape of the old bug: `1550` came back as `550`."""
    windows, truth = corpus
    leading_ones = [c for c, t in truth.items() if t.startswith("1") and len(t) == 4]
    assert len(leading_ones) >= 20, "corpus should be full of 1xxx prints"
    bad = []
    for card in leading_ones:
        value, _ = read_print_number_from_window(windows[card])
        if value is None or str(value) != truth[card]:
            bad.append(f"{card} want {truth[card]} got {value}")
    assert len(bad) <= 1, "leading digit lost on: " + ", ".join(bad)


def test_no_read_is_off_by_a_digit_count(corpus):
    """A wrong digit is survivable; a wrong *length* means a glyph was lost."""
    got = _read_all(corpus)
    bad = [f"{c} want {w} got {g or '-'}" for c, (w, g) in got.items()
           if g and len(g) != len(w)]
    assert len(bad) <= 1, "length mismatch on: " + ", ".join(bad)


def test_the_atlas_generalises_to_cards_it_has_never_seen():
    """Leave-one-card-out: no glyph is classified using its own card.

    Reading the corpus the atlas was built from would only prove it can
    memorise. Holding out the whole card -- every glyph on it -- is the
    honest measure.
    """
    tpl, labels, cards = atlas_samples()
    assert len(tpl) >= 250, f"atlas too small to trust: {len(tpl)}"
    X = tpl.reshape(len(tpl), -1).astype(np.float32)
    X -= X.mean(axis=1, keepdims=True)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-6)

    ok = total = 0
    for card in np.unique(cards):
        held = cards == card
        rest = ~held
        pred = labels[rest][np.argmax(X[held] @ X[rest].T, axis=1)]
        ok += int((pred == labels[held]).sum())
        total += int(held.sum())
    assert ok / total >= 0.99, f"glyph accuracy on unseen cards {ok}/{total}"


# --- wiring: the pipeline must actually use this reader -------------------

def test_the_pipeline_reads_a_real_card_that_tesseract_got_wrong():
    """End to end on the card the old reader turned into `550`."""
    import cv2
    from gacha_vision.analyze import analyze_cards

    img = cv2.imread(str(DATA / "card_print_1550.png"))
    assert img is not None
    card = analyze_cards(img, expected=1, read_names=False)[0]

    assert card.print_no == 1550
    assert not card.no_number
    assert card.print_trusted, "a clean atlas read must be actionable"


def test_a_font_the_atlas_has_never_seen_falls_back_rather_than_guessing():
    """The atlas is trained on the game's font; anything else must not be
    forced through it, or a rendered test card reads as nonsense."""
    from gacha_vision.digits import read_print_number
    from gacha_vision.models import FrameTier
    from gacha_vision.synth import draw_card

    _, conf = read_print_number(draw_card(tier=FrameTier.NORMAL, badge="1655"))
    assert conf < 0.90, f"synthetic font scored {conf}, inside the trusted band"
