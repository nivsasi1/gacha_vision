"""Regression tests against 182 hand-labelled cards from 91 real spawns.

The images stay on the machine that took them; what is committed here is the
measurement CSV the pipeline emitted for them, plus the human labels. That is
enough to replay every reconciliation and ranking decision, which is where the
real failures were -- and it keeps the suite from being, as it once was,
entirely synthetic cards drawn by the code under test.

The corpus these numbers come from:

    101 E cards, 81 numbered, badge OCR exact on 68% of them.

Badge OCR is *not* what these tests police. Digits get misread constantly and
mostly harmlessly -- #1609 read as #4609 is junk either way. What costs a card
is the E/number confusion, and that is what is asserted here.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import pytest

from gacha_vision.config import Policy
from gacha_vision.frame import resolve_frame
from gacha_vision.models import Action, Card, FrameTier
from gacha_vision.ocr import MIN_TRUSTED_CONFIDENCE
from gacha_vision.rank import decide

LABELS = Path(__file__).parent / "data" / "real_labels.csv"
P = Policy()


@pytest.fixture(scope="module")
def rows():
    with LABELS.open(encoding="utf-8") as f:
        out = [r for r in csv.DictReader(f) if (r.get("true_frame") or "").strip()]
    assert len(out) == 182, f"expected the full labelled corpus, got {len(out)}"
    return out


def _replay(row) -> Card:
    """Rebuild the Card the pipeline would produce from a row's measurements."""
    badge_print = int(row["print_no"]) if row["print_no"] else None
    badge_no_number = row["no_number"] == "1"
    tier, print_no, no_number = resolve_frame(
        badge_print, badge_no_number, row["pixel_frame"]
    )
    conf = float(row["ocr_conf"])
    return Card(
        slot=int(row["slot"]),
        print_no=print_no,
        no_number=no_number,
        frame=tier,
        print_trusted=conf >= MIN_TRUSTED_CONFIDENCE,
        ocr_confidence=conf,
    )


def _truth_card(row) -> Card:
    if row["true_print"] == "E":
        return Card(slot=int(row["slot"]), no_number=True, frame=FrameTier.E)
    return Card(slot=int(row["slot"]), print_no=int(row["true_print"]),
                frame=FrameTier.NORMAL)


def _by_spawn(rows, build):
    spawns = defaultdict(list)
    for r in rows:
        spawns[r["image"]].append(r)
    return {
        img: [build(r) for r in sorted(rs, key=lambda r: int(r["slot"]))]
        for img, rs in spawns.items()
    }


def test_frame_is_right_on_essentially_every_real_card(rows):
    ok = sum(1 for r in rows if _replay(r).frame.value == r["true_frame"])
    assert ok / len(rows) >= 0.99, f"frame accuracy {ok}/{len(rows)}"


def test_no_e_card_is_given_a_print_number(rows):
    """The expensive error: junk promoted to a card worth spending a pick on."""
    bad = [r["image"] + " slot" + r["slot"] for r in rows
           if r["true_frame"] == "e" and not _replay(r).no_number]
    assert bad == []


def test_no_numbered_card_is_written_off_as_an_e(rows):
    """The other expensive error: a real print discarded as an E."""
    bad = [r["image"] + " slot" + r["slot"] for r in rows
           if r["true_frame"] == "normal" and _replay(r).no_number]
    assert bad == []


def test_no_spawn_is_claimed_that_the_labels_say_to_skip(rows):
    """A false claim spends one of two picks on nothing."""
    got = _by_spawn(rows, _replay)
    want = _by_spawn(rows, _truth_card)
    bad = [img for img in want
           if decide(want[img], P, {}).action is Action.SKIP
           and decide(got[img], P, {}).action is not Action.SKIP]
    assert bad == []


def test_decisions_match_the_labels_on_almost_every_spawn(rows):
    got = _by_spawn(rows, _replay)
    want = _by_spawn(rows, _truth_card)
    agree = sum(
        1 for img in want
        if (lambda a, b: (a.action, a.slots) == (b.action, b.slots))(
            decide(got[img], P, {}), decide(want[img], P, {})
        )
    )
    assert agree / len(want) >= 0.98, f"decision accuracy {agree}/{len(want)}"


# --- wiring: the pipeline itself must go through resolve_frame ------------

def test_an_ornate_card_whose_badge_reads_as_a_digit_is_still_an_e():
    """The failure that cost a pick on the real corpus, in miniature.

    An E card carrying a badge Tesseract reports as ``8``. The border is
    unmistakable, so the card must come back as an E with no print number --
    not as a numbered card worth ranking above one.
    """
    from gacha_vision.analyze import analyze_cards
    from gacha_vision.synth import draw_card, draw_spawn

    img = draw_spawn([{"tier": FrameTier.E, "badge": "8"},
                      {"tier": FrameTier.E, "badge": "8"}])
    cards = analyze_cards(img, expected=2, read_names=False)

    assert [c.frame for c in cards] == [FrameTier.E, FrameTier.E]
    assert all(c.no_number for c in cards)
    assert all(c.print_no is None for c in cards)


def test_a_badge_that_contradicts_the_border_is_still_flagged_for_review():
    """Trusting the border must not mean resolving the conflict silently.

    The border wins, but a badge that disagreed with it is the signal that
    something was misread -- on the real corpus it fired on 22 cards and was a
    genuine badge error every time. Keep it visible.
    """
    from gacha_vision.analyze import analyze_cards
    from gacha_vision.synth import draw_spawn

    lying = analyze_cards(
        draw_spawn([{"tier": FrameTier.E, "badge": "8"},
                    {"tier": FrameTier.E, "badge": "8"}]),
        expected=2, read_names=False,
    )
    assert all(c.frame_disagrees for c in lying)

    agreeing = analyze_cards(
        draw_spawn([{"tier": FrameTier.E, "badge": "E"},
                    {"tier": FrameTier.E, "badge": "E"}]),
        expected=2, read_names=False,
    )
    assert not any(c.frame_disagrees for c in agreeing)


def test_the_right_card_is_picked_on_every_spawn_worth_claiming(rows):
    """Recall, and more importantly *which* card.

    Under the current policy a claim is spent on almost every spawn -- only
    an all-E drop is passed on -- so "did it claim" is no longer the
    interesting question. "Did it claim the card the labels say is best" is.
    """
    got = _by_spawn(rows, _replay)
    want = _by_spawn(rows, _truth_card)
    worth = [img for img in want if decide(want[img], P, {}).action is not Action.SKIP]
    assert len(worth) >= 50, f"expected most spawns to be claimable, got {len(worth)}"
    missed = [img for img in worth
              if decide(got[img], P, {}).slots != decide(want[img], P, {}).slots]
    assert missed == [], f"picked the wrong card on: {missed}"
