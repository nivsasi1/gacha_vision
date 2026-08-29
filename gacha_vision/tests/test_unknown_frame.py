"""Detecting a frame the catalogue has never seen.

The game has two frames in the corpus -- `E` and `NORMAL` -- and rarer ones
that nobody here has collected enough of to describe. Trying to recognise
those directly is hopeless with one example; recognising that a border
matches *neither known frame* is not, and is the same answer for the purpose
that matters: an uncatalogued frame is claimed on sight, whatever its print.

The evidence this rests on, measured leave-one-card-out over 182 real cards:
the 181 catalogued ones sit at most 0.102 from the nearest known frame
(median 0.005), and the one wearing an uncatalogued frame sits at 0.264 --
2.6x further than the worst catalogued card.

That is one positive example. The threshold is set with a wide margin on
both sides for exactly that reason, and a second example of any unfamiliar
frame should be added to the corpus before trusting it further.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from gacha_vision.config import Policy
from gacha_vision.frame import guess_frame, resolve_frame
from gacha_vision.models import Action, Card, FrameTier
from gacha_vision.rank import decide

DATA = Path(__file__).parent / "data"


def test_the_blue_water_frame_is_recognised_as_uncatalogued():
    img = cv2.imread(str(DATA / "card_uncatalogued_frame.png"))
    assert img is not None
    tier, _ = guess_frame(img)
    assert tier is FrameTier.OTHER


def test_an_ordinary_card_is_not_flagged_as_uncatalogued():
    img = cv2.imread(str(DATA / "card_print_1550.png"))
    assert img is not None
    tier, _ = guess_frame(img)
    assert tier is not FrameTier.OTHER


def test_no_catalogued_card_in_the_corpus_is_flagged():
    """A false positive claims a junk spawn, so the whole corpus must stay
    inside the known-frame boundary."""
    from gacha_vision.frame import frame_atlas_samples, ring_descriptor  # noqa: F401

    desc, labels, cards = frame_atlas_samples()
    assert len(desc) >= 150, f"atlas too small to trust: {len(desc)}"
    # Every stored reference is by construction a catalogued frame; held out
    # against the rest, none may look uncatalogued.
    from gacha_vision.frame import UNKNOWN_FRAME_DISTANCE

    flagged = []
    for i in range(len(desc)):
        rest = np.arange(len(desc)) != i
        d = 1.0 - desc[rest] @ desc[i]
        if d.min() > UNKNOWN_FRAME_DISTANCE:
            flagged.append(str(cards[i]))
    assert flagged == [], f"catalogued cards flagged as unknown: {flagged}"


def test_an_uncatalogued_frame_is_claimed_whatever_its_print():
    """The whole point: a rare frame is worth a pick even on a bad number."""
    odd = Card(slot=1, print_no=9999, frame=FrameTier.OTHER)
    plain = Card(slot=2, print_no=12, frame=FrameTier.NORMAL)
    d = decide([odd, plain], Policy(), {})
    assert 1 in d.slots, d.reasons


def test_an_uncatalogued_frame_keeps_its_print_number():
    """OTHER describes the border, not the badge -- a number still reads."""
    tier, print_no, no_number = resolve_frame(14, False, FrameTier.OTHER.value)
    assert tier is FrameTier.OTHER
    assert print_no == 14 and not no_number


def test_an_uncatalogued_frame_with_no_readable_badge_is_still_claimed():
    """cards30's badge sits somewhere else on that frame; the card is still
    the best in the corpus and must not be lost to an unreadable number."""
    odd = Card(slot=1, print_no=None, no_number=False, frame=FrameTier.OTHER)
    plain = Card(slot=2, print_no=1500, frame=FrameTier.NORMAL)
    assert 1 in decide([odd, plain], Policy(), {}).slots
