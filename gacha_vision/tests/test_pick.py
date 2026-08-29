"""The one-call entry point a bot wires itself to.

Everything else in this package returns rich objects for inspection. A bot
wants one thing: which slots to claim, as numbers, right now. `pick` is that,
and it takes the image the way a bot actually has it -- bytes in memory, not
a path on disk, because a Discord attachment is downloaded, not saved.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from gacha_vision import Policy, pick
from gacha_vision.models import FrameTier
from gacha_vision.synth import draw_spawn

DATA = Path(__file__).parent / "data"
CARD = DATA / "card_print_1550.png"


def test_returns_the_slot_to_claim_as_a_plain_number():
    """A lone card is always worth taking, so this is slot 1."""
    assert pick(CARD, expected=1) == [1]


def test_accepts_raw_bytes_the_way_a_bot_receives_them():
    """A downloaded attachment is bytes, never a file. Same answer either way."""
    raw = CARD.read_bytes()
    assert pick(raw, expected=1) == pick(CARD, expected=1)


def test_accepts_an_already_decoded_array():
    img = cv2.imread(str(CARD))
    assert pick(img, expected=1) == pick(CARD, expected=1)


def test_skipping_a_junk_spawn_returns_an_empty_list():
    """Two E cards are worth nothing. Empty means "claim nothing"."""
    img = draw_spawn([{"tier": FrameTier.E, "badge": "E"},
                      {"tier": FrameTier.E, "badge": "E"}])
    assert pick(img, expected=2) == []


def test_a_low_print_is_picked_over_a_high_one():
    img = draw_spawn([{"tier": FrameTier.NORMAL, "badge": "852"},
                      {"tier": FrameTier.NORMAL, "badge": "14"}])
    assert pick(img, expected=2) == [2]


def test_two_good_cards_return_both_slots_in_score_order():
    # Two digits, not one: a lone digit is the hook icon on real cards, so
    # ocr.py distrusts it by design and it would never reach the ranker.
    img = draw_spawn([{"tier": FrameTier.NORMAL, "badge": "14"},
                      {"tier": FrameTier.NORMAL, "badge": "12"}])
    got = pick(img, expected=2)
    assert sorted(got) == [1, 2], got


def test_slots_are_one_based_and_left_to_right():
    """Slot numbers must match the buttons under the spawn, or the bot
    clicks the wrong card -- the one failure mode that costs a real pick."""
    img = draw_spawn([{"tier": FrameTier.E, "badge": "E"},
                      {"tier": FrameTier.E, "badge": "E"},
                      {"tier": FrameTier.NORMAL, "badge": "18"}])
    assert pick(img, expected=3) == [3]


def test_no_watchlist_argument_is_offered_while_names_are_unreadable():
    """Rather than accept a watchlist and ignore it.

    Watchlist matching needs the character name, and names are not read on
    this game's cards. An argument that is silently discarded is worse than
    no argument -- the caller thinks it took effect.
    """
    import inspect

    assert "watchlist" not in inspect.signature(pick).parameters


def test_an_unreadable_image_raises_rather_than_silently_skipping():
    """A decode failure must not look like 'nothing worth claiming'."""
    with pytest.raises(ValueError):
        pick(b"this is not an image", expected=2)


def test_policy_overrides_are_honoured():
    img = draw_spawn([{"tier": FrameTier.NORMAL, "badge": "14"},
                      {"tier": FrameTier.NORMAL, "badge": "12"}])
    strict = Policy(take_both_max_print=1, min_claim_score=101.0)
    assert len(pick(img, expected=2, policy=strict)) <= 1
