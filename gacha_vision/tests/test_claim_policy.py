"""The claim policy, stated as the cases it was specified from.

The rules, in the owner's words: one card per spawn normally, two only when
both are genuinely good; a numbered card always beats an `E`; and a spawn of
nothing but `E` comes back empty so a human can pick at random rather than
have the reader pretend it chose.

These are ranking tests, not vision tests -- they build Cards directly, so a
change in policy fails here loudly and a change in OCR does not.
"""

from __future__ import annotations

import pytest

from gacha_vision.config import Policy
from gacha_vision.models import Action, Card, FrameTier
from gacha_vision.rank import decide

P = Policy()


def numbered(slot: int, print_no: int) -> Card:
    return Card(slot=slot, print_no=print_no, frame=FrameTier.NORMAL)


def e_card(slot: int) -> Card:
    return Card(slot=slot, no_number=True, frame=FrameTier.E)


def slots(cards) -> list[int]:
    return sorted(decide(cards, P, {}).slots)


# --- the four spawns the policy was specified from ------------------------

def test_cards30_takes_the_low_print_and_leaves_the_high_one():
    """#14 against #852: one card, the good one."""
    assert slots([numbered(1, 14), numbered(2, 852)]) == [1]


def test_88_takes_the_numbered_card_over_the_e():
    assert slots([e_card(1), numbered(2, 20)]) == [2]


def test_47_takes_a_print_under_200():
    """#99 is junk by the old floor of ~60 and worth taking by this one."""
    assert slots([numbered(1, 99), e_card(2)]) == [1]


def test_37_takes_the_only_numbered_card_however_high_its_print():
    """#2155 is a bad card. It is still better than two E cards, and a
    spawn is not worth passing on entirely for lack of a good one."""
    assert slots([e_card(1), e_card(2), numbered(3, 2155)]) == [3]


# --- the boundaries -------------------------------------------------------

def test_two_cards_under_200_are_both_worth_a_pick():
    assert slots([numbered(1, 150), numbered(2, 180)]) == [1, 2]


def test_only_one_card_under_200_takes_only_that_one():
    assert slots([numbered(1, 150), numbered(2, 900)]) == [1]


def test_a_spawn_of_nothing_but_e_comes_back_empty():
    """Deliberate: there is nothing to choose between, so the reader says so
    instead of inventing a preference."""
    d = decide([e_card(1), e_card(2)], P, {})
    assert d.action is Action.SKIP
    assert d.slots == []


def test_a_lone_e_card_is_also_empty():
    """A single E card is still a spawn of nothing but E."""
    assert slots([e_card(1)]) == []


def test_a_lone_numbered_card_is_taken_whatever_its_print():
    assert slots([numbered(1, 3000)]) == [1]


def test_never_more_than_two_cards():
    four = [numbered(i, 10 * i) for i in range(1, 5)]
    assert len(slots(four)) <= 2


def test_the_lowest_prints_win_when_more_than_two_qualify():
    got = slots([numbered(1, 190), numbered(2, 12), numbered(3, 60)])
    assert got == [2, 3], got


def test_an_unreadable_badge_ranks_below_a_readable_print():
    """A number we could not read might be anything; a number we did read is
    known. Prefer the known one rather than gambling."""
    unreadable = Card(slot=1, print_no=None, no_number=False,
                      frame=FrameTier.NORMAL, print_trusted=False)
    assert slots([unreadable, numbered(2, 1500)]) == [2]


def test_an_unreadable_badge_still_beats_an_e():
    """It is a numbered card, so it is not junk by definition the way E is."""
    unreadable = Card(slot=1, print_no=None, no_number=False,
                      frame=FrameTier.NORMAL, print_trusted=False)
    assert slots([unreadable, e_card(2)]) == [1]


@pytest.mark.parametrize("take_both", [50, 200, 500])
def test_the_take_both_threshold_is_configurable(take_both):
    pol = Policy(take_both_max_print=take_both)
    cards = [numbered(1, 199), numbered(2, 199)]
    got = sorted(decide(cards, pol, {}).slots)
    assert got == ([1, 2] if take_both >= 199 else [1])
