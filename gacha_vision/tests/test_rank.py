"""Ranking policy tests -- the stated rules, written as assertions."""

from __future__ import annotations

import pytest

from gacha_vision.config import Policy, normalise
from gacha_vision.frame import frame_from_badge
from gacha_vision.models import Action, Card, FrameTier
from gacha_vision.rank import decide, score_card

P = Policy()


def card(slot, print_no=None, no_number=False, frame=None, character="", series="",
         trusted=True):
    """Mirror production: the badge decides the frame unless one is forced."""
    if frame is None:
        frame = frame_from_badge(print_no, no_number)
    return Card(slot=slot, print_no=print_no, no_number=no_number,
                frame=frame, character=character, series=series,
                print_trusted=trusted)


# --- a print we do not trust must not drive a decision ------------------

def test_a_low_confidence_print_is_scored_as_unreadable():
    """A shaky read is not evidence of a low number."""
    shaky = score_card(card(1, 16, trusted=False), P, {})
    solid = score_card(card(2, 16, trusted=True), P, {})
    assert shaky.print_score < solid.print_score
    assert shaky.print_score == P.score_unreadable


def test_low_confidence_prints_never_trigger_take_both():
    """Regression from 91 real spawns.

    Prints there run 1600-2200, yet 59% of spawns came back CLAIM_BOTH --
    the OCR was misreading long numbers as short ones and the take-both
    rule was acting on reads the pipeline itself flagged as unreliable.
    """
    d = decide([card(1, 16, trusted=False), card(2, 5, trusted=False)], P, {})
    assert d.action is not Action.CLAIM_BOTH


def test_a_low_confidence_print_one_is_not_auto_claimed():
    # Two cards, because a lone card is claimed on sight regardless.
    d = decide([card(1, 1, trusted=False), card(2, 9999)], P, {})
    assert d.slots != [1]


def test_trust_only_gates_the_print_not_the_watchlist():
    """Fame is read from text, not the badge, so it survives a shaky number."""
    wl = {normalise("Hiro"): 95}
    d = decide([card(1, 1655, trusted=False, character="Hiro")], P, wl)
    assert d.action is Action.CLAIM


# --- rule: lower print number is better ---------------------------------

@pytest.mark.parametrize("lo,hi", [(1, 2), (14, 852), (20, 21), (99, 1584)])
def test_lower_print_scores_higher(lo, hi):
    a = score_card(card(1, lo), P, {})
    b = score_card(card(2, hi), P, {})
    assert a.print_score > b.print_score


def test_claims_the_lower_print_of_two_similar_cards():
    d = decide([card(1, 14), card(2, 852)], P, {})
    assert d.action is Action.CLAIM
    assert d.slots == [1]


# --- rule: "E" (no number) is bad, but not disqualifying ----------------

def test_e_card_scores_below_any_real_print():
    e = score_card(card(1, no_number=True), P, {})
    worst = score_card(card(2, 99999), P, {})
    assert e.print_score < worst.print_score


def test_e_card_never_triggers_take_both():
    # Two E cards must not be read as "two cards under 20".
    d = decide([card(1, no_number=True), card(2, no_number=True)], P, {})
    assert d.action is not Action.CLAIM_BOTH


def test_fame_can_rescue_an_e_card_over_a_high_print():
    # The observed case: both cards weak, but the E card is a famous
    # character, so it still wins.
    watchlist = {normalise("Hiro"): 92}
    cards = [
        card(1, no_number=True, character="Hiro"),
        card(2, print_no=1584, character="Eijun"),
    ]
    d = decide(cards, P, watchlist)
    assert d.action is Action.CLAIM
    assert d.slots == [1]


# --- rule: both under 20 -> spend the extra pick ------------------------

def test_both_under_twenty_takes_both():
    d = decide([card(1, 7), card(2, 12)], P, {})
    assert d.action is Action.CLAIM_BOTH
    assert sorted(d.slots) == [1, 2]


def test_only_one_under_twenty_takes_one():
    d = decide([card(1, 7), card(2, 400)], P, {})
    assert d.action is Action.CLAIM
    assert d.slots == [1]


def test_boundary_twenty_is_inclusive():
    d = decide([card(1, 20), card(2, 20)], P, {})
    assert d.action is Action.CLAIM_BOTH


def test_boundary_twentyone_is_not():
    d = decide([card(1, 21), card(2, 21)], P, {})
    assert d.action is not Action.CLAIM_BOTH


def test_take_both_respects_max_claims():
    p = P.with_overrides(max_claims=2)
    d = decide([card(1, 3), card(2, 4), card(3, 5)], p, {})
    assert d.action is Action.CLAIM_BOTH
    assert len(d.slots) == 2


# --- frames: E and NORMAL are both commons ------------------------------

def test_badge_decides_the_frame():
    assert card(1, print_no=1655).frame is FrameTier.NORMAL
    assert card(1, no_number=True).frame is FrameTier.E
    assert card(1).frame is FrameTier.UNKNOWN          # badge unreadable


def test_the_two_known_frames_score_almost_the_same():
    """Both are the game's commons, so neither may swing a decision."""
    gap = abs(P.frame_score(FrameTier.NORMAL) - P.frame_score(FrameTier.E))
    assert gap * P.w_frame < 1.0


def test_an_unfamiliar_frame_outranks_the_known_commons():
    """OTHER might be a genuinely rare frame, so it is nudged up..."""
    assert P.frame_score(FrameTier.OTHER) > P.frame_score(FrameTier.NORMAL)


def test_an_unfamiliar_frame_is_always_claimed():
    """...and a frame matching neither known one is taken on sight.

    Stated rule: an uncatalogued frame might be the rare one, so it is worth
    a pick and a human glance even on a junk print.
    """
    d = decide([card(1, 9999, frame=FrameTier.OTHER), card(2, 8888)], P, {})
    assert d.action is Action.CLAIM and d.slots == [1]


def test_a_lone_card_is_always_claimed():
    """A spawn with one card has nothing to weigh it against."""
    for c in (card(1, 9999), card(1, no_number=True)):
        assert decide([c], P, {}).action is Action.CLAIM


def test_frame_breaks_a_print_tie():
    """With prints equal the unfamiliar frame ranks higher -- but only ranks.

    Neither card clears the claim floor here, which is the point: a frame
    can order two cards without making either worth taking.
    """
    plain = score_card(card(1, 300, frame=FrameTier.NORMAL), P, {})
    other = score_card(card(2, 300, frame=FrameTier.OTHER), P, {})
    assert other.total > plain.total


# --- must-claim overrides ----------------------------------------------

def test_ornate_frame_never_rescues_an_unnumbered_card():
    """Regression, from real spawns.

    In Gachapon the ornate rainbow frames sit on the *worst* cards: every
    observed E card wore one while the numbered cards wore a plain border.
    A frame must therefore never outrank a print number, or the ranker
    picks the junk half of every spawn.
    """
    e_ornate = card(1, no_number=True, frame=FrameTier.E)
    numbered = card(2, print_no=1655, frame=FrameTier.NORMAL)
    d = decide([e_ornate, numbered], P, {})
    assert d.slots != [1], "an E card outranked a numbered card on frame alone"
    assert score_card(e_ornate, P, {}).total < score_card(numbered, P, {}).total


def test_frame_cap_applies_to_every_frame_on_an_unnumbered_card():
    base = score_card(card(1, no_number=True, frame=FrameTier.E), P, {})
    for tier in (FrameTier.NORMAL, FrameTier.OTHER):
        assert score_card(card(1, no_number=True, frame=tier), P, {}).total == base.total


def test_frame_still_counts_for_numbered_cards():
    plain = score_card(card(1, 300, frame=FrameTier.NORMAL), P, {})
    fancy = score_card(card(1, 300, frame=FrameTier.OTHER), P, {})
    assert fancy.total > plain.total


def test_frame_cap_can_be_turned_off():
    p = P.with_overrides(frame_lifts_unnumbered=True)
    assert (score_card(card(1, no_number=True, frame=FrameTier.OTHER), p, {}).total
            > score_card(card(1, no_number=True, frame=FrameTier.E), p, {}).total)


def test_print_one_is_always_claimed():
    d = decide([card(1, 1)], P, {})
    assert d.action is Action.CLAIM


# --- skip ---------------------------------------------------------------

def test_junk_spawn_is_skipped():
    cards = [card(1, 4200), card(2, no_number=True)]
    d = decide(cards, P, {})
    assert d.action is Action.SKIP
    assert d.slots == []


def test_empty_spawn_is_skipped():
    d = decide([], P, {})
    assert d.action is Action.SKIP


# --- unreadable OCR is neutral, not fatal -------------------------------

def test_unreadable_print_scores_between_e_and_a_good_print():
    unknown = score_card(card(1), P, {})
    e = score_card(card(2, no_number=True), P, {})
    good = score_card(card(3, 10), P, {})
    assert e.print_score < unknown.print_score < good.print_score


# --- watchlist matching -------------------------------------------------

def test_series_watchlist_lifts_the_card():
    wl = {normalise("Dragon Ball"): 85}
    plain = score_card(card(1, 500, series="Dragon Ball"), P, {})
    lifted = score_card(card(1, 500, series="Dragon Ball"), P, wl)
    assert lifted.total > plain.total


def test_watchlist_matching_ignores_case_and_punctuation():
    wl = {normalise("Ace of the Diamond"): 70}
    s = score_card(card(1, 500, series="ACE OF THE DIAMOND!"), P, wl)
    assert s.fame_score == 70


# --- explanations are always populated ----------------------------------

@pytest.mark.parametrize("cards", [
    [card(1, 5), card(2, 9)],
    [card(1, 900), card(2, 800)],
    [card(1, no_number=True)],
])
def test_every_decision_explains_itself(cards):
    d = decide(cards, P, {})
    assert d.reasons and d.explain().strip()
    assert len(d.scores) == len(cards)
