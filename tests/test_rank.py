"""Ranking policy tests -- the stated rules, written as assertions."""

from __future__ import annotations

import pytest

from gacha_vision.config import Policy, normalise
from gacha_vision.models import Action, Card, FrameTier
from gacha_vision.rank import decide, score_card

P = Policy()


def card(slot, print_no=None, no_number=False, frame=FrameTier.COMMON, character="", series=""):
    return Card(slot=slot, print_no=print_no, no_number=no_number,
                frame=frame, character=character, series=series)


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
        card(1, no_number=True, frame=FrameTier.COMMON, character="Hiro"),
        card(2, print_no=1584, frame=FrameTier.COMMON, character="Eijun"),
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


# --- rule: better frame wins when prints are comparable -----------------

def test_frame_breaks_a_print_tie():
    cards = [card(1, 300, frame=FrameTier.COMMON), card(2, 300, frame=FrameTier.HOLO)]
    d = decide(cards, P, {})
    assert d.slots == [2]


def test_frame_tiers_are_ordered():
    tiers = [FrameTier.COMMON, FrameTier.UNCOMMON, FrameTier.RARE, FrameTier.HOLO]
    scores = [P.frame_score(t) for t in tiers]
    assert scores == sorted(scores)


# --- must-claim overrides ----------------------------------------------

def test_holo_is_always_claimed_even_with_a_terrible_print():
    d = decide([card(1, 99999, frame=FrameTier.HOLO)], P, {})
    assert d.action is Action.CLAIM


def test_print_one_is_always_claimed():
    d = decide([card(1, 1, frame=FrameTier.COMMON)], P, {})
    assert d.action is Action.CLAIM


# --- skip ---------------------------------------------------------------

def test_junk_spawn_is_skipped():
    cards = [card(1, 4200, frame=FrameTier.COMMON), card(2, no_number=True, frame=FrameTier.COMMON)]
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
