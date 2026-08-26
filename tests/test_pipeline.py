"""End-to-end tests: rendered spawn image -> Decision.

The metric that matters is *decision* accuracy, not character accuracy. A
badge misread from #1 to #4 changes nothing -- both are claimed. A misread
that flips CLAIM to SKIP, or picks the wrong slot, is a real loss. These
tests assert on the decision.
"""

from __future__ import annotations

import cv2
import pytest

from gacha_vision.analyze import analyze_cards
from gacha_vision.config import Policy, normalise
from gacha_vision.frame import classify_frame
from gacha_vision.models import Action, FrameTier
from gacha_vision.ocr import read_badge
from gacha_vision.rank import decide
from gacha_vision.synth import draw_card, draw_spawn

P = Policy()


def spawn(specs):
    return draw_spawn(specs)


def run(specs, watchlist=None):
    img = spawn(specs)
    cards = analyze_cards(img, expected=len(specs), layout="auto", read_names=False)
    return cards, decide(cards, P, watchlist or {})


# --- segmentation --------------------------------------------------------

@pytest.mark.parametrize("n", [2, 3, 4])
def test_finds_every_card_in_the_spawn(n):
    specs = [dict(tier=FrameTier.COMMON, badge=str(100 + i)) for i in range(n)]
    cards, _ = run(specs)
    assert len(cards) == n
    assert [c.slot for c in cards] == list(range(1, n + 1))


def test_slots_are_ordered_left_to_right():
    cards, _ = run([
        dict(tier=FrameTier.COMMON, badge="852"),
        dict(tier=FrameTier.COMMON, badge="430"),
    ])
    assert cards[0].print_no == 852 and cards[1].print_no == 430


# --- badge OCR -----------------------------------------------------------

@pytest.mark.parametrize("badge", ["3", "7", "12", "14", "20", "42", "430", "852", "1584", "9999"])
def test_reads_numeric_badges(badge):
    assert read_badge(draw_card(badge=badge))["print_no"] == int(badge)


@pytest.mark.parametrize("tier", list(FrameTier)[1:])
def test_reads_e_badge_as_no_number(tier):
    r = read_badge(draw_card(tier=tier, badge="E"))
    assert r["no_number"] is True and r["print_no"] is None


def test_never_reports_no_number_for_a_numbered_card():
    """The costliest error: a good card written off as an E."""
    for badge in ["1", "2", "7", "14", "42", "430", "1584"]:
        for tier in list(FrameTier)[1:]:
            assert read_badge(draw_card(tier=tier, badge=badge))["no_number"] is False


def test_confidence_is_reported():
    assert 0.0 < read_badge(draw_card(badge="852"))["confidence"] <= 1.0


# --- frame classification -----------------------------------------------

@pytest.mark.parametrize("tier", list(FrameTier)[1:])
@pytest.mark.parametrize("hue", [5, 95])
def test_frame_tier_round_trips(tier, hue):
    assert classify_frame(draw_card(tier=tier, badge="42", art_hue=hue))[0] is tier


def test_rarity_index_is_monotonic_across_tiers():
    idx = [
        classify_frame(draw_card(tier=t, badge="42"))[1]["rarity_index"]
        for t in [FrameTier.COMMON, FrameTier.UNCOMMON, FrameTier.RARE, FrameTier.HOLO]
    ]
    assert idx == sorted(idx), f"tiers not monotonic: {idx}"


# --- whole-pipeline decisions -------------------------------------------

def test_low_print_beats_high_print():
    _, d = run([
        dict(tier=FrameTier.COMMON, badge="14"),
        dict(tier=FrameTier.COMMON, badge="852"),
    ])
    assert d.action is Action.CLAIM and d.slots == [1]


def test_two_low_prints_take_both():
    _, d = run([
        dict(tier=FrameTier.COMMON, badge="7"),
        dict(tier=FrameTier.COMMON, badge="12"),
    ])
    assert d.action is Action.CLAIM_BOTH and sorted(d.slots) == [1, 2]


def test_two_junk_cards_are_skipped():
    _, d = run([
        dict(tier=FrameTier.COMMON, badge="E"),
        dict(tier=FrameTier.COMMON, badge="1584"),
    ])
    assert d.action is Action.SKIP


def test_famous_character_rescues_a_junk_spawn():
    """Observed case: both cards weak, but one is a watchlist favourite."""
    wl = {normalise("HIRO"): 95}
    specs = [
        dict(tier=FrameTier.COMMON, badge="E", character="HIRO", series="FRANXX"),
        dict(tier=FrameTier.COMMON, badge="1584", character="EIJUN", series="ACE DIAMOND"),
    ]
    img = spawn(specs)
    cards = analyze_cards(img, expected=2, layout="auto", read_names=False)
    cards[0].character, cards[1].character = "HIRO", "EIJUN"
    d = decide(cards, P, wl)
    assert d.action is Action.CLAIM and d.slots == [1]


def test_holo_frame_is_claimed_despite_a_bad_print():
    _, d = run([
        dict(tier=FrameTier.HOLO, badge="900"),
        dict(tier=FrameTier.COMMON, badge="430"),
    ])
    assert d.action is Action.CLAIM and d.slots == [1]


def test_column_layout_matches_auto_layout():
    specs = [dict(tier=FrameTier.RARE, badge="14"), dict(tier=FrameTier.COMMON, badge="852")]
    img = spawn(specs)
    a = decide(analyze_cards(img, expected=2, layout="auto", read_names=False), P, {})
    c = decide(analyze_cards(img, expected=2, layout="columns", read_names=False), P, {})
    assert a.action is c.action and a.slots == c.slots


def test_decision_accuracy_over_a_scenario_grid():
    """Aggregate check: the pipeline must get the ACTION right every time."""
    cases = [
        ([("14", FrameTier.COMMON), ("852", FrameTier.COMMON)], Action.CLAIM, [1]),
        ([("852", FrameTier.COMMON), ("14", FrameTier.COMMON)], Action.CLAIM, [2]),
        ([("7", FrameTier.COMMON), ("12", FrameTier.COMMON)], Action.CLAIM_BOTH, [1, 2]),
        ([("3", FrameTier.UNCOMMON), ("20", FrameTier.COMMON)], Action.CLAIM_BOTH, [1, 2]),
        ([("E", FrameTier.COMMON), ("1584", FrameTier.COMMON)], Action.SKIP, []),
        ([("9999", FrameTier.COMMON), ("E", FrameTier.COMMON)], Action.SKIP, []),
        ([("900", FrameTier.HOLO), ("430", FrameTier.COMMON)], Action.CLAIM, [1]),
        ([("430", FrameTier.COMMON), ("900", FrameTier.HOLO)], Action.CLAIM, [2]),
    ]
    wrong = []
    for badges, want_action, want_slots in cases:
        specs = [dict(tier=t, badge=b) for b, t in badges]
        _, d = run(specs)
        if d.action is not want_action or sorted(d.slots) != want_slots:
            wrong.append((badges, want_action.value, want_slots, d.action.value, d.slots))
    assert not wrong, "decision mismatches:\n" + "\n".join(map(str, wrong))
