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
from gacha_vision.frame import guess_frame
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
    specs = [dict(tier=FrameTier.NORMAL, badge=str(100 + i)) for i in range(n)]
    cards, _ = run(specs)
    assert len(cards) == n
    assert [c.slot for c in cards] == list(range(1, n + 1))


def test_slots_are_ordered_left_to_right():
    cards, _ = run([
        dict(tier=FrameTier.NORMAL, badge="852"),
        dict(tier=FrameTier.NORMAL, badge="430"),
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

@pytest.mark.parametrize("tier", [FrameTier.NORMAL, FrameTier.E])
@pytest.mark.parametrize("hue", [5, 95])
def test_pixel_guess_round_trips_the_two_known_frames(tier, hue):
    """guess_frame only ever answers NORMAL or E -- it cannot invent OTHER."""
    assert guess_frame(draw_card(tier=tier, badge="42", art_hue=hue))[0] is tier


def test_the_e_frame_measures_as_more_ornate_than_normal():
    normal = guess_frame(draw_card(tier=FrameTier.NORMAL, badge="42"))[1]["ornateness"]
    e = guess_frame(draw_card(tier=FrameTier.E, badge="E"))[1]["ornateness"]
    assert normal < e


def test_badge_overrides_the_border_and_the_conflict_is_flagged():
    """An ornate border with a numeric badge is NORMAL, and gets flagged.

    The badge is the game's definition of the frame, so it wins. The
    disagreement is still surfaced, because it means either OCR slipped or
    this is a frame we have not catalogued.
    """
    img = spawn([dict(tier=FrameTier.E, badge="430"), dict(tier=FrameTier.NORMAL, badge="852")])
    cards = analyze_cards(img, expected=2, layout="auto", read_names=False)
    assert cards[0].print_no == 430
    assert cards[0].frame is FrameTier.NORMAL           # badge wins
    assert cards[0].frame_features["pixel_frame"] == FrameTier.E.value
    assert cards[0].frame_disagrees
    assert not cards[1].frame_disagrees


# --- whole-pipeline decisions -------------------------------------------

def test_low_print_beats_high_print():
    _, d = run([
        dict(tier=FrameTier.NORMAL, badge="14"),
        dict(tier=FrameTier.NORMAL, badge="852"),
    ])
    assert d.action is Action.CLAIM and d.slots == [1]


def test_two_low_prints_take_both():
    _, d = run([
        dict(tier=FrameTier.NORMAL, badge="7"),
        dict(tier=FrameTier.NORMAL, badge="12"),
    ])
    assert d.action is Action.CLAIM_BOTH and sorted(d.slots) == [1, 2]


def test_two_junk_cards_are_skipped():
    _, d = run([
        dict(tier=FrameTier.NORMAL, badge="E"),
        dict(tier=FrameTier.NORMAL, badge="1584"),
    ])
    assert d.action is Action.SKIP


def test_famous_character_rescues_a_junk_spawn():
    """Observed case: both cards weak, but one is a watchlist favourite."""
    wl = {normalise("HIRO"): 95}
    specs = [
        dict(tier=FrameTier.NORMAL, badge="E", character="HIRO", series="FRANXX"),
        dict(tier=FrameTier.NORMAL, badge="1584", character="EIJUN", series="ACE DIAMOND"),
    ]
    img = spawn(specs)
    cards = analyze_cards(img, expected=2, layout="auto", read_names=False)
    cards[0].character, cards[1].character = "HIRO", "EIJUN"
    d = decide(cards, P, wl)
    assert d.action is Action.CLAIM and d.slots == [1]


def test_a_decorated_border_no_longer_rescues_a_junk_print():
    """Regression: this pair used to CLAIM on the ornate frame alone."""
    _, d = run([
        dict(tier=FrameTier.E, badge="900"),
        dict(tier=FrameTier.NORMAL, badge="430"),
    ])
    assert d.action is Action.SKIP


def test_column_layout_matches_auto_layout():
    specs = [dict(tier=FrameTier.OTHER, badge="14"), dict(tier=FrameTier.NORMAL, badge="852")]
    img = spawn(specs)
    a = decide(analyze_cards(img, expected=2, layout="auto", read_names=False), P, {})
    c = decide(analyze_cards(img, expected=2, layout="columns", read_names=False), P, {})
    assert a.action is c.action and a.slots == c.slots


def test_decision_accuracy_over_a_scenario_grid():
    """Aggregate check: the pipeline must get the ACTION right every time."""
    cases = [
        ([("14", FrameTier.NORMAL), ("852", FrameTier.NORMAL)], Action.CLAIM, [1]),
        ([("852", FrameTier.NORMAL), ("14", FrameTier.NORMAL)], Action.CLAIM, [2]),
        ([("7", FrameTier.NORMAL), ("12", FrameTier.NORMAL)], Action.CLAIM_BOTH, [1, 2]),
        ([("3", FrameTier.NORMAL), ("20", FrameTier.NORMAL)], Action.CLAIM_BOTH, [1, 2]),
        ([("E", FrameTier.NORMAL), ("1584", FrameTier.NORMAL)], Action.SKIP, []),
        ([("9999", FrameTier.NORMAL), ("E", FrameTier.NORMAL)], Action.SKIP, []),
        # Ornate border, junk print: the border must not carry the decision.
        ([("900", FrameTier.E), ("430", FrameTier.NORMAL)], Action.SKIP, []),
        ([("430", FrameTier.NORMAL), ("900", FrameTier.E)], Action.SKIP, []),
    ]
    wrong = []
    for badges, want_action, want_slots in cases:
        specs = [dict(tier=t, badge=b) for b, t in badges]
        _, d = run(specs)
        if d.action is not want_action or sorted(d.slots) != want_slots:
            wrong.append((badges, want_action.value, want_slots, d.action.value, d.slots))
    assert not wrong, "decision mismatches:\n" + "\n".join(map(str, wrong))
