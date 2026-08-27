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
from gacha_vision.frame import frame_from_badge, guess_frame
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


@pytest.mark.parametrize("tier", [FrameTier.NORMAL, FrameTier.E])
def test_reads_e_badge_as_no_number(tier):
    """Only the frames the game actually has. OTHER is a placeholder for a
    frame nobody has catalogued, so asserting on how it renders would test
    the fixture rather than the reader."""
    r = read_badge(draw_card(tier=tier, badge="E"))
    assert r["no_number"] is True and r["print_no"] is None


def test_never_reports_no_number_for_a_numbered_card():
    """The costliest error: a good card written off as an E.

    Only the NORMAL frame carries a number, so that is the combination
    worth guarding; an ornate frame with a print does not exist in game.
    """
    for badge in ["1", "2", "7", "14", "42", "430", "1584", "1655"]:
        card = draw_card(tier=FrameTier.NORMAL, badge=badge)
        assert read_badge(card)["no_number"] is False


@pytest.mark.parametrize("badge_h", [40, 30, 22, 18])
def test_small_badges_are_still_read(badge_h):
    """Regression from 91 real spawns.

    Real badges are ~3-4% of card height. At that scale the digits fell
    under a size filter tuned to an oversized synthetic badge, so 64% of
    real cards read as "4" -- the fixed hook icon every card shares -- at
    one repeated confidence. The strip is now scale-normalised so the
    filters mean the same thing at any card size.
    """
    for badge in ["7", "852", "1655"]:
        r = read_badge(draw_card(tier=FrameTier.NORMAL, badge=badge, badge_h=badge_h))
        assert r["no_number"] is False, f"{badge} at {badge_h}px was written off as E"
        if badge_h >= 22:
            assert r["print_no"] == int(badge)


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


def test_badge_decides_the_frame_and_a_conflict_is_flagged():
    """The badge defines the frame; the border only corroborates.

    Asserted on the Card rather than through a rendered image, because the
    conflicting case -- an ornate border carrying a print number -- is one
    the game never produces, so rendering it would test the fixtures rather
    than the rule.
    """
    from gacha_vision.models import Card

    agree = Card(slot=1, print_no=852, frame=frame_from_badge(852, False),
                 frame_features={"pixel_frame": FrameTier.NORMAL.value})
    conflict = Card(slot=2, print_no=430, frame=frame_from_badge(430, False),
                    frame_features={"pixel_frame": FrameTier.E.value})
    assert conflict.frame is FrameTier.NORMAL          # badge wins
    assert conflict.frame_disagrees
    assert not agree.frame_disagrees


# --- whole-pipeline decisions -------------------------------------------

def test_low_print_beats_high_print():
    _, d = run([
        dict(tier=FrameTier.NORMAL, badge="14"),
        dict(tier=FrameTier.NORMAL, badge="852"),
    ])
    assert d.action is Action.CLAIM and d.slots == [1]


def test_two_low_prints_take_both():
    # Two digits, not one: a lone digit is deliberately never trusted, so a
    # "#7" here would be testing that rule rather than this one.
    _, d = run([
        dict(tier=FrameTier.NORMAL, badge="17"),
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
    specs = [dict(tier=FrameTier.NORMAL, badge="14"), dict(tier=FrameTier.NORMAL, badge="852")]
    img = spawn(specs)
    a = decide(analyze_cards(img, expected=2, layout="auto", read_names=False), P, {})
    c = decide(analyze_cards(img, expected=2, layout="columns", read_names=False), P, {})
    assert a.action is c.action and a.slots == c.slots


def test_decision_accuracy_over_a_scenario_grid():
    """Aggregate check: the pipeline must get the ACTION right every time."""
    cases = [
        ([("14", FrameTier.NORMAL), ("852", FrameTier.NORMAL)], Action.CLAIM, [1]),
        ([("852", FrameTier.NORMAL), ("14", FrameTier.NORMAL)], Action.CLAIM, [2]),
        ([("17", FrameTier.NORMAL), ("12", FrameTier.NORMAL)], Action.CLAIM_BOTH, [1, 2]),
        ([("13", FrameTier.NORMAL), ("20", FrameTier.NORMAL)], Action.CLAIM_BOTH, [1, 2]),
        # A lone digit is the hook icon, so it must NOT carry the spawn.
        ([("4", FrameTier.NORMAL), ("1584", FrameTier.NORMAL)], Action.SKIP, []),
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


# --- spawns bigger than a pair -------------------------------------------
#
# Real spawns have only ever dropped two or three cards, but four is
# reported as possible, so the pipeline is exercised at that width now
# rather than discovering it the day it happens.

@pytest.mark.parametrize("n", [3, 4])
def test_a_wide_spawn_still_picks_the_best_card(n):
    """The one good print is claimed no matter how many duds surround it."""
    specs = [dict(tier=FrameTier.NORMAL, badge="1600") for _ in range(n)]
    specs[n - 1] = dict(tier=FrameTier.NORMAL, badge="18")
    cards, d = run(specs)
    assert len(cards) == n
    assert d.action is Action.CLAIM and d.slots == [n]


@pytest.mark.parametrize("n", [3, 4])
def test_a_wide_spawn_spends_at_most_the_allowed_picks(n):
    """Three good cards in one spawn still yield only the two best picks."""
    # 11 and 15 are avoided on purpose: synth.py's own hook glyph merges into
    # them (11 -> 41, 15 -> 415), which is the same artefact this reader
    # fights on real cards but would be testing the wrong thing here.
    specs = [dict(tier=FrameTier.NORMAL, badge=b) for b in ("12", "14", "16", "18")[:n]]
    _, d = run(specs)
    assert d.action is Action.CLAIM_BOTH
    assert len(d.slots) == P.max_claims
    assert sorted(d.slots) == [1, 2]          # the two lowest prints


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_a_plausible_card_count_is_not_flagged(n):
    from gacha_vision.batch import PLAUSIBLE_CARDS
    assert PLAUSIBLE_CARDS[0] <= n <= PLAUSIBLE_CARDS[1]


# --- badge candidate filtering -------------------------------------------
#
# These are the two rules fitted to the 182 labelled real cards: no print is
# longer than four digits, and a lone digit is the hook icon rather than a
# print. Unit-tested directly because the real failures they fix cannot be
# drawn by synth.py -- the game font's hook glyph is what produces them.

def test_a_five_digit_read_loses_its_leading_hook_glyph():
    from gacha_vision.ocr import _plausible_print
    assert _plausible_print("11695") == "1695"
    assert _plausible_print("12527") == "2527"


def test_a_plausible_print_is_left_alone():
    from gacha_vision.ocr import _plausible_print
    for t in ("7", "20", "852", "2850"):
        assert _plausible_print(t) == t


def test_a_five_digit_read_that_would_gain_a_leading_zero_is_rejected():
    """A genuine 10234 must not be corrupted into a spectacular #234."""
    from gacha_vision.ocr import _plausible_print
    assert _plausible_print("10234") is None
    assert _plausible_print("29999") is None
    assert _plausible_print("123456") is None


@pytest.mark.parametrize("badge", ["1", "3", "4", "7", "9"])
@pytest.mark.parametrize("blur", [0, 5, 9])
def test_a_single_digit_read_is_never_trusted(badge, blur):
    """A lone digit is the hook icon, at any confidence.

    Tesseract's certainty carries no information here: the most confident
    impostor in the labelled set read at 1.000, above every correct
    multi-digit read. So the demotion cannot be gated on confidence.
    """
    from gacha_vision.ocr import MIN_TRUSTED_CONFIDENCE, read_badge
    img = draw_card(tier=FrameTier.NORMAL, badge=badge)
    if blur:
        img = cv2.GaussianBlur(img, (blur, blur), 0)
    r = read_badge(img)
    if r["print_no"] is not None and r["print_no"] < 10:
        assert r["confidence"] < MIN_TRUSTED_CONFIDENCE, r


def test_an_untrusted_lone_digit_does_not_win_a_pick():
    """The point of the demotion: #4 must stop outranking a real print."""
    from gacha_vision.analyze import analyze_cards
    img = spawn([dict(tier=FrameTier.NORMAL, badge="4"),
                 dict(tier=FrameTier.NORMAL, badge="1584")])
    cards = analyze_cards(img, expected=2, read_names=False)
    lone = cards[0]
    assert lone.print_no == 4, "the value is kept for review, only the trust is withheld"
    assert not lone.print_trusted
    assert decide(cards, P, {}).action is Action.SKIP


def test_a_multi_digit_read_is_still_trusted_on_its_confidence():
    """The rule is narrow: two digits and up go through the ordinary gate."""
    from gacha_vision.analyze import analyze_cards
    img = spawn([dict(tier=FrameTier.NORMAL, badge="14"),
                 dict(tier=FrameTier.NORMAL, badge="852")])
    cards = analyze_cards(img, expected=2, read_names=False)
    assert cards[0].print_no == 14 and cards[0].print_trusted
    assert decide(cards, P, {}).action is Action.CLAIM
