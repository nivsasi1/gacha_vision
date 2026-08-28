"""The name reader, against the real name bands of all 182 corpus cards."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from gacha_vision.names import prepare_band, segment_lines

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def corpus():
    bands = np.load(DATA / "name_bands.npz")
    with (DATA / "name_truth.csv").open(encoding="utf-8") as f:
        truth = {r["card"]: (r["true_character"], r["true_series"])
                 for r in csv.DictReader(f)}
    assert len(truth) == 182
    # Task 1 review finding: nothing guarded that the npz and csv key sets
    # actually agree. They must, or a card silently drops out of one side.
    assert set(bands.files) == set(truth.keys())
    return bands, truth


def test_finds_at_least_one_line_on_every_card(corpus):
    bands, truth = corpus
    # cards33.png#1 genuinely carries no name text (empty truth), so it
    # cannot be expected to yield a line -- exclude it from this assertion.
    named = [c for c, (ch, se) in truth.items() if ch]
    empty = [c for c in named if not segment_lines(prepare_band(bands[c]))]
    assert empty == [], f"no text found on {len(empty)} cards: {empty[:10]}"


def test_line_count_covers_the_wrapped_series_names(corpus):
    """A series like 'I've Been Killing Slimes for 300 Years...' wraps to
    three lines, so a two-line assumption drops most of it."""
    bands, truth = corpus
    wrapped = [c for c, (ch, se) in truth.items() if len(se) > 40]
    assert wrapped, "corpus should contain long wrapped series names"
    thin = [c for c in wrapped if len(segment_lines(prepare_band(bands[c]))) < 3]
    assert len(thin) <= len(wrapped) * 0.2, (
        f"{len(thin)}/{len(wrapped)} long series found on fewer than 3 lines")


def test_top_line_component_count_tracks_the_character_name(corpus):
    """Line *counts* alone don't prove segmentation is looking at the real
    text -- a stub that returns 18 fake single-box lines for every card,
    ignoring its input, passes both tests above. Bind the topmost line (the
    character name) to the label: its component count should scale with the
    number of letters in `true_character`. Glyphs aren't split individually
    at this stage (that's Task 3), so touching letters fuse into fewer
    components -- the count is a lower bound, not an exact match. 0.3x the
    letter count is a threshold real segmentation clears on ~90% of the
    corpus (measured directly against name_truth.csv); a fixed-shape stub
    that ignores the image clears it on essentially none."""
    bands, truth = corpus
    named = [c for c, (ch, se) in truth.items() if ch]
    thin = []
    for c in named:
        ch, _ = truth[c]
        n_letters = len(ch.replace(" ", ""))
        lines = segment_lines(prepare_band(bands[c]))
        top_count = len(lines[0]) if lines else 0
        if top_count < 0.3 * n_letters:
            thin.append(c)
    assert len(thin) <= len(named) * 0.15, (
        f"{len(thin)}/{len(named)} cards' topmost line component count "
        f"doesn't track the character name's letter count: {thin[:10]}")


def test_word_gap_falls_between_the_two_words(corpus):
    """Counting *how many* gaps a line has isn't enough -- a stub that
    always inserts exactly one fixed gap into a fixed-shape output, ignoring
    `work_gray`/`line` entirely, would clear that on every two-word card.
    Bind the gap's *position*: the glyph count on each side must match each
    word's own letter count, not just sum to the total. 0.20x is measured
    directly against the real implementation (0.299 on the corpus); a
    shape-only stub cannot clear it since its gap position never tracks the
    actual word split."""
    from gacha_vision.names import split_line
    bands, truth = corpus
    two_word = [c for c, (ch, _) in truth.items() if ch.count(" ") == 1]
    ok = 0
    for card in two_word:
        character, _ = truth[card]
        w1, w2 = character.split(" ")
        work = prepare_band(bands[card])
        lines = segment_lines(work)
        if not lines:
            continue
        out = split_line(work, lines[0])
        none_idx = [i for i, g in enumerate(out) if g is None]
        if len(none_idx) != 1:
            continue
        before = sum(1 for g in out[:none_idx[0]] if g is not None)
        after = sum(1 for g in out[none_idx[0] + 1:] if g is not None)
        if before == len(w1) and after == len(w2):
            ok += 1
    assert ok >= 0.20 * len(two_word), (
        f"only {ok}/{len(two_word)} two-word names split at the right glyph "
        f"counts on each side of the gap")


# --------------------------------------------------------------------------
# Task R2: the free-running reader's accuracy, leave-one-card-out.


def _levenshtein_cer(pred: str, want: str) -> float:
    """Character error rate: Levenshtein distance over the reference
    length. Matches an empty prediction against a non-empty reference as a
    full-length edit (CER 1.0), same convention the R1-era spec draft used.
    """
    if not want:
        return 0.0 if not pred else 1.0
    prev = list(range(len(pred) + 1))
    for j, wc in enumerate(want, 1):
        cur = [j]
        for i, pc in enumerate(pred, 1):
            cur.append(min(prev[i] + 1, cur[i - 1] + 1, prev[i - 1] + (pc != wc)))
        prev = cur
    return prev[len(pred)] / len(want)


@pytest.fixture(scope="module")
def loo_reads(corpus):
    """Every named card (cards33.png#1 excluded -- no name text) read with
    its *own* atlas templates dropped first. This is the measurement the
    whole task is about, so every test below shares one computation of it.

    Read through `_read_names_raw`, not the public `read_names_from_band`:
    gating a read to "" below the trust floor can only match or worsen its
    edit distance to the truth (an empty guess costs a full-length edit; a
    wrong-but-close guess usually costs less), so scoring accuracy *through*
    the gate would reward lowering MIN_TRUSTED_MATCH to reject everything --
    backwards from what the floor is for. This measures the reader itself;
    test_read_names_from_band_gates_low_confidence_reads_to_empty and
    test_read_names_from_band_passes_through_reads_at_or_above_the_floor
    below separately check the gate.
    """
    from gacha_vision.names import _read_names_raw
    bands, truth = corpus
    out = []
    for card, (want_char, want_series) in truth.items():
        if not want_char:
            continue  # cards33.png#1: no name printed on the card at all
        got = _read_names_raw(bands[card], exclude_card=card)
        out.append({
            "card": card,
            "want_char": want_char, "got_char": got.character,
            "want_series": want_series, "got_series": got.series,
            "confidence": got.confidence,
            "cer": _levenshtein_cer(got.character, want_char),
        })
    return out


def test_character_names_read_accurately_leave_one_card_out(loo_reads):
    """Task R2's actual deliverable. Global Constraints' bar: character
    -level accuracy (1-CER) >= 0.95, exact match >= 0.85. Series exact
    -match is reported only -- no floor.

    Per the task brief: if this comes in under the bar, the assertions stay
    as written -- a truthful low number, not a loosened threshold, is the
    deliverable in that case. See task-R2-report.md for the full breakdown
    if so.
    """
    n = len(loo_reads)
    assert n >= 180, f"expected ~181 named cards, got {n}"

    exact = sum(r["got_char"] == r["want_char"] for r in loo_reads)
    series_exact = sum(r["got_series"] == r["want_series"] for r in loo_reads)
    accuracy = 1.0 - sum(r["cer"] for r in loo_reads) / n
    exact_rate = exact / n
    series_exact_rate = series_exact / n

    worst = sorted(loo_reads, key=lambda r: (-r["cer"], r["card"]))[:15]
    print(f"\ncharacter-level accuracy (1-CER): {accuracy:.4f}")
    print(f"character exact-match: {exact_rate:.4f} ({exact}/{n})")
    print(f"series exact-match (report only, no bar): {series_exact_rate:.4f} "
          f"({series_exact}/{n})")
    print("\n15 worst cards by CER:")
    for r in worst:
        print(f"  {r['card']:16s} cer={r['cer']:.2f} conf={r['confidence']:.3f}  "
              f"want={r['want_char']!r:28s} got={r['got_char']!r}")

    assert accuracy >= 0.95, (
        f"character-level accuracy {accuracy:.3f} over {n} cards, bar is 0.95 "
        f"-- see task-R2-report.md for the failure breakdown")
    assert exact_rate >= 0.85, (
        f"exact match {exact}/{n} = {exact_rate:.3f}, bar is 0.85 "
        f"-- see task-R2-report.md for the failure breakdown")


def test_read_names_from_band_gates_low_confidence_reads_to_empty(corpus, loo_reads):
    """The public entry point gates on MIN_TRUSTED_MATCH: a read below it
    comes back as empty strings rather than a guess, same contract as
    digits.read_print_number_from_window, with the confidence value itself
    still reported so a caller can see how far below the floor it was.

    Every leave-one-card-out read in this corpus is below the floor (see
    MIN_TRUSTED_MATCH's own comment in names.py for why it's calibrated
    that way, and task-R2-report.md for the numbers) -- exercised here
    through the real `read_names_from_band` entry point, on a sample, not
    re-derived from loo_reads' arithmetic.
    """
    from gacha_vision.names import MIN_TRUSTED_MATCH, read_names_from_band
    bands, _truth = corpus
    below = [r for r in loo_reads if r["confidence"] < MIN_TRUSTED_MATCH]
    assert len(below) == len(loo_reads), (
        f"expected every read in this corpus to sit below the trust floor "
        f"(see MIN_TRUSTED_MATCH's calibration note); {len(loo_reads) - len(below)} did not")
    for r in below[:15]:
        got = read_names_from_band(bands[r["card"]], exclude_card=r["card"])
        assert got.character == "" and got.series == "", (r["card"], got)
        assert got.confidence == r["confidence"], r["card"]


def test_read_names_from_band_passes_through_reads_at_or_above_the_floor(monkeypatch):
    """The other half of the gate. This corpus has no leave-one-card-out
    read at or above MIN_TRUSTED_MATCH to exercise this branch against (the
    previous test), so it's checked directly: force a high-confidence read
    and confirm the gate lets it through unchanged, rather than asserting a
    corpus property that isn't true.
    """
    import gacha_vision.names as names
    trusted = names.NameRead("Test", "Series", round(names.MIN_TRUSTED_MATCH + 0.001, 3))
    monkeypatch.setattr(names, "_read_names_raw",
                         lambda band, exclude_card=None: trusted)
    got = names.read_names_from_band(np.zeros((10, 10), dtype=np.uint8))
    assert got == trusted
