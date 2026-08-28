"""Learn a glyph atlas by forced alignment instead of by cutting glyphs.

Free segmentation has a 55-62% ceiling on this font (see the
2026-08-28 revision in docs/superpowers/plans/2026-08-27-name-reader.md), so
instead of trying to cut a line into glyphs and then recognise them, this
script uses the fact that every card's correct text is already known
(Task 1's hand-read labels) to run forced alignment: slide each class's
current template along the line and let dynamic programming find the best
positions *for that known string*. That always produces a full placement,
which is then scored, filtered, and turned into the next round's templates
-- an EM loop:

    E-step: forced_align every usable line against the current templates.
    M-step: harvest the well-scoring placements, average per class.

reported per iteration until the mean alignment score stops improving.

Usage: python tools/build_glyph_atlas.py
Writes: gacha_vision/data/glyph_atlas.npz (templates uint8[N,GLYPH_H,GLYPH_W],
        labels str[N], card str[N])
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gacha_vision.names import prepare_band, segment_lines, split_line  # noqa: E402
from gacha_vision.names_align import (  # noqa: E402
    LINE_HEIGHT,
    line_ink_bounds,
    normalise_line,
    forced_align,
    harvest,
    mean_templates,
    match_score,
)

BANDS = ROOT / "gacha_vision" / "tests" / "data" / "name_bands.npz"
TRUTH = ROOT / "gacha_vision" / "tests" / "data" / "name_truth.csv"
OUT = ROOT / "gacha_vision" / "data" / "glyph_atlas.npz"

# EM stops once an iteration's mean score gains less than this over the
# previous one. Measured run (task-R1-report.md): mean score climbed
# 0.333 -> 0.370 -> 0.376 -> 0.379 -> 0.380 over 5 iterations, essentially
# flat by iteration 3 -- the 0.002 floor stops one iteration after that,
# rather than chasing rounding noise. MAX_ITERS is a cap regardless, so a
# pathological oscillation can't run forever.
MAX_ITERS = 8
MIN_IMPROVEMENT = 0.002

# A harvested placement below this match score is more likely a
# mis-segmented line (decorative artwork, a fused neighbouring line -- see
# names_align.py's docstring on segment_lines' ~86% reliability) than the
# real glyph, and is dropped before it can drag the class average toward
# noise. An unseeded class is harvested regardless -- see
# `align_and_harvest`'s docstring for why that's not the same risk.
# Measured against the converged atlas's own re-alignment (task-R1-report.md):
# scores range -0.22 (p5) to 0.77 (p95), median 0.405, mean 0.354 -- 0.35
# sits just below the median, keeping 56% of placements while cutting the
# worst-scoring tail rather than only the outliers.
HARVEST_MIN_SCORE = 0.35

# Final atlas template shape. Height matches every normalised line
# (LINE_HEIGHT); width is fixed so `templates` can be one rectangular
# uint8 array as the format requires. 16 is the corpus's own median
# harvested glyph width (measured at convergence: median 16.0, mean 13.9),
# so the average class is barely resampled going into storage.
GLYPH_H = LINE_HEIGHT
GLYPH_W = 16


def _load_corpus():
    bands = np.load(BANDS)
    with TRUTH.open(encoding="utf-8") as f:
        truth = {r["card"]: (r["true_character"], r["true_series"])
                 for r in csv.DictReader(f)}
    return bands, truth


def _precompute_lines(bands, truth):
    """`segment_lines` doesn't depend on the templates, so it's run once
    per card up front rather than once per EM iteration."""
    out = {}
    for card, (character, _series) in truth.items():
        if not character:
            continue
        work = prepare_band(bands[card])
        lines = segment_lines(work)
        if lines:
            out[card] = (work, lines)
    return out


def _training_texts(character: str, series: str, n_lines: int):
    """(line_index, text) pairs worth forced-aligning for one card.

    Line 0 is always the character name. The series is only used when it
    fits on exactly one line (n_lines == 2 -- character line + single
    series line): a longer series wraps at points the game's renderer
    decides, not us, so there is no reliable way to split the label to
    match a specific wrapped line without risking a silently wrong
    label -- and a wrong label would poison every glyph harvested from it.
    """
    out = [(0, character)]
    if series and n_lines == 2:
        out.append((1, series))
    return out


def seed_pool(card_lines: dict, truth: dict) -> tuple[dict[str, list[np.ndarray]], int]:
    """First-round templates from the cards where Task 2/3's naive
    per-glyph split (`split_line`) already agrees with the character name's
    letter count -- Task R1's prescribed bootstrap. `split_line`'s boxes are
    in `work_gray` coordinates; `line_ink_bounds` recovers the same crop
    origin and scale `normalise_line` used, so a box maps onto the
    normalised line correctly.
    """
    pool: dict[str, list[np.ndarray]] = defaultdict(list)
    used = 0
    for card, (character, _series) in truth.items():
        if card not in card_lines:
            continue
        work, lines = card_lines[card]
        boxes = split_line(work, lines[0])
        glyphs = [g for g in boxes if g is not None]
        letters = [c for c in character if c != " "]
        if len(glyphs) != len(letters):
            continue
        x0, y0, x1, y1 = line_ink_bounds(lines[0], work.shape)
        scale = LINE_HEIGHT / (y1 - y0)
        norm = normalise_line(work, lines[0])
        for (gx, gy, gw, _gh), ch in zip(glyphs, letters):
            nx0 = round((gx - x0) * scale)
            nx1 = round((gx + gw - x0) * scale)
            nx0, nx1 = max(0, nx0), min(norm.shape[1], nx1)
            if nx1 > nx0:
                pool[ch].append(norm[:, nx0:nx1].copy())
        used += 1
    return dict(pool), used


def align_and_harvest(card_lines: dict, truth: dict, templates: dict[str, np.ndarray]):
    """One E-step (align) + the harvesting half of the M-step: forced-align
    every usable line, score each placed glyph against its current
    template, and keep the ones that pass `HARVEST_MIN_SCORE`.

    A class with no template yet is harvested unconditionally -- there's
    nothing to score it against, and refusing to harvest it would mean it
    can *never* get a first example (the classic bootstrap problem). The
    risk this trades away -- a bad first sample for a rare class -- is
    bounded: the next iteration has a real template for it and can start
    rejecting bad placements, and `mean_templates` averaging further dilutes
    one bad early sample once more come in.

    Returns `(pool, mean_score, n_scored)` where `pool` maps class ->
    list of `(bitmap, source_card)` and `mean_score` is the mean match
    score over every *scored* placement (unseeded ones excluded, since
    there is no score to include).
    """
    pool: dict[str, list[tuple[np.ndarray, str]]] = defaultdict(list)
    scores: list[float] = []
    for card, (character, series) in truth.items():
        if card not in card_lines:
            continue
        work, lines = card_lines[card]
        for line_idx, text in _training_texts(character, series, len(lines)):
            if line_idx >= len(lines) or not text:
                continue
            norm = normalise_line(work, lines[line_idx])
            if norm.size == 0:
                continue
            spans = forced_align(norm, text, templates)
            keep = []
            for c, x0, x1 in spans:
                tpl = templates.get(c)
                if tpl is None:
                    keep.append((c, x0, x1))
                    continue
                patch = norm[:, x0:x1]
                cmp_patch = patch if patch.shape[1] == tpl.shape[1] else cv2.resize(
                    patch, (tpl.shape[1], tpl.shape[0]), interpolation=cv2.INTER_AREA)
                s = match_score(cmp_patch, tpl)
                scores.append(s)
                if s >= HARVEST_MIN_SCORE:
                    keep.append((c, x0, x1))
            for c, bitmaps in harvest(norm, keep).items():
                for b in bitmaps:
                    pool[c].append((b, card))
    mean_score = float(np.mean(scores)) if scores else 0.0
    return dict(pool), mean_score, len(scores)


def _bitmaps_only(pool: dict[str, list[tuple[np.ndarray, str]]]) -> dict[str, list[np.ndarray]]:
    return {c: [b for b, _card in samples] for c, samples in pool.items()}


def run_em(card_lines: dict, truth: dict):
    seed, used = seed_pool(card_lines, truth)
    templates = mean_templates(seed)
    print(f"seed: {used} cards, {sum(len(v) for v in seed.values())} glyphs, "
          f"{len(templates)} classes")

    pool: dict[str, list[tuple[np.ndarray, str]]] = {}
    prev_score = None
    print(f"{'iter':>4}  {'mean_score':>10}  {'n_scored':>8}  {'classes':>7}  "
          f"{'glyphs':>6}  {'median/class':>12}")
    for it in range(1, MAX_ITERS + 1):
        pool, mean_score, n_scored = align_and_harvest(card_lines, truth, templates)
        counts = sorted(len(v) for v in pool.values())
        n_glyphs = sum(counts)
        median_samples = counts[len(counts) // 2] if counts else 0
        print(f"{it:>4}  {mean_score:>10.4f}  {n_scored:>8}  {len(pool):>7}  "
              f"{n_glyphs:>6}  {median_samples:>12}")
        templates = mean_templates(_bitmaps_only(pool))
        if prev_score is not None and mean_score - prev_score < MIN_IMPROVEMENT:
            print(f"stopping: gain {mean_score - prev_score:+.4f} < {MIN_IMPROVEMENT}")
            break
        prev_score = mean_score
    return pool


def save_atlas(pool: dict[str, list[tuple[np.ndarray, str]]], path: Path):
    tpl, lab, src = [], [], []
    for c, samples in pool.items():
        for bitmap, card in samples:
            resized = bitmap if bitmap.shape == (GLYPH_H, GLYPH_W) else cv2.resize(
                bitmap, (GLYPH_W, GLYPH_H), interpolation=cv2.INTER_AREA)
            tpl.append(resized.astype(np.uint8))
            lab.append(c)
            src.append(card)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, templates=np.stack(tpl),
                        labels=np.array(lab), card=np.array(src))
    print(f"wrote {path}: {len(tpl)} glyphs, {len(set(lab))} classes")

    thin = sorted(c for c in set(lab) if lab.count(c) < 3)
    print(f"thin classes (<3 samples): {thin}")


def main():
    bands, truth = _load_corpus()
    card_lines = _precompute_lines(bands, truth)
    pool = run_em(card_lines, truth)
    save_atlas(pool, OUT)


if __name__ == "__main__":
    main()
