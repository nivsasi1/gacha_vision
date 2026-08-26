# gacha_vision

Reads a screenshot of an anime card spawn and says which card is worth taking.

Given an image of two or more cards, it reports each card's **print number**,
**frame rarity** and **character**, then applies a configurable policy:

* lower print number is better (`#14` beats `#852`)
* `E` (no print number) is the bottom — below *any* numbered card
* two cards both under `#20` → take both, spend the extra pick
* a better frame wins when prints are comparable
* a watchlist character can rescue an otherwise weak card

**Scope:** image in, decision out. There is no game integration, no network
access and no account handling in this package, by design. It analyses
screenshots you already have.

## Install

```bash
pip install -r gacha_vision/requirements.txt
sudo apt install tesseract-ocr          # the OCR engine itself
```

## Use

```bash
# score a spawn screenshot
python -m gacha_vision analyze shot.png --expected 2

# with a watchlist of characters you care about
python -m gacha_vision analyze shot.png --watchlist gacha_vision/data/watchlist.example.json

# machine-readable
python -m gacha_vision analyze shot.png --json

# render synthetic spawns and score them (no screenshots needed)
python -m gacha_vision demo

# dump frame features so the rarity thresholds can be tuned
python -m gacha_vision calibrate shot.png --expected 2
```

Typical output:

```
slot 1: BULMA (#14, rare)      ocr=99%   rarity_idx=0.695
slot 2: card 2 (#852, common)  ocr=100%  rarity_idx=0.175

CLAIM slot 1
  - slot 1: BULMA (#14, rare) -> 64.8
  - claimed: print #14 <= 20
  - passed on slot 2 card 2 (#852, common) (28.3)
```

Every decision explains itself. That is deliberate: the thresholds are
guesses until they are calibrated against real spawns, and you cannot
calibrate what you cannot see.

As a library:

```python
from gacha_vision import analyze_spawn, Policy

cards, decision = analyze_spawn("shot.png", policy=Policy(), expected=2)
print(decision.action, decision.slots)
```

## Calibrating against real screenshots

The thresholds shipped here were fitted to synthetic cards. Real frames and a
real game font will land somewhere else, so there is a loop for correcting
them — and it never requires moving your screenshots anywhere.

```bash
# 1. crop every card and build a labelling page
python -m gacha_vision extract shots/ --out crops/

# 2. open crops/sheet.html, fix whatever it got wrong, Download labels.csv
#    (every field is prefilled with what the pipeline read, so most cards
#     need no touching — you are correcting, not transcribing)

# 3. fit thresholds to your labels
python -m gacha_vision fit labels.csv
```

`fit` prints a ready-to-paste `THRESHOLDS` block, the accuracy of each cut
point, and an OCR report that separates the two errors worth caring about:
a numbered card read as `E`, and an `E` read as a number.

To score a folder without labelling anything:

```bash
python -m gacha_vision batch shots/ --out report.csv
```

Both commands write a CSV carrying every measurement the classifier used, so
the CSV alone is enough to tune thresholds — the images can stay on your
machine.

## Tuning

`config.py` holds every threshold worth arguing about — component weights,
the print-score curve, the take-both cut-off, the claim floor. Override them
in JSON and pass `--policy`:

```json
{ "take_both_max_print": 15, "min_claim_score": 50.0 }
```

The watchlist is a flat `{name: fame}` map; series names lift every character
in that show. Fame ≥ `must_claim_fame` (default 90) forces a claim.

**The watchlist matters more than it looks.** Observed real prints run to
1600–2200, so the "two cards under #20" rule almost never fires and most
spawns are correctly skipped. In practice the characters you care about are
the signal that decides most spawns — an empty watchlist means skipping
nearly everything.

## How it works

| stage | file | approach |
|---|---|---|
| find cards | `segment.py` | gutter projection, falling back to contours then equal columns |
| read the badge | `ocr.py` | component analysis → tight glyph crops → Tesseract, confidence-weighted vote |
| frame rarity | `frame.py` | hue entropy + perplexity + saturation of the border ring |
| decide | `rank.py` | weighted score per card, then the policy rules |

Five findings shaped this code, each caught by measurement rather than
assumption:

* **The badge plate must be separated from its glyphs.** OCR-ing the plate
  box feeds Tesseract the border strokes, which is how `1` becomes `2`.
* **Polarity must be decided, not averaged.** Handed an inverted crop,
  Tesseract read a `1` as `4` at confidence 69 while the correct `1` scored
  12 — so averaging both polarities let the wrong answer win. Ink is the
  minority class, so crops are normalised to black-on-white and read once.
* **Counting hue buckets is anti-correlated with rarity.** A true rainbow
  spreads ~5.6% into each of 18 buckets and clears a 6% cut-off *less* often
  than a coarse 5-hue frame, which ranked `rare` above `holo`. Perplexity
  (`exp(entropy)`) is monotonic and fixed it.
* **Cards are found by their gutters, not their outlines.** A common frame is
  grey and low-contrast, so Canny returns fragments of the artwork and no
  closed card boundary — an all-common spawn collapsed into a single box.
  The flat background *between* cards is unmistakable, so splitting on
  low-variance columns finds every card instead.
* **A fancy frame is not a rare card — in this game it is the opposite.**
  The scorer began by assuming ornate frames meant rarity. Real spawns say
  otherwise: across every observed spawn, each `E` card wore the ornate
  gold/chain frame with rainbow corners while each *numbered* card wore a
  plain thin border. Worse, the rarity index reads those ornate frames as
  `holo`, which used to force an automatic claim — so the ranker picked the
  junk half of every spawn. Frames no longer force a claim, and cannot lift
  a card that has no print number (`frame_lifts_unnumbered`). Watchlist fame
  still rescues an `E` card; only decoration is capped.
* **Parallelism is opt-in for a reason.** `fork` deadlocks (OpenCV and
  Tesseract have already started threads, and the children sit at 0% CPU),
  while `spawn` re-imports the main module — which under `python -m` is
  `__main__.py`, re-run through `runpy` where its relative import cannot
  resolve. Serial is ~150 ms/card, so the safe default costs little;
  `--workers N` opts in and falls back to serial if the pool will not start.

## Accuracy

On synthetic spawns (`python -m gacha_vision demo`):

| metric | result |
|---|---|
| badge OCR, 20 values × 4 frame tiers | 76/80 = **95%** |
| frame tier round-trip | 16/16 = **100%** |
| card segmentation, 40-spawn corpus | 93/93 cards, **100%** |
| test suite | **81 tests pass** |
| throughput | **~150 ms/card** (40 spawns / 93 cards in 21 s) |

On a 40-spawn corpus scored end to end, `fit` recovered frame thresholds at
98% accuracy and the badge reader hit 92% — with **zero costly errors**: no
numbered card read as `E`, and no `E` read as a number.

The four badge misses are all an isolated `#1` read as `#4`. Worth knowing:
`12`, `14`, `100` and `1584` all read correctly, so `1` only fails alone —
and because both `1` and `4` are below `must_claim_print`, the *decision* is
unchanged. That distinction is the point: **decision accuracy is the metric
that matters**, and a misread that flips `CLAIM` to `SKIP` is the only kind
that costs anything.

Mean confidence is 0.97 on correct reads versus 0.64 on wrong ones, so the
confidence figure is a usable review signal. Anything under 0.55 is flagged
in the CLI output.

These numbers are on synthetic cards drawn by `synth.py` — a real game font
and real artwork will differ. Treat them as a regression baseline, not a
promise, and run the calibration loop above on real screenshots before
trusting the frame thresholds.

## Testing

```bash
python -m pytest gacha_vision/tests/ -q
```

`test_rank.py` covers the policy in isolation (fast, no image work).
`test_pipeline.py` renders spawns and asserts on the resulting decision.
