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

## Tuning

`config.py` holds every threshold worth arguing about — component weights,
the print-score curve, the take-both cut-off, the claim floor. Override them
in JSON and pass `--policy`:

```json
{ "take_both_max_print": 15, "min_claim_score": 50.0 }
```

The watchlist is a flat `{name: fame}` map; series names lift every character
in that show. Fame ≥ `must_claim_fame` (default 90) forces a claim.

## How it works

| stage | file | approach |
|---|---|---|
| find cards | `segment.py` | contour detection for portrait rectangles, with an equal-column fallback |
| read the badge | `ocr.py` | component analysis → tight glyph crops → Tesseract, confidence-weighted vote |
| frame rarity | `frame.py` | hue entropy + perplexity + saturation of the border ring |
| decide | `rank.py` | weighted score per card, then the policy rules |

Three findings shaped this code, each caught by measurement rather than
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

## Accuracy

On synthetic spawns (`python -m gacha_vision demo`):

| metric | result |
|---|---|
| badge OCR, 20 values × 4 frame tiers | 76/80 = **95%** |
| frame tier round-trip | 16/16 = **100%** |
| end-to-end decisions | 61/61 tests pass |

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
promise. Drop real screenshots in and run `calibrate` before trusting the
frame thresholds.

## Testing

```bash
python -m pytest gacha_vision/tests/ -q
```

`test_rank.py` covers the policy in isolation (fast, no image work).
`test_pipeline.py` renders spawns and asserts on the resulting decision.
