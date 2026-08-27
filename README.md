# gacha_vision

Reads a screenshot of an anime card spawn and says which card is worth taking.

Given an image of one to four cards, it reports each card's **print number**,
**frame** and **character**, then applies a configurable policy:

* lower print number is better — `#14` beats `#852`
* `E` (no print number) is the bottom, below *any* numbered card
* two cards both under `#20` → take both, spend the extra pick
* a watchlist character can rescue an otherwise weak card

Every decision explains itself:

```
slot 1: BULMA (#14, normal)   badge=100%  sat=149.5
slot 2: YAEKA (#852, normal)  badge=100%  sat=106.3

CLAIM slot 1
  - slot 1: BULMA (#14, normal) -> 53.4
  - claimed: print #14 <= 20
  - passed on slot 2 YAEKA (#852, normal) (29.8)
```

**Scope:** image in, decision out. No game integration, no network access, no
account handling — by design. It analyses screenshots you already have.

## Accuracy

Measured on **182 cards from 91 real screenshots**, every badge labelled by
hand. The labels ship with the repo and the test suite replays them, so these
are a test result rather than a claim.

| metric | result |
|---|---|
| **spawns worth claiming, actually claimed** | **3/3**, no false claims |
| decision (action + slots) vs the labels | 91/91 = **100%** |
| badge digits, exact value | 181/182 = **99%** |
| frame (`E` vs numbered) | 182/182 = **100%** |
| costly errors — an `E` given a number, or a print lost as `E` | **0** |
| screenshots segmented correctly | 91/91 |

**Read the first row, not the second.** 88 of these 91 spawns should be
skipped, so a bot that skips everything scores 97% on decision accuracy. The
row that carries information is how many of the three spawns holding a good
card were claimed. Three spawns is a thin basis — treat it as a floor that
must not drop, not proof the ranker is finished.

## Install

```bash
pip install -r requirements.txt
```

Tesseract is only needed for the fallback paths (character names, and badges
in a font the digit atlas does not know):

```bash
sudo apt install tesseract-ocr                    # Linux
winget install UB-Mannheim.TesseractOCR           # Windows
```

Then confirm the machine is actually ready:

```bash
python -m gacha_vision doctor
```

`doctor` checks Python, every dependency and the tesseract binary, then runs
an end-to-end self-test on a card it draws itself — so a pass means the whole
chain works, not just that the imports resolved. On Windows the binary is
found automatically in the usual install locations; `TESSERACT_CMD` overrides
that.

## Use

```bash
# score a spawn screenshot
python -m gacha_vision analyze shot.png --expected 2

# with a watchlist of characters you care about
python -m gacha_vision analyze shot.png --watchlist gacha_vision/data/watchlist.example.json

# machine-readable
python -m gacha_vision analyze shot.png --json

# score a whole folder to a CSV
python -m gacha_vision batch shots/ --out report.csv

# render synthetic spawns and score them (no screenshots needed)
python -m gacha_vision demo
```

As a library:

```python
from gacha_vision import analyze_spawn, Policy

cards, decision = analyze_spawn("shot.png", policy=Policy(), expected=2)
print(decision.action, decision.slots)
```

## How it works

| stage | file | approach |
|---|---|---|
| find cards | `segment.py` | gutter projection, falling back to contours then equal columns |
| read the badge | `digits.py` | threshold search scored on badge shape → glyph atlas, 1-NN |
| fallback badge read | `ocr.py` | tesseract, for fonts the atlas does not know |
| frame | `frame.py` | border saturation decides `NORMAL` vs `E`; the badge cross-checks |
| decide | `rank.py` | weighted score per card, then the policy rules |

Two findings shaped this design, both of which reversed an assumption the code
started with. They are worth reading before changing either stage.

### The border decides the frame, not the badge

`E` is *by definition* the frame with no print number, which makes reading the
badge look like the exact answer and the border only an approximation. Real
cards say the opposite: over 182 of them the border called the frame right
**182 times and the badge 160**, and in all 22 disagreements the border was
right.

The asymmetry is in the difficulty, not the definition. Telling a gold ring
from a pale one is a measurement over thousands of border pixels; recognising
the badge glyph is a few strokes beside a hook icon shaped like a `1`, at a
size where OCR drops digits. So the border settles whether a card carries a
number, and the badge is left the job it can do. Getting this backwards cost
19 cards: 16 `E`s promoted to print numbers and 3 real prints discarded.

### The digits were never the problem — the crop was

Tesseract read 38 of 81 numbered badges. The glyphs are crisp white numerals
in one fixed font, so that number is not about recognition. It is that
producing a clean crop is the hard part: the badge plate is semi-transparent
over the card artwork, so its brightness moves card to card, and any single
global threshold either fuses neighbouring digits into one blob or dissolves
the thin ones. `1550` came back as `550` because the leading `1` fused with
the `5` and the fused blob was then discarded for being too wide.

So `digits.py` does not threshold once. It searches, and scores each result
against what a badge must be:

> one to four glyphs, equal height, on a common baseline, evenly spaced

which is a strong enough shape that the right cut point identifies itself. A
blob still holding two glyphs is split — no digit in this font is wider than
about 0.72 of its height, so anything wider is two — and each glyph is matched
against an atlas of examples taken from labelled real cards. Held out card by
card, it classifies **every glyph in the corpus correctly**.

It is also roughly a hundred times faster than shelling out to tesseract per
candidate crop, and on a confident read tesseract is not consulted at all.

## Calibrating against your own screenshots

The thresholds here were fitted to one player's spawns. There is a loop for
correcting them, and it never requires moving your screenshots anywhere.

```bash
# 1. crop every card and build a labelling page
python -m gacha_vision extract shots/ --out crops/

# 2. open crops/sheet.html, fix whatever it got wrong, Download labels.csv
#    (every field is prefilled with what the pipeline read, so most cards
#     need no touching — you are correcting, not transcribing)

# 3. fit thresholds to your labels
python -m gacha_vision fit labels.csv
```

`fit` prints a ready-to-paste `E_SATURATION` value, the accuracy of each cut
point, and a report separating the two errors worth caring about: a numbered
card read as `E`, and an `E` read as a number.

**Write `E` in the print column for an `E` card, not a blank.** `fit` skips
rows whose `true_print` is empty, so blanks silently drop every `E` from the
badge report and make the costly-error count read as zero when it is not. This
is how an earlier version of this README came to claim zero costly errors
while there were 19.

Both `extract` and `batch` write a CSV carrying every measurement the
classifier used, so the CSV alone is enough to tune thresholds — the images
can stay on your machine.

## Tuning

`config.py` holds every threshold worth arguing about — component weights, the
print-score curve, the take-both cut-off, the claim floor. Override them in
JSON and pass `--policy`:

```json
{ "take_both_max_print": 15, "min_claim_score": 50.0 }
```

The watchlist is a flat `{name: fame}` map; series names lift every character
in that show. Fame ≥ `must_claim_fame` (default 90) forces a claim.

Observed real prints run to 1600–2200, so the "two cards under #20" rule
almost never fires and most spawns are correctly skipped. In principle the
watchlist is what decides the rest — but see below.

## Known gaps

* **The name reader does not work.** It is not weak, it is broken: across 59
  hand-checked cards, not one character name was read correctly and not one
  was even 80% similar. `Yoshiko Tsushima` comes back as `occ`. Scale
  normalisation, brightness masking, per-line segmentation and confidence-based
  selection were each built and measured; none moved it, and the same code
  reads a rendered fixture perfectly at every size — so the difficulty is the
  game's display font, not the pipeline around it.

  This matters more than it looks, because the watchlist is the only consumer
  and the watchlist is what should decide most spawns. Built from every
  character *and* series in the sample at must-claim fame, it matched **1 card
  in 59**. Until the name reader works, `w_fame` — 35% of the score — is
  pinned at its default on essentially every card. Fixing it means training
  tesseract on the game font, matching artwork against a reference library
  instead of reading text, or dropping names and redistributing the weight.

* **`always_claim_unknown_frame` never fires.** `guess_frame` only ever
  returns `E` or `NORMAL`, so nothing in the pipeline produces the `OTHER`
  label the rule keys on. The one uncatalogued frame in the corpus is now read
  correctly by the digit atlas, so this no longer costs a card — but the
  safety net is still dead code.

* **`1140` is the one badge still misread.** Two `1`s fuse into a blob no
  wider than a normal `0`, so nothing in the shape gives the split away. A
  valley-detection rule was tried and reverted: it did not fix this card and
  it over-split `2511`.

* **`test_decision_accuracy_over_a_scenario_grid` fails.** A synthetic spawn
  of `#13` and `#20` claims one card where the grid expects both: the badge
  reads `13` as a single digit, which the lone-digit rule demotes below the
  trust floor by design. Either the grid's expectation is stale or that rule
  costs more than intended.

* **Three- and four-card spawns** are handled and tested, but no real
  four-card spawn has been seen, so those widths are only synthetic-verified.
  `max_claims` caps picks at 2, which is the game's own limit.

## Testing

```bash
python -m pytest gacha_vision/tests/ -q
```

134 tests, 133 passing (see Known gaps). The suite is deliberately not all
synthetic — an earlier version was, which is why the failures above went
unnoticed:

* `test_real_corpus.py` replays the 182 hand-labelled real cards and holds the
  floors on frame accuracy, costly errors, and claim recall.
* `test_badge_digits.py` runs against **real pixels** — `badge_windows.npz`
  holds the actual badge corner of all 81 numbered cards — and validates the
  glyph atlas leave-one-card-out, so it measures generalisation rather than
  memorisation.
* `test_rank.py` covers the policy in isolation, fast and image-free.
* `test_pipeline.py` renders spawns and asserts on the resulting decision.
