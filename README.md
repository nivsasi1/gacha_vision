# gacha_vision

Reads a screenshot of an anime card spawn and says which card is worth taking.

Given an image of two or more cards, it reports each card's **print number**,
**frame** and **character**, then applies a configurable policy:

* lower print number is better (`#14` beats `#852`)
* `E` (no print number) is the bottom — below *any* numbered card
* two cards both under `#20` → take both, spend the extra pick
* a watchlist character can rescue an otherwise weak card

**Frames.** The game has two, and both are commons: `NORMAL` carries the
print number (the "rating"), `E` carries none. That definition makes the badge
look authoritative — but on 182 hand-labelled real cards the border called the
frame right 182 times and the badge 160, and the border won all 22
disagreements. So the border decides whether a card carries a number and the
badge only supplies the digits. A card whose badge contradicts the border is
still flagged for review, since that is reliably where a misread lives.

**Scope:** image in, decision out. There is no game integration, no network
access and no account handling in this package, by design. It analyses
screenshots you already have.

## Install

```bash
pip install -r gacha_vision/requirements.txt

# the OCR engine itself
sudo apt install tesseract-ocr                    # Linux
winget install UB-Mannheim.TesseractOCR           # Windows

# then confirm the machine is actually ready:
python -m gacha_vision doctor
```

`doctor` checks Python, every dependency and the tesseract binary, then runs
an end-to-end self-test on a card it draws itself — so a pass means the whole
chain works, not just that the imports resolved. On Windows the binary is
found automatically in the usual install locations; `TESSERACT_CMD` overrides
that if yours lives somewhere else.

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

# dump border features so the E/NORMAL cut point can be tuned
python -m gacha_vision calibrate shot.png --expected 2
```

Typical output:

```
slot 1: BULMA (#14, normal)   ocr=100%  ornate=0.301
slot 2: YAEKA (#852, normal)  ocr=100%  ornate=0.302

CLAIM slot 1
  - slot 1: BULMA (#14, normal) -> 53.4
  - claimed: print #14 <= 20
  - passed on slot 2 YAEKA (#852, normal) (29.8)
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

`fit` prints a ready-to-paste `E_SATURATION` value, the accuracy of each cut
point, and an OCR report that separates the two errors worth caring about:
a numbered card read as `E`, and an `E` read as a number.

The sheet also shows the character and series names the text OCR produced,
so you can see at a glance what it is getting. Those two boxes are optional
— fill one in only for cards you want counted, and `fit` will report name
accuracy over exactly those.

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
| fallback badge read | `ocr.py` | tesseract, for fonts the atlas does not know |
| read the badge | `digits.py` | threshold search scored on badge shape -> glyph atlas, 1-NN |
| frame | `frame.py` | border saturation decides `NORMAL` vs `E`; badge cross-checks |
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
  The scorer began by assuming ornate frames meant rarity, and invented a
  `common/uncommon/rare/holo` ladder to read off the border's colour. Real
  spawns say otherwise: each `E` card wore the ornate gold/chain frame with
  rainbow corners while each *numbered* card wore a plain thin border.
  Worse, the ornate frames measured as `holo`, which used to force an
  automatic claim — so the ranker picked the junk half of every spawn.
  The invented ladder is gone, replaced by the game's own two names, and
  the badge now decides which frame a card wears.
* **When a label is definitional, do not infer it from pixels.** `E` *is*
  the frame with no print number, so reading the badge answers the frame
  question exactly, while a border classifier can only approximate it. The
  measurement is still taken — as a cross-check that catches OCR slips and
  surfaces frames not yet catalogued — but it no longer decides anything.
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
| badge OCR, 20 values × 4 border styles | 76/80 = **95%** |
| card segmentation, 40-spawn corpus | 93/93 cards, **100%** |
| test suite | **134 tests**, 133 passing (see Known gaps) |
| throughput | **~150 ms/card** (40 spawns / 93 cards in 21 s) |

On a 40-spawn corpus scored end to end, `fit` recovered frame thresholds at
98% accuracy and the badge reader hit 92%. It also reported zero costly errors
— but that figure came from `ocr_report`, which silently drops every card
whose `true_print` label is blank; see the real-spawn section below. These
synthetic numbers have not been re-measured since that was found, so treat the
costly-error count here as unknown rather than zero.

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

### On real spawns

182 cards from 91 real screenshots, every badge and 59 of the name blocks
labelled by hand. The labels are committed at
`gacha_vision/tests/data/real_labels.csv` and `test_real_corpus.py` replays
them, so these numbers are a test, not a memory.

| metric | result |
|---|---|
| frame (`E` vs numbered), border at `sat_mean` 152.65 | 182/182 = **100%** |
| **costly errors** — an `E` given a number, or a number lost as `E` | **0/182** |
| badge digits, exact value | 181/182 = **99%** |
| **decision** (action + slots) vs the labels | 91/91 spawns = **100%** |
| — the same corpus if you simply skip everything | 88/91 = **97%** |
| spawns worth claiming, actually claimed | 3/3 |
| character name, *correct* | 0/59 = **0%** |
| character name, produced any text at all | 49/59 = **83%** |
| character name, string similarity to the truth | **0.18** |
| series name, string similarity to the truth | **0.07** |

**Read the claim row, not the decision row.** 100% decision accuracy sounds
strong until you notice that 88 of these 91 spawns should be skipped, so doing
nothing scores 97%. The row that carries information is the next one: all
three spawns holding a good card are claimed, with no false claims. That is
still only three spawns — treat it as a floor that must not drop, not as proof
the ranker is finished.

The badge reader no longer goes through tesseract on a card it recognises.
It reads 181 of 182, against 38 of 81 numbered cards before — and the failure
was never the glyphs, which are crisp white numerals in one fixed font. It was
that producing a clean crop is the hard part: the plate is semi-transparent
over the artwork, so its brightness moves card to card, and one global
threshold either fuses neighbouring digits or dissolves thin ones. `digits.py`
searches thresholds instead and scores each result against what a badge must
be — one to four glyphs, equal height, common baseline, evenly spaced — then
matches each glyph against an atlas built from labelled real cards. Held out
card by card it reads every glyph in the corpus correctly, and it is around a
hundred times faster than shelling out to tesseract per candidate crop.

That fix also recovered the corpus's best card: `cards30.png` is print **#14**
on an uncatalogued frame whose badge tesseract read as empty text. It is now
read and claimed.

Two rules from the tesseract era still apply on the fallback path:

* **A lone digit is the hook icon, not a print.** Of 27 single-digit reads,
  *none* was a genuine print — 19 of them on `NORMAL` frames, where a number
  certainly exists, so this is the reader losing three digits of a four-digit
  print rather than the `E` frame leaking through. Two-, three- and
  four-digit reads land at 29%, 29% and 82% exact and go through the ordinary
  confidence gate; one digit is 0%, so it is demoted unconditionally.
  Tesseract's own certainty carries no information here — the most confident
  impostor read at 1.000, above every correct multi-digit read.
* **No print is longer than four digits.** All 14 five-digit candidates were
  a real print with the hook glued on as a leading `1` (`1695` → `11695`).
  One leading `1` comes off, and only when the tail could itself be a print.
  Selection then prefers the longest surviving candidate, since a short one
  means glyphs were dropped.

The cost of the lone-digit rule is that a genuine `#1`–`#9` would be flagged
for review rather than claimed. That has never been observed — the lowest
real print across 182 cards was two digits — while the opposite error fired
27 times. The comment in `ocr.py` marks the line to revisit if a real
single-digit print ever appears in a labelled batch.

One rule was measured and **rejected**: claiming the best unreadable card
whenever the spawn would be skipped anyway — "a pick you were not going to
spend costs nothing" — collapsed to 31%. Unreadable badges are common enough
that it claims on almost every spawn.

A second was rejected once and has since been **reinstated**, which is worth
recording because the first verdict was not wrong so much as measured against
the wrong thing. Letting the border decide whether a number exists was judged
to score "no better than the lone-digit rule alone" — true, on *decision*
accuracy, which cannot separate them: 88 of these 91 spawns should be skipped,
so the metric saturates at 97% before either rule is applied. Scored on what
the rules actually change, they are not alternatives at all:

* the lone-digit rule withholds *trust* from a bad read, leaving `print_no`
  set and `no_number` false — so an `E` card still comes out wearing a
  `NORMAL` frame and scoring 25.0 as "unreadable" rather than 6.0 as an `E`;
* the border gate fixes the *frame*, which is a different field.

Measured on the 182 cards: frame 160/182 → **182/182**, and the costly errors
— 16 `E`s promoted to numbers, 3 numbers written off as `E` — go **19 → 0**.
The badge is no longer authoritative on whether a number exists; it is still
the only thing that reads the digits. `test_real_corpus.py` holds both floors.


* **The border beats the badge, and it is not close.** Border 182/182 against
  badge 160/182, and the border was right in all 22 disagreements. Letting the
  badge decide the frame was costing 19 cards — 16 `E`s promoted to numbers
  and 3 numbers written off as `E`. Both classes are now zero.
* **"Zero costly errors" was a measurement artifact.** `fit`'s `ocr_report`
  skips any row whose `true_print` is blank, which is how the labelling sheet
  records an `E` card — so all 101 `E`s were dropped and `E_read_as_number`
  could not be anything but zero. Write `E` in the print column, not a blank,
  and the same file reports 10 of them. (It still misses the 6 read as
  unreadable rather than as a digit.)
* **Confidence does not separate right digits from wrong ones.** The oft-quoted
  0.72-against-0.28 is real but comes entirely from the 101 clean `E` reads
  averaging 0.81. On numbered cards alone it is 0.31 correct against 0.27
  wrong — no signal. Only 48% of badges clear the 0.55 trust floor at all, so
  half the corpus scores as a flat "unreadable" constant.

### The name reader does not work

It is not weak, it is broken, and the earlier "reads something on 82% of
cards" figure was the metric flattering itself — a smear of letters counts
as a read. Against eight cards whose names were read off a screenshot by
eye (`data/name_truth_8cards.csv`):

| true character | read | true series | read |
|---|---|---|---|
| Yoshiko Tsushima | `occ` | Love Live! Series | `Ox` |
| Ryukyu | `avatars` | My Hero Academia | `hahahah aly` |
| Chika Fujiwara | `IA 7` | Kaguya-Sama: Love is War? | — |
| Pastry Cookie | `BP VOVOVOVE` | Cookie Run | — |
| Seras Victoria | `Abarat` | HELLSING | — |
| Moran | — | NIKKE: Goddess of Victory | — |
| Hinageshi Usuzumi | `OOOH` | Momentary Lily | `toy` |
| Misono Arisuin | — | SERVAMP | — |

**Zero of sixteen** share meaningful letters with the truth. A wider check
agrees: over 59 hand-read name blocks, none matched and none was even 80%
similar. These are not
near-misses that better thresholding would sharpen — that is what it looks
like when Tesseract does not recognise the typeface at all. Scale
normalisation (the fix that worked for the badge), brightness masking for
white-on-artwork text, per-line segmentation and confidence-based candidate
selection were each built and measured against a fixture; none moved it, and
the fixture reads its own names perfectly at every size, which is itself the
tell: the difficulty is the game's display font, not the pipeline around it.

Nothing depends on these. The watchlist is the only consumer, and a name
that matches nothing scores as `fame_default`, so the fame term degrades to
a constant rather than to a wrong answer — but it *is* a constant: a
watchlist built from every character and series in the sample, all at
must-claim fame, matched 1 card in 59, so `w_fame` (35% of the score) is
pinned at its default on essentially every card. `read_name` still reports its
text, and now its confidence alongside it, so the sheet shows how little to
trust it.

Getting it working means one of:

* **Train Tesseract on the game font** — the 182 crops already collected are
  most of what a `tesstrain` run needs, plus hand-typed labels.
* **Match the artwork instead of the text** — build a reference library of
  card images and match crops against it. Robust to the font entirely, but
  needs the library.
* **Drop names and set `w_fame` to 0**, redistributing the weight to the
  print number. Honest, and costs nothing that currently works.

## Known gaps

* **`test_decision_accuracy_over_a_scenario_grid` fails.** A synthetic spawn of
  `#13` and `#20` now claims one card where the grid expects both: the badge
  reads `13` as a single digit, which the lone-digit rule demotes below the
  trust floor by design. Either the grid's expectation is stale or the rule is
  costing more than intended — it wants a decision, not a silent edit.

* **Name OCR** (above). Needs a font-trained model or artwork matching, not
  tuning; the tuning was tried.
* **Dropped glyphs.** 41 of the 182 badges came back without the true value
  anywhere among the candidates — usually a four-digit print read as two or
  three digits. Length-first selection picks the best of what is there; it
  cannot recover a digit Tesseract never emitted. This is the ceiling on the
  68%, and fixing it means a better crop or a better binarisation, not a
  better vote.
* **Three- and four-card spawns.** Segmentation, scoring and the flag range
  all handle 1–4 cards and are tested at 3 and 4, but no real four-card
  spawn has been seen, so the widths above 3 are only synthetic-verified.
  `max_claims` still caps picks at 2, which is the game's own limit.
* **The `OTHER` frame** has never appeared in a labelled batch, so its cut
  point is unfitted. It is handled by always claiming it, which is the safe
  behaviour precisely because it is uncalibrated.

## Testing

```bash
python -m pytest gacha_vision/tests/ -q
```

`test_rank.py` covers the policy in isolation (fast, no image work).
`test_pipeline.py` renders spawns and asserts on the resulting decision.
