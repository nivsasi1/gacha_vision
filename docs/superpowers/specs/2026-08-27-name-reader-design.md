# Name reader — design

**Date:** 2026-08-27
**Status:** abandoned

**Postmortem (2026-08-28):** Both options below were built and measured, not
just A. Option A (learned glyph atlas) shipped a segmentation-free redesign
after per-glyph cutting hit a hard ceiling (55-62% exact glyph-count match,
oracle-given the correct answer), then a free-running reader on top of that
came in at -0.611 leave-one-card-out character accuracy against this doc's
own 0.95 bar. Option B (font identification) found no font within 0.3 of its
0.90 bar. The root cause neither could have coded around: the source images
are the original files downloaded from Discord, and the name text renders at
7 pixels tall -- well under what Tesseract or template matching need, and
not fixable by iterating on the algorithm. See the README's "Name reader"
section for the full comparison against the badge reader (9px, 99% exact)
and what remains -- the labelled corpus and working line segmentation --
for a future attempt.

## Problem

`read_name` cannot read the game's font. Measured against 59 hand-read cards:
**0 correct, and not one read even 80% similar** to the truth. `Yoshiko
Tsushima` comes back as `occ`, `Pastry Cookie` as `BP VOVOVOVE`. These are not
near-misses that better thresholding would sharpen — that is what it looks
like when Tesseract does not recognise the typeface at all. Scale
normalisation, brightness masking, per-line segmentation and confidence-based
candidate selection were each built and measured against a fixture; none moved
the number, and the same code reads a rendered fixture perfectly at every
size.

It matters because the watchlist is the only consumer of names, and the
watchlist is what should decide most spawns. Built from every character *and*
series in the sample at must-claim fame, it matched **1 card in 59**. So
`w_fame` — 35% of the scoring weight — is pinned at its default on
essentially every card.

## Goal

Read the character and series names off a card, **open vocabulary**: the
correct text for any card, including names never seen before. Not merely
matching against a known watchlist.

### Success criteria

Measured leave-one-card-out over the labelled corpus:

| metric | floor |
|---|---|
| character-level accuracy (1 − CER) on the character name | ≥ 95% |
| exact match on the character name | ≥ 85% |
| exact match on the series name | reported, no floor in this iteration |

These are deliberately below the badge reader's 99%. Badges are 10 rigid
classes in a monospaced plate; names are ~70 proportional classes where
letters touch, case is significant (`I` vs `l`, `O` vs `0`), and there is no
dictionary to fall back on. Setting the bar honestly now is better than
missing an invented one later.

### Non-goals

* Matching against the watchlist. That already works once the text is right.
* Reading Japanese/CJK glyphs. Every name in the corpus is Latin script.
  Latin-1 accented forms are in scope — `Pokémon` appears in the corpus —
  and are treated as their own atlas classes, not folded to ASCII.
* Fixing `1140`, the `OTHER` frame rule, or the scenario-grid test.

## Approach

Two options were considered and one is the foundation for the other.

**A — learned glyph atlas.** Segment the name lines into glyphs, align them to
hand-read labels, classify by 1-NN. This is exactly what took the badge digits
from 47% to 99%, on this codebase, so the machinery is proven.

**B — identify the font, render a complete atlas.** Use A's labelled glyphs as
a fingerprint, match against candidate font files, and once identified render
the entire character set. Complete coverage, no further labelling, and it
generalises to any glyph.

**Decision: build A, then attempt B as an upgrade.** They share every stage
except the atlas itself, so A costs nothing extra and guarantees a working
result. B is attempted opportunistically.

Coverage arithmetic is what makes this safe. 182 cards × two fields is roughly
**5,000 glyphs** — enough to cover A–Z, a–z, 0–9 and common punctuation many
times over, with even `Q`/`Z`/`j` appearing often enough for 1-NN. Rare
accented forms are the likeliest coverage gap, and are the clearest thing
stage B would fix. So B is an upgrade for completeness, not a rescue from a
gap.

**C — retraining Tesseract (tesstrain)** was rejected: heaviest toolchain,
slowest iteration, and the evidence from the badge work is that the wins come
from crop quality rather than the engine.

## Architecture

New module `gacha_vision/names.py`, parallel to `digits.py`. `analyze.py`
calls it in place of `ocr.read_name`.

### Pipeline

1. **Locate the name block.** Bright ink in the lower third. Not a fixed crop:
   the block sits higher and larger on `E` frames than on numbered ones, so
   the region is found from the ink rather than assumed.
2. **Group ink into lines.** Cluster components by row centre. The line count
   is variable, *not two* — `I've Been Killing Slimes for 300 Years and Maxed
   Out My Level` wraps to three. The current code's two-line assumption is
   wrong and is part of why series reads worse than character.
3. **Split character from series.** The character line is the topmost and
   measurably brighter — at threshold 248 it survives alone while the series
   line dims out. Remaining lines join with single spaces.
4. **Segment glyphs.** Connected components, then split fused pairs. Word
   gaps are spacing outliers within a line, the same principle as the digit
   splitter but tuned for a proportional face.
5. **Classify.** 1-NN against the glyph atlas, cosine on contrast-normalised
   16×24 bitmaps — the representation already proven on digits.
6. **Assemble.** glyphs → words → lines → `(character, series, confidence)`.

Stages 1–4 are shared by A and B; only stage 5's atlas differs.

### Interfaces

```python
read_names(card_bgr) -> NameRead          # (character, series, confidence)
segment_name_lines(work_gray) -> list[Line]
atlas_samples() -> (templates, labels, cards)
```

`NameRead.confidence` is the mean cosine similarity to matched templates, the
same signal that separates a clean digit read from a smeared one. Below a
trust floor the reader returns empty strings rather than a guess.

## Data and labelling

Names are hand-read from montages, as the badge labels were — 59 already done,
roughly 19 more montage reads for the remainder. No labelling work for the
user.

Alignment self-validates: a card contributes training glyphs **only** when its
segmented glyph count matches its label string with spaces removed. Mismatches
are dropped, never guessed at. This is the rule that kept the digit atlas
clean.

Shipped artefacts:

* `gacha_vision/data/glyph_atlas.npz` — templates, labels, source card
* `gacha_vision/tests/data/name_bands.npz` — real pixels, the name block of
  every labelled card
* `gacha_vision/tests/data/name_truth.csv` — hand-read character and series

## Font identification (stage B)

1. Build the learned atlas from A.
2. For each candidate font file, render the same glyph set at matching size
   and stroke weight.
3. Score by mean best-match cosine across all learned glyphs.
4. If a font scores decisively above the field, render the complete character
   set from it and ship that as the atlas; otherwise keep A's.

Candidate fonts come from the local system and from reputable public font
repositories (Google Fonts and similar). Font files are **data consumed by a
renderer, never executed**.

Risks: the face may be proprietary or modified, and rasterisation differences
(antialiasing, the outline stroke the game draws around glyphs) can degrade
matching even when the font is correct. Both are tolerable because A stands
alone.

## Testing

Mirrors `test_badge_digits.py`, which is the pattern that caught the badge
failures:

* real-pixel fixtures, committed, not renders
* leave-one-card-out over whole cards, so no glyph is classified using its own
  card — this measures generalisation, not memorisation
* floors on character-level accuracy and exact-name match
* a wiring test proving `analyze_cards` actually routes through the new reader
* a regression test naming a specific card the old reader mangled

## Removal

The Tesseract name path (`read_name`, `read_name_scored`, `_ocr_text_block`)
is deleted, along with `Card.name_confidence`'s dependence on it. Keeping a
0/59 reader as a backstop produces garbage text carrying a plausible-looking
confidence, which is worse than an honest empty. The README's "name reader
does not work" section is rewritten to describe what replaced it.

`read_badge` and the rest of `ocr.py` stay: the badge fallback for unfamiliar
fonts is still live and still useful.

## Open questions

None blocking. The bot this corpus comes from was not identified, so font
identification proceeds by fingerprint rather than by name.
