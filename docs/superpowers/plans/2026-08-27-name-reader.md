# Name Reader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Read the character and series names off a card in the game's own font, open vocabulary, replacing a Tesseract path that scores 0 correct out of 59.

**Architecture:** A new `gacha_vision/names.py` mirroring `gacha_vision/digits.py`: locate the name block from its ink, group components into a variable number of lines, split glyphs (including fused pairs and word gaps), and classify each glyph by 1-NN against an atlas of contrast-normalised 16×24 bitmaps. The atlas is learned from hand-read labels first (stage A); stage B then tries to identify the underlying font and render a complete character set from it.

**Tech Stack:** Python 3.14, numpy, OpenCV, Pillow (font rendering for stage B), pytest. No new runtime dependencies — Pillow is already in `requirements.txt`.

## Global Constraints

- **Do not touch the badge or frame paths.** `digits.py`, `frame.py`, `resolve_frame`, and `read_badge` are at 99%/100% and are higher priority than names. No task may modify them.
- **Success bar, measured leave-one-card-out:** character-level accuracy (1 − CER) on the character name ≥ 95%; exact match on the character name ≥ 85%. Series exact match is reported with no floor.
- **Latin-1 accented forms are their own atlas classes** — `é` in `Pokémon` is not folded to `e`. No CJK.
- **Leave-one-card-out means whole cards are held out**, so no glyph is ever classified using another glyph from its own card.
- **Fixtures are real pixels**, committed, never renders.
- **A card contributes training glyphs only when its segmented glyph count matches its label with spaces removed.** Mismatches are dropped, never guessed.
- Corpus lives at `crops_v2/` (gitignored, 182 card PNGs, names `<image>__slot<n>.png`).
- Existing labels: `gacha_vision/tests/data/real_labels.csv` has `true_character`/`true_series` filled for 59 cards.

---

### Task 1: Name-band fixture and hand-read labels

**Files:**
- Create: `gacha_vision/tests/data/name_bands.npz`
- Create: `gacha_vision/tests/data/name_truth.csv`
- Create: `tools/build_name_fixture.py`

**Interfaces:**
- Produces: `name_bands.npz` keyed `"<image>#<slot>"` → grayscale `np.uint8` array of the card's lower region; `name_truth.csv` with columns `card,true_character,true_series`.

- [ ] **Step 1: Write the fixture builder**

```python
# tools/build_name_fixture.py
"""Cut the name band out of every corpus card and save it as a test fixture."""
import csv, sys
import cv2, numpy as np

CROPS = "crops_v2"
BAND_TOP = 0.55          # generous: the block sits higher on E frames
OUT_NPZ = "gacha_vision/tests/data/name_bands.npz"

def band(card_bgr):
    h = card_bgr.shape[0]
    return cv2.cvtColor(card_bgr[int(BAND_TOP * h):, :], cv2.COLOR_BGR2GRAY)

def main():
    rows = list(csv.DictReader(open("gacha_vision/tests/data/real_labels.csv", encoding="utf-8")))
    out = {}
    for r in rows:
        img = cv2.imread(f"{CROPS}/{r['image'][:-4]}__slot{r['slot']}.png")
        if img is None:
            sys.exit(f"missing crop for {r['image']} slot {r['slot']}")
        out[f"{r['image']}#{r['slot']}"] = band(img)
    np.savez_compressed(OUT_NPZ, **out)
    print(f"wrote {len(out)} name bands")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and check the size**

Run: `python tools/build_name_fixture.py && ls -la gacha_vision/tests/data/name_bands.npz`
Expected: `wrote 182 name bands`, file under 1.5 MB. If larger, raise `BAND_TOP` toward 0.60 and re-run.

- [ ] **Step 3: Hand-read the remaining names**

59 cards already have `true_character`/`true_series` in `real_labels.csv`. Read the other 123 from montages, the same way the badge labels were produced: tile the name band of 10 cards per sheet at 3× with an index caption, view the sheet, transcribe.

Write the montage builder to `tools/name_montage.py` (mirror `tools/build_name_fixture.py` structure), generating `sheet_000.png` … one per 10 cards, then transcribe each into `name_truth.csv`.

Transcribe exactly what is rendered: preserve case, punctuation, and accents (`Pokémon`, not `Pokemon`). Where a series wraps across lines, join with single spaces into one string.

- [ ] **Step 4: Write name_truth.csv covering all 182 cards**

Columns: `card,true_character,true_series` where `card` is `"<image>#<slot>"`, matching the npz keys exactly.

- [ ] **Step 5: Verify coverage and glyph inventory**

```python
import csv, collections
rows = list(csv.DictReader(open("gacha_vision/tests/data/name_truth.csv", encoding="utf-8")))
assert len(rows) == 182, len(rows)
glyphs = collections.Counter(
    c for r in rows for c in (r["true_character"] + r["true_series"]) if c != " ")
print(f"{sum(glyphs.values())} glyphs, {len(glyphs)} classes")
print("classes with <3 samples:", sorted(k for k, v in glyphs.items() if v < 3))
```

Run it. Expected: several thousand glyphs across 60–80 classes. Record the thin classes — they are the coverage gap stage B is meant to close.

- [ ] **Step 6: Commit**

```bash
git add tools/build_name_fixture.py tools/name_montage.py gacha_vision/tests/data/name_bands.npz gacha_vision/tests/data/name_truth.csv
git commit -m "Label every card's names, and fixture the bands to test against"
```

---

### Task 2: Line segmentation

**Files:**
- Create: `gacha_vision/names.py`
- Create: `gacha_vision/tests/test_names.py`

**Interfaces:**
- Consumes: `name_bands.npz`, `name_truth.csv` from Task 1.
- Produces: `prepare_band(band_gray) -> np.ndarray` (upscaled working image); `segment_lines(work_gray) -> list[list[tuple[int,int,int,int]]]` — one list of glyph-component boxes per text line, ordered top to bottom, each line's boxes ordered left to right.

- [ ] **Step 1: Write the failing test**

```python
# gacha_vision/tests/test_names.py
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
    return bands, truth


def test_finds_at_least_one_line_on_every_card(corpus):
    bands, truth = corpus
    empty = [c for c in truth if not segment_lines(prepare_band(bands[c]))]
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest gacha_vision/tests/test_names.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'gacha_vision.names'`

- [ ] **Step 3: Implement `prepare_band` and `segment_lines`**

```python
# gacha_vision/names.py
"""Read the character and series names off a card.

Same shape of problem as the badge, and the same answer. The text is crisp
white ink in a fixed font; what defeats a general OCR engine is that the ink
sits over arbitrary artwork at a brightness that moves card to card. So the
threshold is searched rather than assumed, glyphs are cut out by component,
and each one is matched against an atlas built from labelled real cards.

Unlike the badge, the line count is not fixed: the character name takes one
line and the series wraps over one to three more.
"""

from __future__ import annotations

import cv2
import numpy as np

SCALE = 4
THRESHOLDS = tuple(range(205, 253, 6))
MIN_GLYPH_AREA = 18


def prepare_band(band_gray: np.ndarray) -> np.ndarray:
    return cv2.resize(band_gray, None, fx=SCALE, fy=SCALE,
                      interpolation=cv2.INTER_CUBIC)


def _components(work: np.ndarray, thr: int) -> list[tuple[int, int, int, int]]:
    h, w = work.shape
    ink = (work >= thr).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[i][:5])
        if area < MIN_GLYPH_AREA or bh < 0.03 * h or bh > 0.30 * h:
            continue
        if bw > 0.40 * w:
            continue
        out.append((x, y, bw, bh))
    return out


def _group_rows(boxes, tol: float = 0.45):
    """Cluster boxes into text lines by vertical overlap."""
    lines: list[list[tuple[int, int, int, int]]] = []
    for b in sorted(boxes, key=lambda c: c[1] + c[3] / 2):
        cy, bh = b[1] + b[3] / 2, b[3]
        for ln in lines:
            ry = sum(o[1] + o[3] / 2 for o in ln) / len(ln)
            rh = sum(o[3] for o in ln) / len(ln)
            if abs(cy - ry) <= tol * max(rh, bh):
                ln.append(b)
                break
        else:
            lines.append([b])
    for ln in lines:
        ln.sort(key=lambda c: c[0])
    lines.sort(key=lambda ln: sum(o[1] for o in ln) / len(ln))
    return lines


def _line_score(lines) -> float:
    """Prefer a threshold giving many glyphs of consistent height per line."""
    if not lines:
        return -1e9
    total = sum(len(ln) for ln in lines)
    hs = np.array([b[3] for ln in lines for b in ln], dtype=float)
    if len(hs) < 2:
        return total
    return total - 30.0 * float(hs.std() / max(hs.mean(), 1e-6))


def segment_lines(work_gray: np.ndarray):
    best, best_score = [], -1e18
    for thr in THRESHOLDS:
        lines = _group_rows(_components(work_gray, thr))
        sc = _line_score(lines)
        if sc > best_score:
            best_score, best = sc, lines
    return best
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest gacha_vision/tests/test_names.py -q`
Expected: 2 passed. If `test_line_count_covers_the_wrapped_series_names` fails, widen `THRESHOLDS` and loosen the `_components` height band — do not loosen the assertion.

- [ ] **Step 5: Commit**

```bash
git add gacha_vision/names.py gacha_vision/tests/test_names.py
git commit -m "Segment the name block into however many lines it actually has"
```

---

### Task 3: Glyph and word segmentation within a line

**Files:**
- Modify: `gacha_vision/names.py`
- Modify: `gacha_vision/tests/test_names.py`

**Interfaces:**
- Consumes: `segment_lines` from Task 2.
- Produces: `split_line(work_gray, line) -> list[tuple[int,int,int,int] | None]` — glyph boxes left to right, with `None` marking a word gap.

- [ ] **Step 1: Write the failing test**

```python
def test_glyph_count_matches_the_label_on_most_cards(corpus):
    """The alignment gate: a card only trains the atlas when its segmented
    glyph count matches its label. This measures how many qualify."""
    from gacha_vision.names import split_line
    bands, truth = corpus
    ok = 0
    for card, (character, _series) in truth.items():
        work = prepare_band(bands[card])
        lines = segment_lines(work)
        if not lines:
            continue
        glyphs = [g for g in split_line(work, lines[0]) if g is not None]
        if len(glyphs) == len(character.replace(" ", "")):
            ok += 1
    assert ok >= 0.70 * len(truth), (
        f"only {ok}/{len(truth)} character lines segment to their label length")


def test_word_gaps_are_marked(corpus):
    """'Yoshiko Tsushima' is two words; the reader must see the space."""
    from gacha_vision.names import split_line
    bands, truth = corpus
    two_word = [c for c, (ch, _) in truth.items() if ch.count(" ") == 1]
    assert two_word, "corpus should contain two-word character names"
    found = 0
    for card in two_word:
        work = prepare_band(bands[card])
        lines = segment_lines(work)
        if lines and sum(1 for g in split_line(work, lines[0]) if g is None) == 1:
            found += 1
    assert found >= 0.70 * len(two_word), (
        f"only {found}/{len(two_word)} two-word names had exactly one gap")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest gacha_vision/tests/test_names.py -q`
Expected: FAIL, `ImportError: cannot import name 'split_line'`

- [ ] **Step 3: Implement `split_line`**

```python
# append to gacha_vision/names.py

# No glyph in this face is wider than about this fraction of its height; a
# fused pair runs well above it. Same principle as the badge digit splitter,
# but a proportional face needs a looser bound than a monospaced one.
MAX_GLYPH_ASPECT = 1.05


def _split_fused(work: np.ndarray, box, med_h: float):
    x, y, bw, bh = box
    if bh <= 0 or bw / bh <= MAX_GLYPH_ASPECT:
        return [box]
    k = max(2, int(round(bw / (0.60 * med_h))))
    sub = work[y:y + bh, x:x + bw]
    ink = (sub >= np.percentile(sub, 60)).astype(np.float64)
    columns = ink.sum(axis=0)
    cuts = []
    for j in range(1, k):
        target = j * bw / k
        lo = int(max(1, target - 0.30 * bw / k))
        hi = int(min(bw - 1, target + 0.30 * bw / k))
        if hi > lo:
            cuts.append(lo + int(np.argmin(columns[lo:hi])))
    bounds = [0] + sorted(set(cuts)) + [bw]
    return [(x + a, y, b - a, bh) for a, b in zip(bounds, bounds[1:]) if b - a >= 2]


def split_line(work_gray: np.ndarray, line):
    """Glyph boxes for one line, left to right, with None marking a space."""
    if not line:
        return []
    med_h = float(np.median([b[3] for b in line]))
    glyphs = []
    for box in sorted(line, key=lambda c: c[0]):
        glyphs.extend(_split_fused(work_gray, box, med_h))
    glyphs.sort(key=lambda c: c[0])
    if len(glyphs) < 2:
        return glyphs

    gaps = [glyphs[i + 1][0] - (glyphs[i][0] + glyphs[i][2])
            for i in range(len(glyphs) - 1)]
    # A word space is a clear outlier against the letter spacing of the same
    # line, so the cut point is derived per line rather than fixed.
    inter = float(np.median([g for g in gaps])) if gaps else 0.0
    cut = max(inter * 2.2, 0.22 * med_h)

    out = [glyphs[0]]
    for g, gap in zip(glyphs[1:], gaps):
        if gap >= cut:
            out.append(None)
        out.append(g)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest gacha_vision/tests/test_names.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add gacha_vision/names.py gacha_vision/tests/test_names.py
git commit -m "Cut a name line into glyphs and words"
```

---

### Task 4: Glyph atlas and classifier

**Files:**
- Create: `tools/build_glyph_atlas.py`
- Create: `gacha_vision/data/glyph_atlas.npz`
- Modify: `gacha_vision/names.py`
- Modify: `gacha_vision/tests/test_names.py`

**Interfaces:**
- Consumes: `split_line`, `segment_lines`, `prepare_band`.
- Produces: `glyph_vector(work, box) -> np.ndarray | None`; `atlas_samples() -> (templates uint8[N,24,16], labels str[N], cards str[N])`; `read_names_from_band(band_gray) -> NameRead`; `NameRead` dataclass with fields `character: str`, `series: str`, `confidence: float`.

- [ ] **Step 1: Write the atlas builder**

> Note: this script imports `GLYPH_W`/`GLYPH_H` from `names.py`, which Step 2
> adds. Write this file now but run it only after Step 2.

```python
# tools/build_glyph_atlas.py
"""Learn a glyph atlas from the hand-read names."""
import csv
import cv2, numpy as np
from gacha_vision.names import prepare_band, segment_lines, split_line, GLYPH_W, GLYPH_H

BANDS = "gacha_vision/tests/data/name_bands.npz"
TRUTH = "gacha_vision/tests/data/name_truth.csv"
OUT = "gacha_vision/data/glyph_atlas.npz"


def crop(work, box):
    x, y, w, h = box
    sub = work[y:y + h, x:x + w]
    if sub.size == 0:
        return None
    sub = cv2.normalize(sub, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.resize(sub, (GLYPH_W, GLYPH_H), interpolation=cv2.INTER_AREA).astype(np.uint8)


def _distribute(fields: list[str], n_lines: int) -> list[str]:
    """Map label fields onto rendered lines.

    The character name always occupies line 0. The series may wrap over the
    remaining lines, and the wrap points are decided by the game's renderer,
    not by us -- so a wrapped series cannot be aligned word-for-word to a
    single line. Return the character label for line 0 and, only when the
    series happens to fit on exactly one line, that series for line 1.
    Everything else yields "" and is skipped by the alignment gate, which
    costs some training data but never mislabels any.
    """
    out = [fields[0]] + [""] * (n_lines - 1)
    if len(fields) > 1 and n_lines == 2:
        out[1] = fields[1]
    return out


def main():
    bands = np.load(BANDS)
    truth = {r["card"]: (r["true_character"], r["true_series"])
             for r in csv.DictReader(open(TRUTH, encoding="utf-8"))}
    tpl, lab, src = [], [], []
    used = 0
    for card, (character, series) in truth.items():
        work = prepare_band(bands[card])
        lines = segment_lines(work)
        if not lines:
            continue
        # Line 0 is the character name; the remaining lines are the series,
        # wrapped. Both are training data -- taking only the character line
        # would throw away half the glyphs, and the series lines are where
        # most of the rarer letters live.
        wanted = [character] + ([series] if series else [])
        matched = False
        for line, label in zip(lines, _distribute(wanted, len(lines))):
            boxes = split_line(work, line)
            letters = [c for c in label if c != " "]
            glyphs = [g for g in boxes if g is not None]
            gaps = sum(1 for g in boxes if g is None)
            # The alignment gate: only train when the segmentation agrees
            # with the label exactly, on both glyph count and word count.
            if gaps != label.count(" ") or len(glyphs) != len(letters):
                continue
            for g, ch in zip(glyphs, letters):
                im = crop(work, g)
                if im is not None:
                    tpl.append(im); lab.append(ch); src.append(card)
            matched = True
        used += int(matched)
    np.savez_compressed(OUT, templates=np.stack(tpl),
                        labels=np.array(lab), card=np.array(src))
    print(f"{len(tpl)} glyphs from {used} cards, {len(set(lab))} classes")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add the classifier to `names.py`**

```python
# append to gacha_vision/names.py
import functools
from dataclasses import dataclass
from pathlib import Path

GLYPH_W, GLYPH_H = 16, 24
_ATLAS_PATH = Path(__file__).parent / "data" / "glyph_atlas.npz"
# Below this the match is a guess. Calibrated in Task 6 against the corpus.
MIN_TRUSTED_MATCH = 0.80


@dataclass
class NameRead:
    character: str = ""
    series: str = ""
    confidence: float = 0.0


def glyph_vector(work_gray: np.ndarray, box) -> np.ndarray | None:
    x, y, w, h = box
    sub = work_gray[y:y + h, x:x + w]
    if sub.size == 0:
        return None
    sub = cv2.normalize(sub, None, 0, 255, cv2.NORM_MINMAX)
    sub = cv2.resize(sub, (GLYPH_W, GLYPH_H), interpolation=cv2.INTER_AREA)
    v = sub.reshape(-1).astype(np.float32)
    v -= v.mean()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else None


def atlas_samples():
    with np.load(_ATLAS_PATH, allow_pickle=False) as z:
        return z["templates"], z["labels"], z["card"]


@functools.lru_cache(maxsize=1)
def _atlas():
    tpl, labels, card = atlas_samples()
    X = tpl.reshape(len(tpl), -1).astype(np.float32)
    X -= X.mean(axis=1, keepdims=True)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-6)
    return X, labels, card


def _read_line(work, line):
    X, labels, _ = _atlas()
    text, sims = [], []
    for box in split_line(work, line):
        if box is None:
            text.append(" ")
            continue
        v = glyph_vector(work, box)
        if v is None:
            continue
        s = X @ v
        i = int(np.argmax(s))
        text.append(str(labels[i]))
        sims.append(float(s[i]))
    return "".join(text).strip(), sims


def read_names_from_band(band_gray: np.ndarray) -> NameRead:
    work = prepare_band(band_gray)
    lines = segment_lines(work)
    if not lines:
        return NameRead()
    character, sims = _read_line(work, lines[0])
    parts, series_sims = [], []
    for ln in lines[1:]:
        txt, s = _read_line(work, ln)
        if txt:
            parts.append(txt)
            series_sims.extend(s)
    conf = float(np.mean(sims + series_sims)) if (sims or series_sims) else 0.0
    if conf < MIN_TRUSTED_MATCH:
        return NameRead(confidence=round(conf, 3))
    return NameRead(character, " ".join(parts), round(conf, 3))


def read_names(card_bgr: np.ndarray) -> NameRead:
    h = card_bgr.shape[0]
    band = cv2.cvtColor(card_bgr[int(0.55 * h):, :], cv2.COLOR_BGR2GRAY)
    return read_names_from_band(band)
```

- [ ] **Step 3: Write the failing accuracy test**

```python
def _cer(pred: str, want: str) -> float:
    """Levenshtein distance normalised by the reference length."""
    if not want:
        return 0.0 if not pred else 1.0
    prev = list(range(len(pred) + 1))
    for j, wc in enumerate(want, 1):
        cur = [j]
        for i, pc in enumerate(pred, 1):
            cur.append(min(prev[i] + 1, cur[i - 1] + 1, prev[i - 1] + (pc != wc)))
        prev = cur
    return prev[len(pred)] / len(want)


def test_character_names_read_accurately_on_unseen_cards(corpus):
    """Leave-one-card-out: the atlas never sees the card it is reading."""
    from gacha_vision.names import (atlas_samples, glyph_vector, split_line)
    bands, truth = corpus
    tpl, labels, cards = atlas_samples()
    X = tpl.reshape(len(tpl), -1).astype(np.float32)
    X -= X.mean(axis=1, keepdims=True)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-6)

    exact = 0; cers = []; n = 0
    for card, (want, _series) in truth.items():
        work = prepare_band(bands[card])
        lines = segment_lines(work)
        if not lines:
            continue
        keep = cards != card
        chars = []
        for box in split_line(work, lines[0]):
            if box is None:
                chars.append(" "); continue
            v = glyph_vector(work, box)
            if v is None:
                continue
            chars.append(str(labels[keep][int(np.argmax(X[keep] @ v))]))
        got = "".join(chars).strip()
        n += 1
        cers.append(_cer(got, want))
        exact += int(got == want)
    acc = 1.0 - float(np.mean(cers))
    assert acc >= 0.95, f"character-level accuracy {acc:.3f} over {n} cards"
    assert exact / n >= 0.85, f"exact match {exact}/{n}"
```

- [ ] **Step 4: Build the atlas and run**

Run:
```bash
python tools/build_glyph_atlas.py
python -m pytest gacha_vision/tests/test_names.py -q
```
Expected: atlas reports several thousand glyphs across 60+ classes; all tests pass. If accuracy misses the bar, the fix is in segmentation (Tasks 2–3), not in loosening the assertion — inspect failures by printing `got` vs `want` for the worst 20 CERs before changing anything.

- [ ] **Step 5: Commit**

```bash
git add tools/build_glyph_atlas.py gacha_vision/data/glyph_atlas.npz gacha_vision/names.py gacha_vision/tests/test_names.py
git commit -m "Classify name glyphs against an atlas learned from real cards"
```

---

### Task 5: Wire into the pipeline and delete the Tesseract name path

**Files:**
- Modify: `gacha_vision/analyze.py:14,26-36,77-90`
- Modify: `gacha_vision/ocr.py:457-511` (delete `read_name`, `read_name_scored`, `_ocr_text_block`)
- Modify: `gacha_vision/models.py:66-69`
- Modify: `gacha_vision/tests/test_names.py`

**Interfaces:**
- Consumes: `read_names(card_bgr) -> NameRead` from Task 4.
- Produces: `Card.character`, `Card.series`, `Card.name_confidence` populated from the atlas reader.

- [ ] **Step 1: Write the failing wiring test**

```python
def test_the_pipeline_reads_a_real_card_name():
    """End to end on a card the Tesseract reader turned into 'occ'."""
    import cv2
    from gacha_vision.analyze import analyze_cards

    img = cv2.imread(str(DATA / "card_print_1550.png"))
    assert img is not None
    card = analyze_cards(img, expected=1, read_names=True)[0]
    assert card.character == "Mugen"
    assert card.series == "Samurai Champloo"
    assert card.name_confidence > 0.8


def test_the_tesseract_name_path_is_gone():
    import gacha_vision.ocr as ocr
    for gone in ("read_name", "read_name_scored", "_ocr_text_block"):
        assert not hasattr(ocr, gone), f"{gone} should have been deleted"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest gacha_vision/tests/test_names.py -q -k "pipeline or tesseract"`
Expected: both FAIL — the first on the name being garbage, the second on the attributes still existing.

- [ ] **Step 3: Rewire `analyze.py`**

Replace the import on line 14 and the name block at lines 77-90:

```python
from .ocr import MIN_TRUSTED_CONFIDENCE, read_badge
from .names import read_names
```

```python
        character, series, name_conf = "", "", 0.0
        if read_names:
            got = read_names(crop)
            character, series, name_conf = got.character, got.series, got.confidence
```

Delete `split_name` (lines 26-36) — the reader now returns the two fields separately, so the "split one OCR blob in half" heuristic has nothing left to do.

- [ ] **Step 4: Delete the Tesseract name path**

Remove `read_name`, `read_name_scored` and `_ocr_text_block` from `ocr.py`, and `_NAME_CFG` on line 72 if nothing else references it (check with `grep -n _NAME_CFG gacha_vision/`). Update the `Card.name_confidence` comment in `models.py:66-68` to point at `names.read_names` rather than `ocr.read_name`.

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest gacha_vision/tests/ -q`
Expected: all pass except the known pre-existing `test_decision_accuracy_over_a_scenario_grid`. **If any badge, frame, or decision test changed status, stop and revert** — those paths are higher priority than names.

- [ ] **Step 6: Commit**

```bash
git add gacha_vision/analyze.py gacha_vision/ocr.py gacha_vision/models.py gacha_vision/tests/test_names.py
git commit -m "Route names through the atlas reader; delete the tesseract path"
```

---

### Task 6: Stage B — identify the font, render a complete atlas

**Files:**
- Create: `tools/identify_font.py`
- Modify: `gacha_vision/data/glyph_atlas.npz` (only if a font wins decisively)
- Modify: `gacha_vision/names.py` (`MIN_TRUSTED_MATCH` calibration)

**Interfaces:**
- Consumes: `atlas_samples()` from Task 4.
- Produces: a ranked report of candidate fonts by mean best-match cosine; optionally a re-rendered atlas with identical `.npz` keys.

- [ ] **Step 1: Write the fingerprint scorer**

```python
# tools/identify_font.py
"""Score candidate fonts against the learned glyph atlas."""
import glob, sys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from gacha_vision.names import atlas_samples, GLYPH_W, GLYPH_H

def render(path, ch, px=64):
    try:
        font = ImageFont.truetype(path, px)
    except Exception:
        return None
    img = Image.new("L", (px * 2, px * 2), 0)
    ImageDraw.Draw(img).text((px // 2, px // 2), ch, fill=255, font=font)
    a = np.array(img)
    ys, xs = np.where(a > 20)
    if len(xs) == 0:
        return None
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return cv2.resize(a, (GLYPH_W, GLYPH_H), interpolation=cv2.INTER_AREA)

def vec(a):
    v = a.reshape(-1).astype(np.float32); v -= v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else None

def main(font_globs):
    tpl, labels, _ = atlas_samples()
    by_class = {}
    for t, l in zip(tpl, labels):
        by_class.setdefault(str(l), []).append(vec(t))
    scores = []
    for pattern in font_globs:
        for path in glob.glob(pattern):
            sims = []
            for ch, vs in by_class.items():
                r = render(path, ch)
                if r is None:
                    continue
                rv = vec(r)
                if rv is None:
                    continue
                sims.append(max(float(rv @ v) for v in vs if v is not None))
            if len(sims) >= 20:
                scores.append((float(np.mean(sims)), len(sims), path))
    for s, n, p in sorted(scores, reverse=True)[:15]:
        print(f"{s:.4f}  ({n} glyphs)  {p}")

if __name__ == "__main__":
    main(sys.argv[1:] or ["C:/Windows/Fonts/*.ttf"])
```

- [ ] **Step 2: Score the local fonts**

Run: `python tools/identify_font.py "C:/Windows/Fonts/*.ttf" "C:/Windows/Fonts/*.otf"`
Expected: a ranked list. Record the top score and the gap to second place.

- [ ] **Step 3: Widen the search to downloaded fonts**

Download candidate families into `tools/fonts/` from reputable public repositories (the Google Fonts GitHub repository is the primary source). Font files are data consumed by the renderer and are never executed. Prioritise rounded geometric sans faces, which is what the card text appears to be.

Run: `python tools/identify_font.py "tools/fonts/**/*.ttf"`

- [ ] **Step 4: Decide**

A font wins only if its mean cosine is **≥ 0.90 and at least 0.05 clear of the runner-up**. If so, re-render the full character set — `A-Za-z0-9` plus `.,!?':;-&/()` and the Latin-1 accented forms found in Task 1 Step 5 — and rebuild `glyph_atlas.npz` from it, keeping the same keys so `names.py` needs no change.

If no font clears the bar, keep the learned atlas and record the top five candidates and their scores in the spec's Font identification section. **This is a legitimate outcome, not a failure** — stage A stands alone.

- [ ] **Step 5: Re-run the accuracy tests either way**

Run: `python -m pytest gacha_vision/tests/test_names.py -q`
Expected: all pass. If a rendered atlas scores *worse* than the learned one on the Task 4 test, keep the learned one and say so.

- [ ] **Step 6: Calibrate `MIN_TRUSTED_MATCH`**

Print the leave-one-card-out confidence distribution for correct versus incorrect whole-name reads. Set `MIN_TRUSTED_MATCH` below the 5th percentile of correct reads and above the median of incorrect ones, exactly as `digits.MIN_TRUSTED_MATCH` was set. Record both numbers in a comment.

- [ ] **Step 7: Commit**

```bash
git add tools/identify_font.py gacha_vision/data/glyph_atlas.npz gacha_vision/names.py
git commit -m "Fingerprint the card font against candidate faces"
```

---

### Task 7: Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-08-27-name-reader-design.md`

- [ ] **Step 1: Rewrite the README's name section**

Replace the "The name reader does not work" gap with measured results: character-level accuracy, exact-match rate, what the atlas covers, and whether the font was identified. Update the accuracy table and the "How it works" stage table to list `names.py`. Update the test count.

- [ ] **Step 2: Record the outcome in the spec**

Set `**Status:**` to `implemented`, and fill the Font identification section with what actually happened, including the candidate scores.

- [ ] **Step 3: Verify every number in the README is real**

For each figure quoted, point at the command that produces it. Any number that cannot be reproduced from a committed test or script is removed.

- [ ] **Step 4: Commit and push**

```bash
git add README.md docs/superpowers/specs/2026-08-27-name-reader-design.md
git commit -m "Document what the name reader actually does now"
git push origin main
```

---

## Self-Review

**Spec coverage:** Problem → Tasks 1–5. Open-vocabulary goal → Task 4 atlas + Task 6 full charset. Success criteria → Task 4 Step 3 assertions. Accented classes → Task 1 Step 3 (transcribe accents), Task 6 Step 4 (render them). Variable line count → Task 2. Character-vs-series split → Task 2 `segment_lines` ordering + Task 4 `read_names_from_band`. Glyph segmentation and word gaps → Task 3. Alignment gate → Task 4 builder. Shipped artefacts → Tasks 1 and 4. Font identification → Task 6. Testing (real pixels, LOO, wiring, regression) → Tasks 2–5. Removal → Task 5. README → Task 7. No gaps.

**Placeholder scan:** No TBD/TODO. Every code step carries runnable code; every run step carries a command and expected output.

**Type consistency:** `prepare_band` → `segment_lines` → `split_line` → `glyph_vector` chain uses consistent box tuples `(x, y, w, h)`; `None` as the word-gap marker is introduced in Task 3 and consumed in Task 4 `_read_line` and the Task 4 test. `GLYPH_W`/`GLYPH_H` are defined in Task 4 Step 2 and imported by Task 4 Step 1's builder and Task 6's scorer — note this ordering when executing: run Task 4 Step 2 before Step 1's script. `NameRead` fields match their use in Task 5. `atlas_samples()` returns the same triple in Tasks 4 and 6.
