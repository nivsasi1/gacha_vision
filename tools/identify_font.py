"""Identify the font the game renders card name text in.

Card names are white glyphs with a dark outline over arbitrary artwork.
`gacha_vision/names.py` already segments them without knowing what font
they are; on ~52 of the 182 corpus cards its glyph count matches the label
exactly, which hands us real, correctly-labelled letter images -- our
fingerprint material. This script renders the same characters in candidate
fonts and scores each by mean cosine similarity, using the same vector
representation `gacha_vision/digits.py` uses for its own atlas match: a
grayscale glyph patch, min-max stretched, resized to a fixed canvas,
flattened, mean-centred, unit-normalised.

Two independent glyph sources are scored against every candidate:

  * letters -- from the name-text corpus above. This is the metric that
    answers the actual question and drives the verdict.
  * digits  -- from `gacha_vision/data/digit_atlas.npz` (badge-plate
    numerals). The badge may or may not share the name text's font; this
    is corroborating evidence only, never the decision metric.

Fonts come from two places:

  * every .ttf/.otf already installed in C:/Windows/Fonts (brute-forced --
    cheap enough that curation isn't worth the risk of excluding the
    answer).
  * a curated set of "rounded geometric sans" families pulled from the
    Google Fonts GitHub repo into tools/fonts/ (gitignored). Re-fetch with
    --fetch-fonts; it skips files already on disk.

Usage:
    python tools/identify_font.py --fetch-fonts
    python tools/identify_font.py --compare 3 --out tools/fonts/results.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gacha_vision.names import prepare_band, segment_lines, split_line  # noqa: E402
from gacha_vision.digits import (  # noqa: E402
    GLYPH_W as DIGIT_GLYPH_W,
    GLYPH_H as DIGIT_GLYPH_H,
    atlas_samples,
)

DATA = ROOT / "gacha_vision" / "tests" / "data"
FONTS_DIR = ROOT / "tools" / "fonts"
COMPARE_DIR = FONTS_DIR / "compare"
WINDOWS_FONTS = Path("C:/Windows/Fonts")

# Normalised canvas for LETTER glyphs. Letters run wider relative to their
# height than the digit atlas's 16x24 (median real-corpus w/h is 0.77 vs
# digits' 0.67) -- a dedicated canvas avoids squeezing every letter through
# a box shaped for numerals. The DIGIT pathway instead reuses digits.py's
# own GLYPH_W/GLYPH_H exactly, since that atlas is already baked to that
# size and re-normalising it a second time would just add resample blur.
LETTER_GLYPH_W, LETTER_GLYPH_H = 24, 32

# Render size for scoring (one glyph at a time) and for the visual-check
# line renders. Large enough that hinting/AA choices wash out once resized
# down to the scoring canvas.
GLYPH_RENDER_PX = 160
LINE_RENDER_PX = 100

# The card face reads "fairly bold" (see task brief); for a variable font
# try these named instances, boldest first isn't required since all three
# are scored and the best wins. Static single-weight files (most of
# Windows' own fonts, plus Poppins/VarelaRound/MPLUSRounded1c/Sniglet/
# ConcertOne here) are just scored as they are.
WEIGHT_PRIORITY = ("Bold", "SemiBold", "ExtraBold")

# A font missing more than this fraction of the needed characters is
# judged unfairly by a partial score (e.g. a symbol/CJK-only font) and is
# dropped rather than ranked.
MIN_COVERAGE = 0.85

# The "rounded geometric sans, fairly bold, generous letter spacing" set
# named in the task brief, plus a few more of the same character. Fetched
# from https://github.com/google/fonts (raw.githubusercontent.com) --
# individual files, not a repo clone (the monorepo is >1GB).
GOOGLE_FONTS = [
    # (ofl family dir, filename on the repo, local filename)
    ("nunito", "Nunito%5Bwght%5D.ttf", "Nunito-Variable.ttf"),
    ("quicksand", "Quicksand%5Bwght%5D.ttf", "Quicksand-Variable.ttf"),
    ("baloo2", "Baloo2%5Bwght%5D.ttf", "Baloo2-Variable.ttf"),
    ("fredoka", "Fredoka%5Bwdth,wght%5D.ttf", "Fredoka-Variable.ttf"),
    ("comfortaa", "Comfortaa%5Bwght%5D.ttf", "Comfortaa-Variable.ttf"),
    ("poppins", "Poppins-Bold.ttf", "Poppins-Bold.ttf"),
    ("poppins", "Poppins-ExtraBold.ttf", "Poppins-ExtraBold.ttf"),
    ("rubik", "Rubik%5Bwght%5D.ttf", "Rubik-Variable.ttf"),
    ("varelaround", "VarelaRound-Regular.ttf", "VarelaRound-Regular.ttf"),
    ("mplusrounded1c", "MPLUSRounded1c-Bold.ttf", "MPLUSRounded1c-Bold.ttf"),
    ("mplusrounded1c", "MPLUSRounded1c-ExtraBold.ttf", "MPLUSRounded1c-ExtraBold.ttf"),
    ("mplusrounded1c", "MPLUSRounded1c-Black.ttf", "MPLUSRounded1c-Black.ttf"),
    ("sniglet", "Sniglet-ExtraBold.ttf", "Sniglet-ExtraBold.ttf"),
    ("concertone", "ConcertOne-Regular.ttf", "ConcertOne-Regular.ttf"),
    ("grandstander", "Grandstander%5Bwght%5D.ttf", "Grandstander-Variable.ttf"),
]

# Hand-picked, visually clean lines used for the side-by-side visual check
# (see the task's decision criteria -- cosine alone can rank a wrong font
# highly). (card key, line index into segment_lines()'s output).
SAMPLE_LINES = [
    ("44.png#1", 0),  # "Cielomort" -- distinctive C/i/e/l/o/m/o/r/t
    ("44.png#1", 1),  # "Fragaria Memories" -- several a's and a g, good for terminal detail
    ("29.png#2", 0),  # "Ryukyu"
    ("29.png#2", 1),  # "My Hero Academia"
]


# --------------------------------------------------------------------------
# glyph vector representation -- mirrors gacha_vision.digits._glyph_vector


def _vectorize(patch: np.ndarray, w: int, h: int) -> np.ndarray | None:
    if patch is None or patch.size == 0:
        return None
    sub = cv2.normalize(patch, None, 0, 255, cv2.NORM_MINMAX)
    sub = cv2.resize(sub, (w, h), interpolation=cv2.INTER_AREA)
    v = sub.reshape(-1).astype(np.float32)
    v -= v.mean()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else None


def letter_vector(patch: np.ndarray) -> np.ndarray | None:
    return _vectorize(patch, LETTER_GLYPH_W, LETTER_GLYPH_H)


def digit_vector(patch: np.ndarray) -> np.ndarray | None:
    return _vectorize(patch, DIGIT_GLYPH_W, DIGIT_GLYPH_H)


# --------------------------------------------------------------------------
# real, labelled glyphs


def load_real_letter_glyphs() -> list[tuple[str, np.ndarray]]:
    """Real letter crops from cards where split_line()'s glyph count
    matches the character name's letter count exactly (names.py's own
    alignment gate -- ~52/182 cards)."""
    bands = np.load(DATA / "name_bands.npz")
    with (DATA / "name_truth.csv").open(encoding="utf-8") as f:
        truth = {r["card"]: r["true_character"] for r in csv.DictReader(f)}
    out = []
    for card, character in truth.items():
        if not character:
            continue
        work = prepare_band(bands[card])
        lines = segment_lines(work)
        if not lines:
            continue
        boxes = [g for g in split_line(work, lines[0]) if g is not None]
        letters = character.replace(" ", "")
        if len(boxes) != len(letters):
            continue
        for ch, (x, y, w, h) in zip(letters, boxes):
            out.append((ch, work[y:y + h, x:x + w]))
    return out


# A card's glyph *count* matching its label length (the gate above) proves
# the line was split into the right number of pieces, not that each piece
# landed on the right letter -- one fused pair plus one over-split glyph
# nets the same total while shifting every position after them out of
# register. Checked directly: ~10% of the 'a' examples this gate lets
# through are unrelated shapes (fragments of neighbouring letters, once
# even a whole different letter) with negative mean cosine to their own
# class -- anti-correlated with their peers, not just noisy. A class needs
# enough examples for "agrees with its peers" to mean anything, so classes
# below MIN_CLASS_FOR_TRIM are left alone; their errors are rare letters'
# problem either way.
MIN_CLASS_FOR_TRIM = 6
INTRA_CLASS_FLOOR = 0.0


def drop_misaligned(glyphs: list[tuple[str, np.ndarray]], vector_fn) -> tuple[list, int]:
    """Drop examples that are anti-correlated with their own class's peers
    -- almost always a box that landed on the wrong letter, not the labelled
    one. Returns (kept, n_dropped)."""
    by_class: dict[str, list[int]] = {}
    for i, (ch, _patch) in enumerate(glyphs):
        by_class.setdefault(ch, []).append(i)

    drop = set()
    for ch, idxs in by_class.items():
        if len(idxs) < MIN_CLASS_FOR_TRIM:
            continue
        vecs = [vector_fn(glyphs[i][1]) for i in idxs]
        valid = [(i, v) for i, v in zip(idxs, vecs) if v is not None]
        if len(valid) < MIN_CLASS_FOR_TRIM:
            continue
        V = np.array([v for _, v in valid])
        S = V @ V.T
        np.fill_diagonal(S, np.nan)
        mean_sim = np.nanmean(S, axis=1)
        for (i, _v), sim in zip(valid, mean_sim):
            if sim < INTRA_CLASS_FLOOR:
                drop.add(i)

    kept = [g for i, g in enumerate(glyphs) if i not in drop]
    return kept, len(drop)


def load_real_digit_glyphs() -> list[tuple[str, np.ndarray]]:
    """Real, labelled digit crops from the badge atlas -- a separate
    source, since the badge plate may not share the name text's font."""
    templates, labels, _card = atlas_samples()
    return [(str(lbl), tpl) for tpl, lbl in zip(templates, labels)]


# --------------------------------------------------------------------------
# font rendering


def render_glyph(font: ImageFont.FreeTypeFont, char: str,
                  px: int = GLYPH_RENDER_PX) -> np.ndarray | None:
    """White-on-black render of one character, tight-cropped to its own ink."""
    canvas = px * 2
    img = Image.new("L", (canvas, canvas), 0)
    ImageDraw.Draw(img).text((canvas // 4, canvas // 4), char, font=font, fill=255)
    arr = np.array(img)
    ys, xs = np.where(arr > 16)
    if len(xs) == 0:
        return None
    return arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _variation_choices(font_path: Path, weight_names) -> list[str | None]:
    """Named instances to try for a variable font; [None] for a static one."""
    try:
        probe = ImageFont.truetype(str(font_path), 64)
        raw_names = probe.get_variation_names()
    except Exception:
        return [None]
    decoded = [n.decode() if isinstance(n, bytes) else n for n in raw_names]
    chosen = [n for n in weight_names if n in decoded]
    return chosen or [decoded[-1]]


def load_font(font_path: Path, variation: str | None,
              px: int = GLYPH_RENDER_PX) -> ImageFont.FreeTypeFont:
    font = ImageFont.truetype(str(font_path), px)
    if variation:
        font.set_variation_by_name(variation)
    return font


# --------------------------------------------------------------------------
# scoring


def score_font(font_path: Path, real_glyphs: list[tuple[str, np.ndarray]],
               vector_fn, weight_names=WEIGHT_PRIORITY,
               min_coverage: float = MIN_COVERAGE) -> list[dict]:
    """One result per weight instance tried. A font/weight that can't
    render most of the needed characters is dropped, not scored partial."""
    needed = sorted({ch for ch, _ in real_glyphs})
    results = []
    for variation in _variation_choices(font_path, weight_names):
        try:
            font = load_font(font_path, variation)
        except Exception:
            continue

        templates: dict[str, np.ndarray] = {}
        for ch in needed:
            patch = render_glyph(font, ch)
            v = vector_fn(patch) if patch is not None else None
            if v is not None:
                templates[ch] = v
        coverage = len(templates) / len(needed)
        if coverage < min_coverage:
            continue

        sims: list[float] = []
        per_letter: dict[str, list[float]] = {}
        for ch, patch in real_glyphs:
            tmpl = templates.get(ch)
            if tmpl is None:
                continue
            v = vector_fn(patch)
            if v is None:
                continue
            s = float(np.dot(tmpl, v))
            sims.append(s)
            per_letter.setdefault(ch, []).append(s)
        if len(sims) < min_coverage * len(real_glyphs):
            continue

        label = f"{font_path.stem}:{variation}" if variation else font_path.stem
        results.append({
            "label": label,
            "path": str(font_path),
            "variation": variation,
            "mean_cosine": float(np.mean(sims)),
            "n_glyphs": len(sims),
            "coverage": round(coverage, 3),
            "per_letter": {ch: round(float(np.mean(v)), 3) for ch, v in per_letter.items()},
        })
    return results


def local_font_files() -> list[Path]:
    if not WINDOWS_FONTS.exists():
        return []
    return sorted(set(WINDOWS_FONTS.glob("*.ttf")) | set(WINDOWS_FONTS.glob("*.otf")))


def downloaded_font_files() -> list[Path]:
    if not FONTS_DIR.exists():
        return []
    return sorted(set(FONTS_DIR.glob("*.ttf")) | set(FONTS_DIR.glob("*.otf")))


def fetch_fonts(force: bool = False) -> None:
    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    base = "https://raw.githubusercontent.com/google/fonts/main/ofl"
    for fam, remote, local in GOOGLE_FONTS:
        out = FONTS_DIR / local
        if out.exists() and not force:
            continue
        url = f"{base}/{fam}/{remote}"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        out.write_bytes(r.content)
        print(f"  fetched {local}  ({len(r.content):,} bytes)  <- {url}")


# --------------------------------------------------------------------------
# visual check: real line crop vs. the candidate font rendering the same text


def _crop_line(work_gray: np.ndarray, line_boxes, pad: int = 8) -> np.ndarray:
    xs = [b[0] for b in line_boxes]
    xe = [b[0] + b[2] for b in line_boxes]
    ys = [b[1] for b in line_boxes]
    ye = [b[1] + b[3] for b in line_boxes]
    x0, x1 = max(0, min(xs) - pad), max(xe) + pad
    y0, y1 = max(0, min(ys) - pad), max(ye) + pad
    return work_gray[y0:y1, x0:x1]


def _render_line(font: ImageFont.FreeTypeFont, text: str) -> np.ndarray:
    w = LINE_RENDER_PX * max(1, len(text)) * 2 + 200
    h = LINE_RENDER_PX * 3
    img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(img).text((100, LINE_RENDER_PX), text, font=font, fill=255)
    arr = np.array(img)
    ys, xs = np.where(arr > 16)
    if len(xs) == 0:
        return np.zeros((10, 10), dtype=np.uint8)
    return arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _pad_to_width(img: np.ndarray, width: int) -> np.ndarray:
    if img.shape[1] >= width:
        return img[:, :width]
    if img.ndim == 3:
        out = np.zeros((img.shape[0], width, img.shape[2]), dtype=img.dtype)
    else:
        out = np.zeros((img.shape[0], width), dtype=img.dtype)
    out[:, :img.shape[1]] = img
    return out


def _label_row(text: str, width: int, height: int = 26) -> np.ndarray:
    row = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(row, text, (4, height - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (80, 220, 80), 1, cv2.LINE_AA)
    return row


def _pair_panel(card: str, text: str, real_gray: np.ndarray,
                 font: ImageFont.FreeTypeFont, font_label: str, up: int = 3) -> np.ndarray:
    rendered = _render_line(font, text)
    scale = real_gray.shape[0] / max(rendered.shape[0], 1)
    new_w = max(1, int(round(rendered.shape[1] * scale)))
    rendered_scaled = cv2.resize(rendered, (new_w, real_gray.shape[0]),
                                  interpolation=cv2.INTER_AREA)

    real_big = cv2.resize(real_gray, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
    rend_big = cv2.resize(rendered_scaled, None, fx=up, fy=up, interpolation=cv2.INTER_NEAREST)

    width = max(real_big.shape[1], rend_big.shape[1]) + 20
    real_bgr = cv2.cvtColor(_pad_to_width(real_big, width), cv2.COLOR_GRAY2BGR)
    rend_bgr = cv2.cvtColor(_pad_to_width(rend_big, width), cv2.COLOR_GRAY2BGR)

    return np.vstack([
        _label_row(f'REAL {card}: "{text}"', width),
        real_bgr,
        _label_row(f"FONT {font_label}", width),
        rend_bgr,
    ])


def _stack(panels: list[np.ndarray], sep: int = 12) -> np.ndarray:
    width = max(p.shape[1] for p in panels)
    parts = []
    for i, p in enumerate(panels):
        if i > 0:
            parts.append(np.full((sep, width, 3), 55, dtype=np.uint8))
        parts.append(_pad_to_width(p, width))
    return np.vstack(parts)


def make_comparisons(top_results: list[dict]) -> list[Path]:
    bands = np.load(DATA / "name_bands.npz")
    with (DATA / "name_truth.csv").open(encoding="utf-8") as f:
        truth = {r["card"]: (r["true_character"], r["true_series"])
                 for r in csv.DictReader(f)}

    samples = []
    for card, line_idx in SAMPLE_LINES:
        work = prepare_band(bands[card])
        lines = segment_lines(work)
        if line_idx >= len(lines):
            continue
        character, series = truth[card]
        text = character if line_idx == 0 else series
        samples.append((card, text, _crop_line(work, lines[line_idx])))

    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for rank, r in enumerate(top_results, 1):
        font = load_font(Path(r["path"]), r.get("variation"), px=GLYPH_RENDER_PX)
        # re-load at the line render size (font object above was for scoring)
        line_font = load_font(Path(r["path"]), r.get("variation"), px=LINE_RENDER_PX)
        panels = [_pair_panel(card, text, img, line_font, r["label"])
                  for card, text, img in samples]
        composite = _stack(panels)
        safe = r["label"].replace(":", "_").replace(" ", "_").replace("/", "_")
        out = COMPARE_DIR / f"{rank}_{safe}.png"
        cv2.imwrite(str(out), composite)
        written.append(out)
        print(f"  wrote {out}")
    return written


# --------------------------------------------------------------------------


def _print_table(results: list[dict], n: int) -> None:
    print(f"{'rank':>4}  {'font':42}  {'cosine':>7}  {'n':>4}  {'cover':>6}")
    for i, r in enumerate(results[:n], 1):
        print(f"{i:>4}  {r['label'][:42]:42}  {r['mean_cosine']:.4f}  "
              f"{r['n_glyphs']:>4}  {r['coverage']:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetch-fonts", action="store_true",
                     help="(re)download the curated Google Fonts candidates into tools/fonts/")
    ap.add_argument("--compare", type=int, default=3,
                     help="write side-by-side visual-check images for the top N letter-score candidates")
    ap.add_argument("--top", type=int, default=20, help="rows to print per table")
    ap.add_argument("--out", type=Path, default=None, help="write full results as JSON")
    args = ap.parse_args()

    if args.fetch_fonts:
        print(f"fetching candidate fonts into {FONTS_DIR} ...")
        fetch_fonts()

    real_letters = load_real_letter_glyphs()
    real_digits = load_real_digit_glyphs()
    print(f"real letter glyphs: {len(real_letters)}  "
          f"({len({c for c, _ in real_letters})} unique characters)")
    print(f"real digit glyphs:  {len(real_digits)}  "
          f"({len({c for c, _ in real_digits})} unique classes)")

    real_letters, n_dropped = drop_misaligned(real_letters, letter_vector)
    if n_dropped:
        print(f"  dropped {n_dropped} letter examples anti-correlated with "
              f"their own class's peers (misaligned box, not a bad font) "
              f"-> {len(real_letters)} remain")

    font_files = local_font_files() + downloaded_font_files()
    print(f"\nscoring {len(font_files)} font files "
          f"({len(local_font_files())} local + {len(downloaded_font_files())} downloaded) ...")

    letter_results: list[dict] = []
    digit_results: list[dict] = []
    for fp in font_files:
        try:
            letter_results += score_font(fp, real_letters, letter_vector)
        except Exception as e:
            print(f"  [skip, letters] {fp.name}: {e}")
        try:
            digit_results += score_font(fp, real_digits, digit_vector)
        except Exception as e:
            print(f"  [skip, digits] {fp.name}: {e}")

    letter_results.sort(key=lambda r: -r["mean_cosine"])
    digit_results.sort(key=lambda r: -r["mean_cosine"])

    print(f"\n{len(letter_results)} font/weight combinations cleared the "
          f"{MIN_COVERAGE:.0%} coverage bar for letters; "
          f"{len(digit_results)} for digits.")

    print("\n=== TOP CANDIDATES vs NAME-TEXT LETTERS (the decision metric) ===")
    _print_table(letter_results, args.top)

    print("\n=== TOP CANDIDATES vs BADGE DIGIT ATLAS (corroborating only) ===")
    _print_table(digit_results, args.top)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"letters": letter_results, "digits": digit_results}, indent=2))
        print(f"\nwrote full results to {args.out}")

    if letter_results:
        top = letter_results[0]
        runner_up = letter_results[1]["mean_cosine"] if len(letter_results) > 1 else 0.0
        margin = top["mean_cosine"] - runner_up
        print(f"\nTop letters score: {top['mean_cosine']:.4f} ({top['label']}); "
              f"margin over runner-up: {margin:.4f}")
        meets_bar = top["mean_cosine"] >= 0.90 and margin >= 0.05
        print(f"Numeric bar (>=0.90 and >=0.05 clear of runner-up): "
              f"{'MET -- still needs the visual check' if meets_bar else 'NOT MET'}")

    if args.compare > 0 and letter_results:
        print(f"\nwriting visual comparisons for the top {args.compare} candidates ...")
        make_comparisons(letter_results[:args.compare])


if __name__ == "__main__":
    main()
