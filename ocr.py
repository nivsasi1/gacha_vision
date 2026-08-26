"""Read the print-number badge and the character/series text off a card.

The badge is a short token near the top of the card: a number (best), or the
letter ``E`` for an un-numbered edition.

Naively OCR-ing the whole top strip does not work -- the frame is loud, Otsu
latches onto the border rather than the glyphs, and Tesseract will happily
hallucinate an ``E`` out of a two-digit number. Misreading a good card as junk
is the worst error this module can make, so instead we:

  1. mask the strip several ways (bright text, dark text, Otsu both ways),
  2. take connected components that are glyph-shaped,
  3. discard components that *contain* other components -- those are the
     badge plate and its border, not text,
  4. group what remains into lines, crop tight, upscale hard,
  5. OCR every candidate and let confidence-weighted votes decide.

An ``E`` is only reported when no candidate produced digits.
"""

from __future__ import annotations

import os
import re
import shutil
import sys

import cv2
import numpy as np
import pytesseract

# Windows installers put tesseract.exe somewhere PATH does not reach, which
# is the first thing that breaks a fresh checkout there. Look for it before
# giving up, and let TESSERACT_CMD override everything.
_WINDOWS_CANDIDATES = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    os.path.expandvars(r"%USERPROFILE%\AppData\Local\Tesseract-OCR\tesseract.exe"),
)


def find_tesseract() -> str | None:
    """Locate the tesseract binary, or None if it cannot be found."""
    override = os.environ.get("TESSERACT_CMD")
    if override and os.path.isfile(override):
        return override
    on_path = shutil.which("tesseract")
    if on_path:
        return on_path
    if sys.platform.startswith("win"):
        for cand in _WINDOWS_CANDIDATES:
            if cand and os.path.isfile(cand):
                return cand
    return None


def configure_tesseract() -> str | None:
    """Point pytesseract at the binary. Returns the path used, or None."""
    found = find_tesseract()
    if found:
        pytesseract.pytesseract.tesseract_cmd = found
    return found


configure_tesseract()

# psm 8 ("single word") and 13 ("raw line") are the only modes that reliably
# handle a tight, stylised numeric crop; psm 7 returns empty on these.
_PSMS = (8, 13)
_WHITELIST = "-c tessedit_char_whitelist=0123456789E"
_NAME_CFG = "--oem 3 --psm 6"

BADGE_STRIP = 0.25          # fraction of card height searched for the badge
# ...and only the right-hand part of it. The badge sits top-right on every
# card observed. Searching the full width dragged in the frame's left-hand
# decoration -- on the ornate frame those corner orbs alone produced enough
# glyph-shaped components to bury a small badge and to cost a tesseract
# spawn each.
BADGE_LEFT_EDGE = 0.42
# The strip is resampled to this height before anything is measured, so the
# fraction-based filters below mean the same thing on a 200px card and a
# 900px one. Without it the filters were implicitly tuned to one card size:
# real badges are ~3-4% of card height, and at that scale the digits fell
# under the minimum-height cut and were discarded -- leaving only the fixed
# hook icon, which every card shares and Tesseract reads as "4".
_WORK_STRIP_H = 220
# Glyph-size floors, as fractions of the NORMALISED strip -- which is what
# makes them meaningful rather than tuned to one card size. Swept against
# real badge proportions; see the note in _glyph_boxes.
_MIN_GLYPH_H_FRAC = 0.06
_MIN_GLYPH_AREA = 12
# Below this confidence a read is evidence, not fact. Real spawns produced
# 93% low-confidence reads against 3% on synthetic cards, and those shaky
# reads were driving "take both" -- so the number is load-bearing, not
# cosmetic, and lives here rather than in three separate call sites.
MIN_TRUSTED_CONFIDENCE = 0.55
_MIN_TOKEN_CONF = 4.0       # weight for a token Tesseract emitted but did not score
# pytesseract shells out to the tesseract binary, so every candidate crop
# costs a process spawn. Once a trustworthy glyph crop has produced a numeric
# read this confident, more candidates only cost time.
_EARLY_EXIT_CONF = 85.0
# An 'E' badge never trips the numeric early exit -- Tesseract is chronically
# unconfident about it -- so without a cap those cards scan every candidate of
# every mask and cost ~6x a numeric read. Candidates are ordered best-first
# (tight glyph crops before plate interiors), so a cap drops the weakest
# evidence, not the decisive kind.
_MAX_CANDIDATES = 6


def _masks(gray: np.ndarray) -> list[np.ndarray]:
    """Several binarisations, so we do not depend on one thresholding guess."""
    out = [
        (gray > 190).astype(np.uint8) * 255,      # light text on a dark plate
        (gray < 70).astype(np.uint8) * 255,       # dark text on a light plate
    ]
    _, o = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out.append(o)
    out.append(cv2.bitwise_not(o))
    return out


def _normalise_strip(gray: np.ndarray) -> np.ndarray:
    """Resample the badge strip to a fixed height.

    Every filter downstream is a fraction of the strip's size, so without
    this they silently mean different things at different resolutions.
    """
    h, w = gray.shape[:2]
    if h == 0 or h == _WORK_STRIP_H:
        return gray
    scale = _WORK_STRIP_H / h
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(gray, (max(1, int(w * scale)), _WORK_STRIP_H), interpolation=interp)


def _glyph_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    H, W = mask.shape
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in stats[i][:5])
        if h < _MIN_GLYPH_H_FRAC * H or h > 0.85 * H:
            continue
        if w < 2 or w > 0.45 * W:
            continue
        if a < _MIN_GLYPH_AREA:
            continue
        boxes.append((x, y, w, h))
    return boxes


def _partition_boxes(boxes):
    """Split components into (plates, glyphs).

    A component enclosing *any* other is the badge plate or its border, not a
    character. Reading a plate box directly feeds the border strokes to
    Tesseract, which is how a '1' becomes a '2'. Plates are still useful --
    their interior is where the glyphs live -- so they are kept separately
    and contribute an inset crop rather than being discarded.
    """
    plates, glyphs = [], []
    for b in boxes:
        bx, by, bw, bh = b
        encloses = any(
            o is not b and bx <= o[0] and by <= o[1]
            and bx + bw >= o[0] + o[2] and by + bh >= o[1] + o[3]
            for o in boxes
        )
        (plates if encloses else glyphs).append(b)
    return plates, glyphs


def _within_any(box, plates, slack: float = 0.15) -> bool:
    """Is this glyph line sitting inside one of the plates?"""
    x, y, w, h = box
    for px, py, pw, ph in plates:
        mx, my = pw * slack, ph * slack
        if (x >= px - mx and y >= py - my
                and x + w <= px + pw + mx and y + h <= py + ph + my):
            return True
    return False


def _inset(box, frac: float = 0.18):
    """Shrink a plate box inward past its border stroke."""
    x, y, w, h = box
    dx, dy = int(w * frac), int(h * frac)
    return (x + dx, y + dy, max(1, w - 2 * dx), max(1, h - 2 * dy))


def _group_lines(boxes, tol: float = 0.5) -> list[tuple[int, int, int, int]]:
    """Cluster glyph boxes that share a horizontal band into line bboxes."""
    lines: list[list[tuple[int, int, int, int]]] = []
    for b in sorted(boxes, key=lambda x: x[1]):
        placed = False
        for ln in lines:
            ry = sum(o[1] + o[3] / 2 for o in ln) / len(ln)
            rh = sum(o[3] for o in ln) / len(ln)
            if abs((b[1] + b[3] / 2) - ry) <= tol * rh:
                ln.append(b)
                placed = True
                break
        if not placed:
            lines.append([b])
    out = []
    for ln in lines:
        x0 = min(o[0] for o in ln)
        y0 = min(o[1] for o in ln)
        x1 = max(o[0] + o[2] for o in ln)
        y1 = max(o[1] + o[3] for o in ln)
        out.append(((x0, y0, x1 - x0, y1 - y0), ln))
    return out


def _ocr_crop(gray_crop: np.ndarray) -> list[tuple[str, float]]:
    """OCR one tight crop; return (text, confidence) per page-seg mode.

    Polarity is *decided*, not averaged. Handing Tesseract an inverted image
    and trusting its confidence is actively harmful: a bare '1' bar reads as
    '4' at confidence 69 in the wrong polarity while the correct '1' scores
    12, so averaging both lets the wrong answer win. Ink is the minority
    class in a tight glyph crop, so we normalise to black-on-white -- what
    Tesseract is trained for -- and read that once.
    """
    if gray_crop.size == 0 or min(gray_crop.shape[:2]) < 4:
        return []
    # Aim for a glyph around 64px tall: a fixed 6x under-scales a small crop
    # and needlessly over-scales a large one.
    scale = max(2.0, min(8.0, 64.0 / max(1, gray_crop.shape[0])))
    up = cv2.resize(gray_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    _, th = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float((th == 255).mean()) < 0.5:      # white is the minority -> it is the ink
        th = cv2.bitwise_not(th)
    th = cv2.copyMakeBorder(th, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)

    results = []
    for psm in _PSMS:
        data = pytesseract.image_to_data(
            th, config=f"--oem 3 --psm {psm} {_WHITELIST}",
            output_type=pytesseract.Output.DICT,
        )
        toks = [(t.strip(), _conf(c)) for t, c in zip(data["text"], data["conf"]) if t.strip()]
        if not toks:
            continue
        conf = sum(c for _, c in toks) / len(toks)
        results.append(("".join(t for t, _ in toks), conf))
        if conf >= _EARLY_EXIT_CONF:
            break                       # a confident read; other modes add nothing
    return results


def read_badge(card_bgr: np.ndarray) -> dict:
    """Return {print_no, no_number, confidence, text}.

    ``print_no`` is int or None. ``no_number`` is True only for a clean 'E'
    read with no digits anywhere -- never as a fallback for failed OCR.
    """
    h, w = card_bgr.shape[:2]
    x0 = int(BADGE_LEFT_EDGE * w)
    strip = card_bgr[0:max(4, int(BADGE_STRIP * h)), x0:]
    gray = _normalise_strip(cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY))
    sw = gray.shape[1]

    seen_boxes: set[tuple[int, int, int, int]] = set()
    votes: dict[str, float] = {}        # label ("1584" / "E") -> weight
    best_conf: dict[str, float] = {}    # label -> best raw Tesseract confidence
    raw: list[str] = []

    done = False
    for mask in _masks(gray):
        if done:
            break
        plates, glyphs = _partition_boxes(_glyph_boxes(mask))
        # A tight crop around real character components is far more
        # trustworthy than a plate interior, which may still catch border
        # strokes. Weight the evidence accordingly rather than letting a
        # contaminated plate read outvote a clean glyph read.
        # The badge is a plate with glyphs inside it. When such a plate is
        # present it identifies the badge exactly, so restrict the search to
        # its contents: the ornate frame's corner orbs are solid blobs that
        # enclose nothing, so they stop competing -- they were being read as
        # "2" over a real "E" -- and each one skipped is a tesseract spawn
        # saved. Only with no plate at all do we fall back to open search.
        lines = _group_lines(glyphs)
        if plates:
            inside = [(box, 1.0) for box, _ in lines if _within_any(box, plates)]
            candidates = inside + [(_inset(p), 0.6) for p in plates]
        else:
            candidates = [(box, 1.0) for box, _ in lines]
        # Rightmost first: the badge sits at the right edge, so the early
        # exit tends to fire before the rest is scanned.
        candidates.sort(key=lambda c: (-c[1], -(c[0][0] + c[0][2] / 2)))
        for box, src_weight in candidates:
            x, y, bw, bh = box
            key = (x // 4, y // 4, bw // 4, bh // 4)     # coarse dedupe
            if key in seen_boxes:
                continue
            seen_boxes.add(key)
            if len(seen_boxes) > _MAX_CANDIDATES:
                done = True
                break
            pad = max(3, bh // 5)
            crop = gray[max(0, y - pad):y + bh + pad, max(0, x - pad):x + bw + pad]
            # Badges sit toward the top-right; nudge those candidates up.
            side_bonus = 1.15 if (x + bw / 2) > sw * 0.5 else 1.0

            for text, conf in _ocr_crop(crop):
                raw.append(text)
                digits = re.findall(r"\d+", text)
                labels = digits if digits else (["E"] if "E" in text.upper() else [])
                for lab in labels:
                    w = conf * len(lab) * side_bonus * src_weight
                    votes[lab] = votes.get(lab, 0.0) + w
                    best_conf[lab] = max(best_conf.get(lab, 0.0), conf)

            # Stop only on a confident read from a tight glyph crop -- plate
            # interiors are the contaminated source and never end the search.
            if src_weight >= 1.0 and votes:
                lead = max(votes, key=votes.get)
                if best_conf.get(lead, 0.0) >= _EARLY_EXIT_CONF:
                    done = True
                    break

    text = " ".join(dict.fromkeys(raw))[:60]
    if not votes:
        return {"print_no": None, "no_number": False, "confidence": 0.0, "text": text}

    best = max(votes, key=votes.get)
    # Confidence reflects both how sure Tesseract was and how much the
    # candidates agreed -- a split vote is exactly what should be reviewed.
    share = votes[best] / sum(votes.values())
    conf = round(min(1.0, share * min(1.0, best_conf[best] / 85.0)), 3)
    if best == "E":
        return {"print_no": None, "no_number": True, "confidence": conf, "text": text}
    return {"print_no": int(best), "no_number": False, "confidence": conf, "text": text}


def read_name(card_bgr: np.ndarray) -> str:
    """Best-effort OCR of the name block near the bottom of the card."""
    h = card_bgr.shape[0]
    strip = card_bgr[int(0.72 * h):, :]
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    up = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    up = cv2.bilateralFilter(up, 5, 55, 55)
    best = ""
    for img in (up, cv2.bitwise_not(up)):
        _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        txt = pytesseract.image_to_string(th, config=_NAME_CFG)
        txt = re.sub(r"[^A-Za-z0-9 \n]+", " ", txt)
        txt = "  ".join(" ".join(ln.split()) for ln in txt.splitlines() if ln.strip())
        if len(txt) > len(best):
            best = txt
    return best.strip()


def _conf(raw) -> float:
    """Tesseract reports -1 (and sometimes 0) even for tokens it did emit.

    A token that exists is evidence, so floor it at a small positive weight
    rather than throwing the read away -- otherwise a clean 'E', which
    Tesseract is chronically unconfident about, is discarded entirely.
    """
    try:
        v = float(raw)
    except (TypeError, ValueError):
        v = 0.0
    return max(_MIN_TOKEN_CONF, v)
