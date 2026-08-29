"""Measure a card's border, and decide its frame from that measurement.

``E`` is by definition the frame with no print number, so reading the badge
looks like the exact answer and the border only a guess. Real cards say the
opposite. Over 182 hand-labelled ones the border called the frame right 182
times and the badge 160, and in every one of the 22 disagreements the border
was the one that was right.

The asymmetry is in the difficulty, not the definition: separating a gold ring
from a pale one is a measurement over thousands of border pixels, while
recognising the badge glyph is a few strokes beside a hook icon shaped like a
``1``, at a size where Tesseract drops digits outright. So the border settles
whether a card carries a number and the badge is left to supply the digits.

The ``E`` frame is a thick gold/chain border with rainbow corner orbs;
``NORMAL`` is a thin pale one. Keeping the badge's opinion alongside is still
worth it:

  * it cross-checks the border, so a card whose badge disagrees is surfaced
    for review rather than silently overruled, and
  * an unfamiliar frame -- one that matches neither -- is exactly the kind of
    card worth a human glance, since every frame seen so far is a common.

    ornateness     composite of the three below, 0..1
    hue_entropy    how evenly hue mass is spread          (rainbow -> high)
    hue_diversity  effective number of hues, as a fraction of all buckets.
                   This is perplexity, exp(entropy), NOT a count of buckets
                   above a fixed share: a true rainbow spreads ~5.6% into
                   each of 18 buckets and so clears a 6% cut-off *less* often
                   than a coarse 5-hue frame does. Counting buckets is
                   anti-correlated with diversity; perplexity is monotonic.
    sat_mean       how vivid the ring is                  (gold/foil -> high)
    colored_frac   how much of the ring is chromatic at all

The cut point below is fitted to 182 labelled real cards.
"""

from __future__ import annotations

import functools
from pathlib import Path

import cv2
import numpy as np

from .models import FrameTier

# Fitted to 182 labelled real cards, and the direction is the opposite of
# what it looks like by eye. The E border is gold: one hue family, deeply
# saturated. The NORMAL border is a pale rainbow: many hues, low saturation.
# So hue variety is HIGHER on NORMAL, and it was saturation that separated
# them cleanly -- 99% against 95% for the composite, and 74% for how much of
# the ring carries colour at all. E sits above the cut.
E_SATURATION = 152.65
# Below this fraction of chromatic pixels the ring is grey/flat.
MIN_COLORED_FRAC = 0.12

# How far a border's colour may sit from the nearest catalogued frame before
# it counts as one nobody here has seen. Leave-one-out over 181 catalogued
# cards: at most 0.110 (median 0.0006), against 0.190 for the single card
# wearing an uncatalogued frame. The cut sits in that gap.
#
# It rests on exactly one positive example. A second example of any
# unfamiliar frame is worth more here than any amount of tuning.
UNKNOWN_FRAME_DISTANCE = 0.15

_FRAME_ATLAS = Path(__file__).parent / "data" / "frame_atlas.npz"
_RING_BINS = 12
_HUE_BINS = 18          # 10-degree buckets across OpenCV's 0..179 hue range

_EMPTY = {"ornateness": 0.0, "hue_entropy": 0.0, "hue_diversity": 0.0,
          "sat_mean": 0.0, "colored_frac": 0.0}


def _ring_mask(h: int, w: int, band: float = 0.13) -> np.ndarray:
    """Boolean mask of the outer border band, excluding the inner artwork."""
    m = np.ones((h, w), dtype=bool)
    by, bx = int(h * band), int(w * band)
    m[by:h - by, bx:w - bx] = False
    return m


def frame_features(card_bgr: np.ndarray, band: float = 0.13) -> dict[str, float]:
    h, w = card_bgr.shape[:2]
    if h < 8 or w < 8:
        return dict(_EMPTY)

    hsv = cv2.cvtColor(card_bgr, cv2.COLOR_BGR2HSV)
    ring = hsv[_ring_mask(h, w, band)]
    if ring.size == 0:
        return dict(_EMPTY)

    hue, sat, val = ring[:, 0], ring[:, 1], ring[:, 2]
    chromatic = (sat > 45) & (val > 45)
    colored_frac = float(chromatic.mean())
    if colored_frac < 1e-6:
        return dict(_EMPTY)

    hist, _ = np.histogram(hue[chromatic], bins=_HUE_BINS, range=(0, 180))
    p = hist.astype(np.float64)
    p /= p.sum()

    nz = p[p > 0]
    entropy_nats = float(-(nz * np.log(nz)).sum())
    entropy = entropy_nats / np.log(_HUE_BINS)                  # 0..1
    hue_diversity = float(np.exp(entropy_nats) / _HUE_BINS)     # ~0.06..1
    sat_mean = float(sat[chromatic].mean())

    ornateness = (
        0.45 * entropy
        + 0.30 * hue_diversity
        + 0.25 * min(1.0, sat_mean / 190.0)
    )
    # A mostly-grey ring cannot be an ornate frame however its few coloured
    # pixels are arranged.
    ornateness *= min(1.0, colored_frac / MIN_COLORED_FRAC)

    return {
        "ornateness": round(float(ornateness), 4),
        "hue_entropy": round(float(entropy), 4),
        "hue_diversity": round(hue_diversity, 4),
        "sat_mean": round(sat_mean, 1),
        "colored_frac": round(colored_frac, 4),
    }


def guess_frame(card_bgr: np.ndarray, band: float = 0.13) -> tuple[FrameTier, dict[str, float]]:
    """Guess the frame from the border alone.

    Only ever a second opinion: :func:`frame_from_badge` is authoritative.
    Splits on ring saturation, which fitted the labels at 99%.
    """
    f = frame_features(card_bgr, band)
    # Before choosing between the two known frames, ask whether this border
    # is either of them. A frame nobody has catalogued is the one worth
    # claiming on sight, and saying "neither" is answerable from one example
    # where naming the frame is not.
    f["frame_distance"] = round(distance_to_known_frames(card_bgr), 4)
    if f["frame_distance"] > UNKNOWN_FRAME_DISTANCE:
        return FrameTier.OTHER, f
    if f["colored_frac"] < MIN_COLORED_FRAC:
        return FrameTier.NORMAL, f
    return (FrameTier.E if f["sat_mean"] >= E_SATURATION else FrameTier.NORMAL), f


def frame_from_badge(print_no: int | None, no_number: bool) -> FrameTier:
    """The authoritative frame: decided by whether the card has a number.

    This is the game's own distinction -- ``E`` is precisely the frame with
    no print number -- so it needs no pixels and cannot be fooled by how
    decorated a border happens to be.
    """
    if no_number:
        return FrameTier.E
    if print_no is not None:
        return FrameTier.NORMAL
    return FrameTier.UNKNOWN


def resolve_frame(
    print_no: int | None, no_number: bool, pixel_frame: str
) -> tuple[FrameTier, int | None, bool]:
    """Reconcile the badge read with the border measurement.

    The border decides *whether* a card carries a print number; the badge only
    supplies the digits. That is the opposite of how this started, and the
    labels are unambiguous about it: over 182 real cards the border called the
    frame correctly 182 times and the badge 160, and in all 22 disagreements
    the border was the one that was right.

    The reason is that the two jobs are not equally hard. Telling a gold ring
    from a pale one is a whole-border measurement over thousands of pixels;
    recognising a glyph is a handful of strokes next to a hook icon that looks
    like a ``1``, at a size where Tesseract drops digits. So the border answers
    the E/number question and the badge is left the job it can do.

    A card whose digits the badge could not read still has a known frame -- the
    border settled that -- so it comes back ``NORMAL`` with ``print_no=None``,
    the state the ranker already scores as "unreadable, review" and the batch
    flags as ``unreadable_badge``. Calling it an E instead would throw away a
    real print, which is the error worth avoiding here.

    Returns the card's ``(frame, print_no, no_number)``.
    """
    if pixel_frame == FrameTier.E.value:
        return FrameTier.E, None, True
    if pixel_frame == FrameTier.NORMAL.value:
        return FrameTier.NORMAL, print_no, False
    if pixel_frame == FrameTier.OTHER.value:
        # OTHER describes the border, not the badge: a number on an
        # uncatalogued frame still reads, and cards30 shows the badge can
        # also sit somewhere the reader does not look. Either way the card
        # is claimed for its frame, so an unread number costs nothing.
        return FrameTier.OTHER, print_no, False
    # No usable border reading: fall back to what the badge said.
    return frame_from_badge(print_no, no_number), print_no, no_number


def ring_descriptor(card_bgr: np.ndarray, band: float = 0.13,
                    top_skip: float = 0.22, bins: int = 18) -> np.ndarray | None:
    """Where the border's colour sits on the hue wheel.

    Hue only, and deliberately. Saturation and brightness move with how the
    card was cropped -- a few pixels of background along one edge shifts them
    enough that a badly cut card scores as unfamiliar as a genuinely unknown
    frame, which is exactly the confusion this has to avoid. Hue does not
    move: gold is gold whether or not the crop is tight.

    The top of the card is skipped because the badge plate sits there, and
    including it made this describe the *number* as much as the frame.

    Returns None when the border carries almost no colour at all, which is
    not a frame this can judge.
    """
    img = cv2.resize(card_bgr, (96, 144), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    mask = np.ones((h, w), dtype=bool)
    by, bx = int(h * band), int(w * band)
    mask[by:h - by, bx:w - bx] = False
    mask[:int(h * top_skip), :] = False

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ring = hsv[mask]
    hue, sat, val = ring[:, 0], ring[:, 1], ring[:, 2]
    chromatic = (sat > 60) & (val > 60)
    if int(chromatic.sum()) < 20:
        return None

    hist, _ = np.histogram(hue[chromatic], bins=bins, range=(0, 180))
    hist = hist.astype(np.float64)
    hist /= hist.sum()
    return hist / max(float(np.linalg.norm(hist)), 1e-9)


def frame_atlas_samples() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Descriptors of every catalogued frame, their class, and their card."""
    with np.load(_FRAME_ATLAS, allow_pickle=False) as z:
        return z["descriptors"], z["labels"], z["card"]


@functools.lru_cache(maxsize=1)
def _frame_atlas():
    try:
        return frame_atlas_samples()
    except FileNotFoundError:
        return None


def distance_to_known_frames(card_bgr: np.ndarray) -> float:
    """How unfamiliar this border is. 0 means identical to something known."""
    atlas = _frame_atlas()
    if atlas is None:
        return 0.0
    desc, _, _ = atlas
    d = ring_descriptor(card_bgr)
    if d is None:
        return 0.0          # too little colour to judge; not a claim on its own
    return float((1.0 - desc @ d).min())
