"""Measure a card's border, to corroborate the frame the badge implies.

The badge decides the frame: a card showing a print number wears ``NORMAL``,
a card showing ``E`` wears ``E``. That is the game's own definition and it
needs no pixels, so this module is deliberately *not* the authority.

What it does is measure how ornate the border is, which is a real and
separable signal -- the ``E`` frame is a thick gold/chain border with rainbow
corner orbs, while ``NORMAL`` is a thin pale one. That measurement is worth
keeping for two reasons:

  * it cross-checks the badge, so a card whose border disagrees with its
    number is surfaced rather than silently trusted, and
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
