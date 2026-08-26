"""Classify a card's border/frame rarity from pixels alone.

The insight that makes this tractable without training data: rarity frames
are *chromatically* louder than common ones. A common frame is flat -- one
colour, or grey. Rare and holographic frames sweep through many hues at high
saturation. So we measure the colour behaviour of the border ring:

    hue_entropy    how evenly hue mass is spread          (rainbow -> high)
    hue_diversity  effective number of hues present, as a fraction of all
                   buckets. This is perplexity, exp(entropy), NOT a count of
                   buckets above a fixed share: a true rainbow spreads ~5.6%
                   into each of 18 buckets and so clears a 6% cut-off *less*
                   often than a coarse 5-hue frame does. Counting buckets is
                   anti-correlated with diversity; perplexity is monotonic.
    sat_mean       how vivid the ring is                  (foil -> high)
    colored_frac   how much of the ring is chromatic at all

These combine into a 0..1 ``rarity_index``. Thresholds are exposed and
every feature is returned, because the cut points genuinely need calibrating
against real spawns -- see ``calibrate`` in the CLI.
"""

from __future__ import annotations

import cv2
import numpy as np

from .models import FrameTier

# rarity_index cut points, worst -> best.
THRESHOLDS = {
    FrameTier.UNCOMMON: 0.28,
    FrameTier.RARE: 0.55,
    FrameTier.HOLO: 0.80,
}
# Below this fraction of chromatic pixels the ring is grey/flat: common.
MIN_COLORED_FRAC = 0.12
_HUE_BINS = 18          # 10-degree buckets across OpenCV's 0..179 hue range


def _ring_mask(h: int, w: int, band: float = 0.13) -> np.ndarray:
    """Boolean mask of the outer border band, excluding the inner artwork."""
    m = np.ones((h, w), dtype=bool)
    by, bx = int(h * band), int(w * band)
    m[by:h - by, bx:w - bx] = False
    return m


def frame_features(card_bgr: np.ndarray, band: float = 0.13) -> dict[str, float]:
    h, w = card_bgr.shape[:2]
    if h < 8 or w < 8:
        return {"rarity_index": 0.0, "hue_entropy": 0.0, "hue_diversity": 0.0,
                "sat_mean": 0.0, "colored_frac": 0.0}

    hsv = cv2.cvtColor(card_bgr, cv2.COLOR_BGR2HSV)
    ring = hsv[_ring_mask(h, w, band)]
    if ring.size == 0:
        return {"rarity_index": 0.0, "hue_entropy": 0.0, "hue_diversity": 0.0,
                "sat_mean": 0.0, "colored_frac": 0.0}

    hue, sat, val = ring[:, 0], ring[:, 1], ring[:, 2]
    chromatic = (sat > 45) & (val > 45)
    colored_frac = float(chromatic.mean())

    if colored_frac < 1e-6:
        return {"rarity_index": 0.0, "hue_entropy": 0.0, "hue_diversity": 0.0,
                "sat_mean": 0.0, "colored_frac": 0.0}

    ch = hue[chromatic]
    hist, _ = np.histogram(ch, bins=_HUE_BINS, range=(0, 180))
    p = hist.astype(np.float64)
    p /= p.sum()

    nz = p[p > 0]
    entropy_nats = float(-(nz * np.log(nz)).sum())
    entropy = entropy_nats / np.log(_HUE_BINS)                  # 0..1
    # Perplexity: the effective number of hues actually in play.
    hue_diversity = float(np.exp(entropy_nats) / _HUE_BINS)     # ~0.06..1
    sat_mean = float(sat[chromatic].mean())

    rarity = (
        0.45 * entropy
        + 0.30 * hue_diversity
        + 0.25 * min(1.0, sat_mean / 190.0)
    )
    # A mostly-grey ring cannot be a rare frame no matter how its few
    # coloured pixels are distributed.
    rarity *= min(1.0, colored_frac / MIN_COLORED_FRAC)

    return {
        "rarity_index": round(float(rarity), 4),
        "hue_entropy": round(float(entropy), 4),
        "hue_diversity": round(hue_diversity, 4),
        "sat_mean": round(sat_mean, 1),
        "colored_frac": round(colored_frac, 4),
    }


def classify_frame(card_bgr: np.ndarray, band: float = 0.13) -> tuple[FrameTier, dict[str, float]]:
    f = frame_features(card_bgr, band)
    idx = f["rarity_index"]
    if f["colored_frac"] < MIN_COLORED_FRAC:
        return FrameTier.COMMON, f
    if idx >= THRESHOLDS[FrameTier.HOLO]:
        return FrameTier.HOLO, f
    if idx >= THRESHOLDS[FrameTier.RARE]:
        return FrameTier.RARE, f
    if idx >= THRESHOLDS[FrameTier.UNCOMMON]:
        return FrameTier.UNCOMMON, f
    return FrameTier.COMMON, f
