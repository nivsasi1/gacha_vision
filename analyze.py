"""Screenshot -> Cards -> Decision. Ties the vision stages to the ranker."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import Policy, load_watchlist
from .frame import frame_from_badge, guess_frame
from .models import Card, Decision
from .ocr import read_badge, read_name
from .rank import decide
from .segment import find_cards


def load_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return img


def split_name(raw: str) -> tuple[str, str]:
    """Card name blocks read as '<character> <series>' across two lines.

    Without a title database this is a guess, so we return the whole string
    as the character and let the watchlist match on substrings.
    """
    parts = [p for p in raw.split("  ") if p.strip()]
    if len(parts) >= 2:
        return parts[0].strip(), " ".join(parts[1:]).strip()
    return raw.strip(), ""


def analyze_cards_with_boxes(
    bgr: np.ndarray,
    expected: int | None = None,
    layout: str = "auto",
    read_names: bool = True,
) -> tuple[list[Card], list[tuple[int, int, int, int]]]:
    """Same as :func:`analyze_cards` but also returns the crop boxes.

    Callers that need the pixels back -- the calibration extractor, mainly --
    should not have to re-run segmentation to get them.
    """
    boxes = find_cards(bgr, expected, layout)
    cards: list[Card] = []
    for i, (x, y, w, h) in enumerate(boxes, start=1):
        crop = bgr[y:y + h, x:x + w]
        badge = read_badge(crop)
        # The badge is authoritative: E is exactly the frame with no print
        # number. The border measurement only corroborates it, and a
        # disagreement is surfaced rather than allowed to override.
        tier = frame_from_badge(badge["print_no"], badge["no_number"])
        pixel_tier, feats = guess_frame(crop)
        feats = dict(feats, pixel_frame=pixel_tier.value)
        character, series = ("", "")
        if read_names:
            character, series = split_name(read_name(crop))
        cards.append(
            Card(
                slot=i,
                print_no=badge["print_no"],
                no_number=badge["no_number"],
                frame=tier,
                character=character,
                series=series,
                ocr_text=badge["text"],
                ocr_confidence=round(badge["confidence"], 3),
                frame_features=feats,
            )
        )
    return cards, boxes


def analyze_cards(
    bgr: np.ndarray,
    expected: int | None = None,
    layout: str = "auto",
    read_names: bool = True,
) -> list[Card]:
    return analyze_cards_with_boxes(bgr, expected, layout, read_names)[0]


def analyze_spawn(
    path: str | Path,
    policy: Policy | None = None,
    watchlist_path: str | Path | None = None,
    expected: int | None = None,
    layout: str = "auto",
    read_names: bool = True,
) -> tuple[list[Card], Decision]:
    policy = policy or Policy()
    cards = analyze_cards(load_image(path), expected, layout, read_names)
    return cards, decide(cards, policy, load_watchlist(watchlist_path))
