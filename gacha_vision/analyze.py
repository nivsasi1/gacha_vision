"""Screenshot -> Cards -> Decision. Ties the vision stages to the ranker."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import Policy, load_watchlist
from .digits import MIN_TRUSTED_MATCH, read_print_number
from .frame import frame_from_badge, guess_frame, resolve_frame
from .models import Card, Decision, FrameTier
from .ocr import MIN_TRUSTED_CONFIDENCE, read_badge
from .rank import decide
from .segment import find_cards


def load_image(source: str | Path | bytes | bytearray | np.ndarray) -> np.ndarray:
    """Decode a spawn image from a path, raw bytes, or an existing array.

    Bytes matter: anything driving this live has the image in memory -- a
    downloaded attachment, a clipboard grab -- and making it round-trip
    through a temp file to be read is pure overhead.
    """
    if isinstance(source, np.ndarray):
        if source.size == 0:
            raise ValueError("empty image array")
        return source
    if isinstance(source, (bytes, bytearray)):
        img = cv2.imdecode(np.frombuffer(bytes(source), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("could not decode image from bytes")
        return img
    img = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read image: {source}")
    return img


def analyze_cards_with_boxes(
    bgr: np.ndarray,
    expected: int | None = None,
    layout: str = "auto",
    read_names: bool = True,
    fast: bool = False,
) -> tuple[list[Card], list[tuple[int, int, int, int]]]:
    """Same as :func:`analyze_cards` but also returns the crop boxes.

    Callers that need the pixels back -- the calibration extractor, mainly --
    should not have to re-run segmentation to get them.

    `read_names` is accepted but does nothing. The name reader was
    abandoned (see the README's "Name reader" section), so `Card.character`
    /`series`/`name_confidence` are always left at their dataclass defaults.
    Kept on the signature rather than removed so existing callers don't need
    editing; the CLI's `--no-names` flag, which did advertise a real
    behaviour, was removed instead.
    """
    boxes = find_cards(bgr, expected, layout)
    cards: list[Card] = []
    for i, (x, y, w, h) in enumerate(boxes, start=1):
        crop = bgr[y:y + h, x:x + w]
        # The digit atlas reads the game's own badge font. When it is sure,
        # that is the answer and tesseract is not consulted at all -- it costs
        # about a second a card and has nothing to add to a clean read. Only
        # an unsure atlas (an unfamiliar font, or an "E", which has no digits
        # to match) falls through to it.
        pixel_tier, feats = guess_frame(crop)
        atlas_no, atlas_conf = read_print_number(crop)
        trusted = atlas_no is not None and atlas_conf >= MIN_TRUSTED_MATCH
        if trusted:
            badge = {"print_no": atlas_no, "no_number": False,
                     "confidence": atlas_conf, "text": str(atlas_no)}
        elif fast and pixel_tier is FrameTier.E:
            # The border has already settled this card: resolve_frame returns
            # E whatever the badge says, so tesseract cannot change the
            # outcome -- and it costs ~1.3s a card, which is the entire
            # latency budget for anything reacting to a live spawn.
            #
            # Opt-in, because it does cost something real: with no tesseract
            # read there is no second opinion to contradict the border, so
            # the frame_disagrees review flag goes quiet on E cards. Offline
            # analysis wants that flag and does not care about the second;
            # anything live is the other way round.
            badge = {"print_no": None, "no_number": True,
                     "confidence": atlas_conf, "text": ""}
        else:
            badge = read_badge(crop)
        feats = dict(
            feats,
            pixel_frame=pixel_tier.value,
            badge_frame=(FrameTier.NORMAL.value if trusted
                         else frame_from_badge(badge["print_no"], badge["no_number"]).value),
        )
        # The border decides whether this card carries a print number at all;
        # the badge is only asked for the digits. resolve_frame explains why
        # it goes that way round.
        tier, print_no, no_number = resolve_frame(
            badge["print_no"], badge["no_number"], pixel_tier.value
        )
        cards.append(
            Card(
                slot=i,
                print_no=print_no,
                no_number=no_number,
                frame=tier,
                ocr_text=badge["text"],
                ocr_confidence=round(badge["confidence"], 3),
                print_trusted=badge["confidence"] >= MIN_TRUSTED_CONFIDENCE,
                frame_features=feats,
            )
        )
    return cards, boxes


def analyze_cards(
    bgr: np.ndarray,
    expected: int | None = None,
    layout: str = "auto",
    read_names: bool = True,
    fast: bool = False,
) -> list[Card]:
    return analyze_cards_with_boxes(bgr, expected, layout, read_names, fast)[0]


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
