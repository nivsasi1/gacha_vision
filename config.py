"""Tunable policy. Every number a human might want to argue about lives here.

Nothing in this file is magic -- the defaults encode the stated policy
("lower print is better, E is bad, two cards under 20 means take both") and
are meant to be recalibrated once real spawns have been scored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

from .models import FrameTier


@dataclass(frozen=True)
class Policy:
    # --- component weights (should sum to 1.0) ---
    w_print: float = 0.50
    w_frame: float = 0.30
    w_fame: float = 0.20

    # --- print number scoring ---
    # score = print_base - print_decay * log10(print_no), clamped.
    print_base: float = 100.0
    print_decay: float = 24.0
    # Floor for absurdly high prints. Must stay ABOVE score_no_number so that
    # "E is bad" holds strictly: an E card ranks below *any* numbered card.
    print_floor: float = 12.0
    # A card showing "E" (no print number at all).
    score_no_number: float = 6.0
    # OCR could not read the badge: neutral-low, and flagged for review.
    score_unreadable: float = 25.0

    # --- frame scoring ---
    frame_scores: dict[str, float] = field(
        default_factory=lambda: {
            FrameTier.UNKNOWN.value: 40.0,
            FrameTier.COMMON.value: 25.0,
            FrameTier.UNCOMMON.value: 50.0,
            FrameTier.RARE.value: 75.0,
            FrameTier.HOLO.value: 95.0,
        }
    )

    # --- fame scoring ---
    # Characters absent from the watchlist get this neutral baseline rather
    # than 0, so an unlisted character is "unknown", not "worthless".
    fame_default: float = 30.0

    # --- decision thresholds ---
    # Stated rule: if two cards are both under this print, spend the extra pick.
    take_both_max_print: int = 20
    # Secondary rule: two cards this good by overall score also justify it
    # (e.g. two holo frames of watchlist characters). Set to 101 to disable.
    take_both_min_score: float = 80.0
    # Below this, the spawn is not worth a claim at all.
    min_claim_score: float = 45.0
    # Always claim regardless of score at or below this print number.
    must_claim_print: int = 5
    # Always claim a character whose watchlist fame is at or above this.
    must_claim_fame: float = 90.0
    # Whether a fancy frame may lift a card that has NO print number.
    #
    # Off, because real Gachapon spawns showed the opposite of the assumption
    # this scorer started with: every "E" card wore the ornate gold/chain
    # frame with rainbow corners, while the numbered cards wore a plain thin
    # border. Frames there are cosmetic, so letting one lift an unnumbered
    # card lets decoration impersonate rarity -- and breaks the first rule,
    # that E ranks below any numbered card. Watchlist fame still rescues an
    # E card; only the frame is capped.
    frame_lifts_unnumbered: bool = False
    # How many cards a single spawn may be claimed from.
    max_claims: int = 2

    def frame_score(self, tier: FrameTier) -> float:
        return self.frame_scores.get(tier.value, self.frame_scores[FrameTier.UNKNOWN.value])

    def with_overrides(self, **kw) -> "Policy":
        return replace(self, **{k: v for k, v in kw.items() if v is not None})


def load_policy(path: str | Path | None) -> Policy:
    """Load a Policy from JSON, falling back to defaults for absent keys."""
    if not path:
        return Policy()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    known = {f for f in Policy.__dataclass_fields__}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown policy keys: {', '.join(sorted(unknown))}")
    return Policy(**data)


def load_watchlist(path: str | Path | None) -> dict[str, float]:
    """Load {normalised name -> fame score} from a watchlist JSON file.

    Accepted shapes:
        {"Bulma": 85, "Dragon Ball": 70}
        [{"name": "Bulma", "fame": 85}, ...]
    Series names live in the same flat map: a card matches on character
    first, then series, so listing a series lifts everyone in it.
    """
    if not path:
        return {}
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    if isinstance(raw, dict):
        items = raw.items()
    else:
        items = ((e["name"], e.get("fame", 75)) for e in raw)
    for name, fame in items:
        if str(name).startswith("_"):   # allow "_comment" keys in the file
            continue
        try:
            out[normalise(str(name))] = float(fame)
        except (TypeError, ValueError):
            raise ValueError(f"watchlist entry {name!r} has non-numeric fame: {fame!r}") from None
    return out


def normalise(s: str) -> str:
    """Casefold and strip punctuation so 'Ace of the Diamond!' matches 'ace of the diamond'."""
    return " ".join("".join(c for c in s.lower() if c.isalnum() or c.isspace()).split())
