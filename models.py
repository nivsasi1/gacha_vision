"""Core data types for card spawn analysis.

A *spawn* is one screenshot containing two or more *cards* side by side.
Every card carries a print number (lower = rarer), a frame whose styling
encodes rarity, and a character/series identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class FrameTier(str, Enum):
    """Border styling, ordered worst -> best.

    The names are deliberately generic: what a given bot calls its frames
    varies, but the visual progression (flat single colour -> saturated
    multi-hue -> full rainbow holo) is near universal.
    """

    UNKNOWN = "unknown"
    COMMON = "common"
    UNCOMMON = "uncommon"
    RARE = "rare"
    HOLO = "holo"

    @property
    def rank(self) -> int:
        return _FRAME_ORDER.index(self)


_FRAME_ORDER = [
    FrameTier.UNKNOWN,
    FrameTier.COMMON,
    FrameTier.UNCOMMON,
    FrameTier.RARE,
    FrameTier.HOLO,
]


class Action(str, Enum):
    CLAIM = "claim"
    CLAIM_BOTH = "claim_both"
    SKIP = "skip"


@dataclass
class Card:
    """One card inside a spawn.

    `slot` is the 1-based position, matching the numbered buttons under the
    spawn image. `print_no` is None when the card shows no number (the "E"
    case) or when OCR could not read it -- `no_number` distinguishes those.
    """

    slot: int
    print_no: int | None = None
    no_number: bool = False
    frame: FrameTier = FrameTier.UNKNOWN
    character: str = ""
    series: str = ""

    # Diagnostics from the vision stage; ignored by ranking.
    ocr_text: str = ""
    ocr_confidence: float = 0.0
    frame_features: dict[str, float] = field(default_factory=dict)

    @property
    def print_known(self) -> bool:
        return self.print_no is not None

    def label(self) -> str:
        who = self.character or f"card {self.slot}"
        if self.no_number:
            num = "E"
        elif self.print_no is not None:
            num = f"#{self.print_no}"
        else:
            num = "#?"
        return f"{who} ({num}, {self.frame.value})"


@dataclass
class Score:
    """Breakdown of one card's desirability, 0-100 per component."""

    slot: int
    total: float
    print_score: float
    frame_score: float
    fame_score: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class Decision:
    """What to do with a spawn, and why."""

    action: Action
    slots: list[int]
    scores: list[Score]
    reasons: list[str] = field(default_factory=list)

    def explain(self) -> str:
        head = {
            Action.CLAIM: f"CLAIM slot {self.slots[0]}" if self.slots else "CLAIM",
            Action.CLAIM_BOTH: f"CLAIM BOTH -> slots {', '.join(map(str, self.slots))}",
            Action.SKIP: "SKIP",
        }[self.action]
        lines = [head]
        lines.extend(f"  - {r}" for r in self.reasons)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "slots": self.slots,
            "reasons": self.reasons,
            "scores": [asdict(s) for s in self.scores],
        }
