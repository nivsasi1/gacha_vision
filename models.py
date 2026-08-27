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
    """The frame a card wears, using the game's own vocabulary.

    This is NOT a rarity ladder. An earlier version invented one
    (common/uncommon/rare/holo) and read it off the border's colour, which
    was wrong twice over: the ornate rainbow border belongs to ``E``, the
    frame that carries *no* print number, while the plain border belongs to
    ``NORMAL``, the one that does. Both are the game's common frames, so
    decoration says nothing about value -- the number does.

    The two known frames are distinguished by whether a print number exists,
    which makes the badge look authoritative. Measured against real cards it
    is not: the border decides the frame and the badge supplies the digits.
    See :func:`gacha_vision.frame.resolve_frame`.
    """

    UNKNOWN = "unknown"   # badge unreadable, so the frame is undetermined
    NORMAL = "normal"     # plain frame; carries the print number ("rating")
    E = "e"               # ornate gold/chain frame; carries no number
    OTHER = "other"       # neither known frame -- possibly rarer, worth review

    @property
    def is_known(self) -> bool:
        return self in (FrameTier.NORMAL, FrameTier.E)


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
    # Tesseract's mean word confidence on the name block. Reported, never
    # acted on: the name text itself has never once been right (see
    # ocr.read_name), so this exists to make that visible on the sheet.
    name_confidence: float = 0.0
    # Whether the print number is solid enough to act on. The vision stage
    # decides this from OCR confidence; the ranker only reads it, so the
    # question "is this legible?" stays separate from "is this worth taking?".
    print_trusted: bool = True
    frame_features: dict[str, float] = field(default_factory=dict)

    @property
    def print_known(self) -> bool:
        return self.print_no is not None

    @property
    def frame_disagrees(self) -> bool:
        """True when the badge did not corroborate the frame the border set.

        The border is what decides the frame, so this is no longer a tie to
        break -- it is a review signal. It fires on exactly the cards whose
        badge was misread, which on the real corpus was every time it fired.
        """
        badge = self.frame_features.get("badge_frame")
        return bool(badge) and self.frame.is_known and badge != self.frame.value

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
