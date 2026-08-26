"""gacha_vision -- read anime card spawns from a screenshot and rank them.

Standalone image-analysis + decision library. Given a screenshot of a card
spawn it reports, for each card, the print number, the frame rarity and the
character, then applies a configurable policy to say which card is worth
taking.

It has no network access, no game integration and no account handling: the
input is an image file, the output is a Decision. Wiring it to anything is
out of scope by design.
"""

from .analyze import analyze_cards, analyze_spawn, load_image
from .config import Policy, load_policy, load_watchlist
from .models import Action, Card, Decision, FrameTier, Score
from .rank import decide, score_card

__all__ = [
    "analyze_cards", "analyze_spawn", "load_image",
    "Policy", "load_policy", "load_watchlist",
    "Action", "Card", "Decision", "FrameTier", "Score",
    "decide", "score_card",
]
