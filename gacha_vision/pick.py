"""One call in, slot numbers out.

The rest of this package returns rich objects so a human can see why. This is
the other audience: something automated that needs the answer as numbers and
nothing else. `pick` returns the slots to claim, left to right as they appear
under the spawn, or an empty list when nothing is worth taking.

It accepts the image however you already hold it -- a path, raw bytes from a
download, or a decoded array. Bytes are the point: anything driving this live
has the image in memory already.

There is deliberately no watchlist parameter. Watchlist matching needs the
character name, and this game's names are unreadable at 7px (see the README),
so `analyze` leaves them empty. Accepting a watchlist here would take an
argument and silently ignore it.
"""

from __future__ import annotations

from .analyze import analyze_cards, load_image
from .config import Policy
from .rank import decide


def pick(
    image,
    *,
    expected: int | None = None,
    policy: Policy | None = None,
    layout: str = "auto",
) -> list[int]:
    """Return the slot numbers worth claiming. Empty list means claim nothing.

    Slots are 1-based, left to right, matching the buttons under the spawn --
    getting that wrong is the one error that costs a real pick.

    ``image`` may be a path, ``bytes``, or a decoded BGR array.
    ``expected`` is the card count when you know it; leave it None to let
    segmentation decide. Raises on an image that cannot be decoded, so a
    broken download never looks like "nothing worth claiming".
    """
    cards = analyze_cards(load_image(image), expected, layout, read_names=False)
    if not cards:
        return []
    return list(decide(cards, policy or Policy(), {}).slots)
