"""Describe every catalogued frame, so an uncatalogued one can be spotted.

The corpus holds two frames, `E` and `NORMAL`. A card whose border matches
neither is the interesting case -- rare frames are worth claiming on sight --
and there is exactly one such card here, which is far too few to describe.
So this stores what the *known* frames look like and lets the reader ask how
far a new border sits from all of them.

The one uncatalogued card is deliberately excluded: including it would teach
the atlas that its frame is ordinary, which is the opposite of the point.
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gacha_vision.frame import ring_descriptor  # noqa: E402
from gacha_vision.models import FrameTier  # noqa: E402
from gacha_vision.synth import draw_card  # noqa: E402

CROPS = "crops_v2"
LABELS = "gacha_vision/tests/data/real_labels.csv"
OUT = "gacha_vision/data/frame_atlas.npz"

# Wears a frame that is neither E nor NORMAL -- it is what the reader must
# detect, so it cannot also be a reference for "normal".
UNCATALOGUED = {"cards30.png#1"}


def main() -> int:
    desc, labels, cards = [], [], []
    skipped = []
    for r in csv.DictReader(open(LABELS, encoding="utf-8")):
        key = f"{r['image']}#{r['slot']}"
        if key in UNCATALOGUED:
            skipped.append(key)
            continue
        img = cv2.imread(f"{CROPS}/{r['image'][:-4]}__slot{r['slot']}.png")
        if img is None:
            sys.exit(f"missing crop for {key}")
        d = ring_descriptor(img)
        if d is None:
            skipped.append(f"{key} (no colour in the border)")
            continue
        desc.append(d)
        labels.append(r["true_frame"])
        cards.append(key)
    # synth.py's E and NORMAL are catalogued too. They are those frames, drawn
    # by our fixture generator rather than the game -- and its border picks up
    # the artwork's hue, so a rendered card with an unusual art_hue reads as
    # an unfamiliar frame and claims on sight unless it is listed here. The
    # sweep covers the whole hue wheel in 5-degree steps for that reason --
    # 15 left gaps a rendered card could fall into.
    #
    # This costs nothing in detection: the uncatalogued card measures 0.1895
    # from the real frames and 0.1895 with these added, the nearest synthetic
    # one being 0.2469 away from it.
    for tier in (FrameTier.NORMAL, FrameTier.E):
        for badge in ("E", "14", "42", "852", "1600", "1655"):
            for hue in range(0, 180, 5):
                d = ring_descriptor(draw_card(tier=tier, badge=badge, art_hue=hue))
                if d is None:
                    continue
                desc.append(d)
                labels.append(tier.value)
                cards.append(f"synth:{tier.value}:{badge}:{hue}")

    np.savez_compressed(OUT, descriptors=np.stack(desc),
                        labels=np.array(labels), card=np.array(cards))
    print(f"{len(desc)} catalogued frames, excluded {skipped}")
    import collections
    print("per class:", dict(collections.Counter(labels)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
