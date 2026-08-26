"""Command line entry point.

    python -m gacha_vision analyze  shot.png [--watchlist wl.json] [--json]
    python -m gacha_vision demo     [--out DIR]
    python -m gacha_vision calibrate shot.png [--expected 2]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analyze import analyze_cards, analyze_spawn, load_image
from .config import Policy, load_policy, load_watchlist
from .models import FrameTier
from .rank import decide
from .synth import draw_spawn


def _print_cards(cards) -> None:
    for c in cards:
        conf = f"{c.ocr_confidence:.0%}" if c.ocr_confidence else "--"
        print(
            f"  slot {c.slot}: {c.label():<44} "
            f"ocr={conf:<5} rarity_idx={c.frame_features.get('rarity_index', 0):.3f}"
        )
        if c.ocr_confidence and c.ocr_confidence < 0.55:
            print(f"    ! low OCR confidence (raw: {c.ocr_text!r}) -- verify before trusting")


def cmd_analyze(a: argparse.Namespace) -> int:
    policy = load_policy(a.policy)
    cards, decision = analyze_spawn(
        a.image,
        policy=policy,
        watchlist_path=a.watchlist,
        expected=a.expected,
        layout=a.layout,
        read_names=not a.no_names,
    )
    if a.json:
        print(json.dumps(
            {"cards": [
                {"slot": c.slot, "print_no": c.print_no, "no_number": c.no_number,
                 "frame": c.frame.value, "character": c.character, "series": c.series,
                 "ocr_confidence": c.ocr_confidence, "frame_features": c.frame_features}
                for c in cards
            ], "decision": decision.to_dict()},
            indent=2,
        ))
        return 0
    print(f"\n{Path(a.image).name} -- {len(cards)} card(s)")
    _print_cards(cards)
    print()
    print(decision.explain())
    print()
    return 0


def cmd_calibrate(a: argparse.Namespace) -> int:
    """Dump raw frame features so THRESHOLDS can be tuned on real spawns."""
    cards = analyze_cards(load_image(a.image), a.expected, a.layout, read_names=False)
    print(f"\n{Path(a.image).name} -- frame features (tune frame.THRESHOLDS from these)\n")
    hdr = f"{'slot':<5}{'tier':<10}{'rarity':<9}{'entropy':<9}{'divers':<8}{'sat':<8}{'colored':<9}"
    print(hdr)
    print("-" * len(hdr))
    for c in cards:
        f = c.frame_features
        print(f"{c.slot:<5}{c.frame.value:<10}{f['rarity_index']:<9.3f}"
              f"{f['hue_entropy']:<9.3f}{f['hue_diversity']:<8.3f}{f['sat_mean']:<8.1f}"
              f"{f['colored_frac']:<9.3f}")
    print()
    return 0


def cmd_demo(a: argparse.Namespace) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scenarios = {
        "low_print_vs_junk": [
            dict(tier=FrameTier.RARE, badge="14", character="BULMA", series="DRAGON BALL", art_hue=95),
            dict(tier=FrameTier.COMMON, badge="852", character="YAEKA", series="YAKUZA GUIDE", art_hue=5),
        ],
        "both_under_20": [
            dict(tier=FrameTier.UNCOMMON, badge="7", character="HIRO", series="FRANXX", art_hue=120),
            dict(tier=FrameTier.RARE, badge="12", character="EIJUN", series="ACE DIAMOND", art_hue=20),
        ],
        "both_bad": [
            dict(tier=FrameTier.COMMON, badge="E", character="HIRO", series="FRANXX", art_hue=130),
            dict(tier=FrameTier.COMMON, badge="1584", character="EIJUN", series="ACE DIAMOND", art_hue=15),
        ],
        "holo_carry": [
            dict(tier=FrameTier.HOLO, badge="430", character="RARE ONE", series="SHOW", art_hue=60),
            dict(tier=FrameTier.COMMON, badge="900", character="FILLER", series="SHOW", art_hue=10),
        ],
    }
    policy = load_policy(a.policy)
    watchlist = load_watchlist(a.watchlist)
    import cv2

    for name, specs in scenarios.items():
        path = out / f"{name}.png"
        cv2.imwrite(str(path), draw_spawn(specs))
        cards = analyze_cards(load_image(path), expected=len(specs), layout="auto")
        d = decide(cards, policy, watchlist)
        print(f"\n=== {name} ===")
        _print_cards(cards)
        print(d.explain())
    print(f"\nimages written to {out}/\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gacha_vision", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("image")
        sp.add_argument("--expected", type=int, default=None,
                        help="number of cards in the spawn (improves segmentation)")
        sp.add_argument("--layout", choices=["auto", "columns"], default="auto")

    a = sub.add_parser("analyze", help="score a spawn screenshot and print the decision")
    common(a)
    a.add_argument("--watchlist", default=None, help="JSON of favoured characters/series")
    a.add_argument("--policy", default=None, help="JSON policy overrides")
    a.add_argument("--json", action="store_true")
    a.add_argument("--no-names", action="store_true", help="skip name OCR (faster)")
    a.set_defaults(func=cmd_analyze)

    c = sub.add_parser("calibrate", help="dump frame features for threshold tuning")
    common(c)
    c.set_defaults(func=cmd_calibrate)

    d = sub.add_parser("demo", help="render synthetic spawns and score them")
    d.add_argument("--out", default="gacha_vision/samples")
    d.add_argument("--watchlist", default=None)
    d.add_argument("--policy", default=None)
    d.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
