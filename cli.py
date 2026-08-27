"""Command line entry point.

    python -m gacha_vision analyze  shot.png [--watchlist wl.json] [--json]
    python -m gacha_vision batch    shots/ [--out report.csv] [--workers 8]
    python -m gacha_vision extract  shots/ --out crops/      # + labelling sheet
    python -m gacha_vision fit      labels.csv               # tune thresholds
    python -m gacha_vision demo     [--out DIR]
    python -m gacha_vision calibrate shot.png [--expected 2]
    python -m gacha_vision doctor                             # check the setup
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analyze import analyze_cards, analyze_spawn, load_image
from .batch import CSV_FIELDS, analyze_folder, summarize, write_csv
from .calibrate import build_sheet, fit_thresholds, name_report, ocr_report, read_csv
from .config import Policy, load_policy, load_watchlist
from .models import FrameTier
from .rank import decide
from .synth import draw_spawn


def _print_cards(cards) -> None:
    for c in cards:
        conf = f"{c.ocr_confidence:.0%}" if c.ocr_confidence else "--"
        print(
            f"  slot {c.slot}: {c.label():<44} "
            f"ocr={conf:<5} sat={c.frame_features.get('sat_mean', 0):.0f}"
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
    print(f"\n{Path(a.image).name} -- border features (tune frame.E_ORNATENESS from these)\n")
    hdr = f"{'slot':<5}{'tier':<10}{'ornate':<9}{'entropy':<9}{'divers':<8}{'sat':<8}{'colored':<9}"
    print(hdr)
    print("-" * len(hdr))
    for c in cards:
        f = c.frame_features
        print(f"{c.slot:<5}{c.frame.value:<10}{f['ornateness']:<9.3f}"
              f"{f['hue_entropy']:<9.3f}{f['hue_diversity']:<8.3f}{f['sat_mean']:<8.1f}"
              f"{f['colored_frac']:<9.3f}")
    print()
    return 0


def cmd_demo(a: argparse.Namespace) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scenarios = {
        "low_print_vs_junk": [
            dict(tier=FrameTier.NORMAL, badge="14", character="BULMA", series="DRAGON BALL", art_hue=95),
            dict(tier=FrameTier.NORMAL, badge="852", character="YAEKA", series="YAKUZA GUIDE", art_hue=5),
        ],
        "both_under_20": [
            dict(tier=FrameTier.NORMAL, badge="7", character="HIRO", series="FRANXX", art_hue=120),
            dict(tier=FrameTier.NORMAL, badge="12", character="EIJUN", series="ACE DIAMOND", art_hue=20),
        ],
        "both_bad": [
            dict(tier=FrameTier.E, badge="E", character="HIRO", series="FRANXX", art_hue=130),
            dict(tier=FrameTier.NORMAL, badge="1584", character="EIJUN", series="ACE DIAMOND", art_hue=15),
        ],
        "ornate_but_junk": [
            dict(tier=FrameTier.E, badge="430", character="ORNATE ONE", series="SHOW", art_hue=60),
            dict(tier=FrameTier.NORMAL, badge="900", character="FILLER", series="SHOW", art_hue=10),
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


def cmd_doctor(a: argparse.Namespace) -> int:
    """Check that this machine can actually run the pipeline.

    Exists because setup problems surface on a machine nobody debugging them
    can see; one paste of this output names the broken piece.
    """
    import platform

    ok = True
    print(f"\npython      {sys.version.split()[0]}  ({platform.system()} {platform.machine()})")
    for mod in ("numpy", "cv2", "PIL", "pytesseract"):
        try:
            m = __import__(mod)
            print(f"{mod:<11} {getattr(m, '__version__', 'ok')}")
        except Exception as exc:
            ok = False
            print(f"{mod:<11} MISSING -- {exc}")
            print(f"            fix: pip install -r gacha_vision/requirements.txt")

    from .ocr import find_tesseract
    exe = find_tesseract()
    if exe:
        print(f"tesseract   {exe}")
    else:
        ok = False
        print("tesseract   NOT FOUND")
        print("            fix (Windows): winget install UB-Mannheim.TesseractOCR")
        print("            fix (Linux):   sudo apt install tesseract-ocr")
        print("            or set TESSERACT_CMD to the full path of tesseract.exe")

    if ok:
        # End-to-end self test on a card we draw here, so a pass means the
        # whole chain works, not just that the imports resolved.
        from .synth import draw_card
        from .ocr import read_badge
        from .frame import guess_frame
        r = read_badge(draw_card(badge="1655"))
        g = guess_frame(draw_card(tier=FrameTier.E, badge="E"))[0]
        print(f"\nself-test   badge 1655 -> {r['print_no']}  (conf {r['confidence']:.0%})")
        print(f"            ornate border -> {g.value}")
        if r["print_no"] != 1655:
            ok = False
            print("            ! OCR self-test FAILED -- paste this output for help")

    print("\n" + ("all good -- run: python -m gacha_vision extract <folder> --out crops"
                  if ok else "setup incomplete; fix the lines above"))
    return 0 if ok else 1


def _progress(i, n):
    if i == n or i % 10 == 0:
        print(f"  ...{i}/{n}", flush=True)


def _run_batch(a, crop_dir=None):
    policy = load_policy(a.policy)
    watchlist = load_watchlist(getattr(a, "watchlist", None))
    print(f"scanning {a.folder} ...")
    results = analyze_folder(
        a.folder, policy, watchlist,
        expected=a.expected, layout=a.layout,
        read_names=not a.no_names, workers=a.workers,
        crop_dir=crop_dir, progress=_progress,
    )
    if not results:
        print(f"no images found under {a.folder}")
        return None, None
    return results, summarize(results)


def _print_summary(s):
    print(f"\nimages: {s['images']}   cards: {s['cards']}   failed: {s['failed']}")
    if s["actions"]:
        print("decisions: " + "  ".join(f"{k}={v}" for k, v in sorted(s["actions"].items())))
    if s["frames"]:
        print("frames:    " + "  ".join(f"{k}={v}" for k, v in sorted(s["frames"].items())))
    print(f"E cards: {s['e_cards']}   unreadable: {s['unreadable_cards']}   "
          f"low-confidence: {s['low_ocr_cards']}")
    if s["flagged_images"]:
        print(f"flagged for review: {s['flagged_images']} image(s)")
    for name, err in s["errors"]:
        print(f"  ERROR {name}: {err}")


def cmd_batch(a: argparse.Namespace) -> int:
    results, s = _run_batch(a)
    if not results:
        return 1
    n = write_csv(results, a.out)
    _print_summary(s)
    print(f"\nwrote {n} card rows -> {a.out}")
    print("send that CSV over to calibrate thresholds; the images can stay put.")
    return 0


def cmd_extract(a: argparse.Namespace) -> int:
    out = Path(a.out)
    results, s = _run_batch(a, crop_dir=out)
    if not results:
        return 1
    csv_path = out / "manifest.csv"
    n = write_csv(results, csv_path)
    rows = [r for res in results for r in res.rows]
    sheet = out / "sheet.html"
    build_sheet(rows, sheet, CSV_FIELDS)
    _print_summary(s)
    print(f"\n{n} crops + manifest.csv -> {out}/")
    print(f"open {sheet} in a browser, fix the labels, hit Download labels.csv")
    print("then: python -m gacha_vision fit labels.csv")
    return 0


def cmd_fit(a: argparse.Namespace) -> int:
    rows = read_csv(a.csv)
    print(f"\n{len(rows)} rows from {a.csv}")

    fit = fit_thresholds(rows, feature=a.feature)
    print(f"\n-- frame thresholds ({fit['feature']}) --")
    print("labelled: " + "  ".join(f"{k}={v}" for k, v in fit["counts"].items()))
    if fit["labelled"] < 4:
        print("  not enough labelled rows; fill in true_frame and re-run")
    else:
        for sp in fit["splits"]:
            if sp.get("threshold") is None:
                print(f"  {sp['boundary']:<20} -- {sp.get('note','')}")
            else:
                print(f"  {sp['boundary']:<20} cut={sp['threshold']:<8} "
                      f"acc={sp['accuracy']:.0%}  (n={sp['n']})")
        print(f"  overall: {fit.get('overall_accuracy', 0):.0%}")
        cut = fit["thresholds"].get(FrameTier.E.value)
        if cut is not None and a.feature == "sat_mean":
            print(f"\n  paste into gacha_vision/frame.py:\n\n      E_SATURATION = {cut}")
        elif cut is not None:
            print(f"\n  cut on {a.feature} is {cut}; frame.py splits on sat_mean, so "
                  f"swap guess_frame's feature too before pasting it")

    o = ocr_report(rows)
    print(f"\n-- badge OCR --")
    if not o["checked"]:
        print("  no true_print labels found; fill them in to measure OCR")
    else:
        print(f"  {o['correct']}/{o['checked']} = {o['accuracy']:.0%}")
        print(f"  number read as E: {o['number_read_as_E']}   "
              f"E read as number: {o['E_read_as_number']}   (these are the costly ones)")
        print(f"  mean confidence: correct={o['mean_conf_correct']}  wrong={o['mean_conf_wrong']}")
        if o["misses"]:
            print("\n  misses:")
            for m in o["misses"]:
                print(f"    {m['image']} slot{m['slot']}: want {m['want']:<6} got {m['got']:<6} "
                      f"conf={m['conf']:.2f} raw={m['raw']!r}")

    n = name_report(rows)
    print(f"\n-- character / series OCR --")
    for field in ("character", "series"):
        d = n[field]
        line = f"  {field:<10} read something on {d['read_something']}/{n['rows']}"
        if d["coverage"] is not None:
            line += f" = {d['coverage']:.0%}"
        print(line)
        if d["labelled"]:
            print(f"             vs labels: {d['exact']} exact + {d['close']} close "
                  f"of {d['labelled']} = {d['accuracy']:.0%}")
            for m in d["misses"]:
                print(f"               want {m['want']!r} got {m['got']!r}")
        else:
            print("             no true_%s labels; fill some in on the sheet to measure" % field)
    print()
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

    def folder_args(sp):
        sp.add_argument("folder", help="directory of screenshots (searched recursively)")
        sp.add_argument("--expected", type=int, default=None)
        sp.add_argument("--layout", choices=["auto", "columns"], default="auto")
        sp.add_argument("--watchlist", default=None)
        sp.add_argument("--policy", default=None)
        sp.add_argument("--workers", type=int, default=None,
                        help="parallel processes (default 1; >1 falls back to "
                             "serial if the pool cannot start)")
        sp.add_argument("--no-names", action="store_true")

    b = sub.add_parser("batch", help="score a whole folder into a CSV")
    folder_args(b)
    b.add_argument("--out", default="report.csv")
    b.set_defaults(func=cmd_batch)

    e = sub.add_parser("extract", help="crop cards + build a labelling sheet")
    folder_args(e)
    e.add_argument("--out", default="crops")
    e.set_defaults(func=cmd_extract)

    f = sub.add_parser("fit", help="fit thresholds from a labelled CSV")
    f.add_argument("csv")
    f.add_argument("--feature", default="sat_mean",
                   help="ring measurement to fit the frame cut on; sat_mean separated "
                        "182 labelled cards at 99%% and is what frame.py ships")
    f.set_defaults(func=cmd_fit)

    doc = sub.add_parser("doctor", help="check python, deps and tesseract on this machine")
    doc.set_defaults(func=cmd_doctor)

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
