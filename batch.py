"""Run the pipeline over a folder of screenshots and emit a calibration set.

The point of this module is the feedback loop. Thresholds in ``frame.py`` and
``config.py`` are educated guesses until they meet real spawns; this turns a
folder of screenshots into a CSV of measurements that can be inspected,
labelled and fitted.

The CSV is the deliverable, not the images: it is small, textual and carries
every feature the classifier used, so calibration never requires moving the
screenshots anywhere.
"""

from __future__ import annotations

import csv
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from .analyze import analyze_cards_with_boxes, load_image
from .config import Policy
from .ocr import MIN_TRUSTED_CONFIDENCE
from .rank import decide

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# A spawn with a card count outside this range probably means segmentation
# grabbed chat UI or missed a card -- worth a human glance before trusting it.
#
# Two and three are what real spawns have shown so far, but one is confirmed
# (a lone-card drop, which is always claimed) and four is reported as
# possible, so neither should be flagged as a segmentation failure. Five is:
# nothing has ever dropped that many, so it is far more likely to be a
# neighbouring message sliced up than a real spawn.
PLAUSIBLE_CARDS = (1, 4)
LOW_OCR = MIN_TRUSTED_CONFIDENCE

CSV_FIELDS = [
    "image", "n_cards", "slot",
    "print_no", "no_number", "ocr_conf", "ocr_text",
    "frame", "pixel_frame", "ornateness", "hue_entropy", "hue_diversity", "sat_mean", "colored_frac",
    "character", "series", "score_total", "action", "chosen_slots", "flags",
    # left blank for you to fill in when labelling:
    "true_print", "true_frame", "true_character", "true_series",
]


@dataclass
class ImageResult:
    path: Path
    rows: list[dict] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    error: str = ""


def iter_images(root: str | Path) -> list[Path]:
    root = Path(root)
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def _flags_for(cards) -> list[str]:
    flags = []
    if len(cards) < PLAUSIBLE_CARDS[0] or len(cards) > PLAUSIBLE_CARDS[1]:
        flags.append(f"odd_card_count={len(cards)}")
    if any(c.ocr_confidence < LOW_OCR for c in cards):
        flags.append("low_ocr")
    if any(c.print_no is None and not c.no_number for c in cards):
        flags.append("unreadable_badge")
    if any(c.frame_disagrees for c in cards):
        flags.append("frame_disagrees_with_badge")
    return flags


def analyze_one(
    path: str | Path,
    policy: Policy,
    watchlist: dict[str, float],
    expected: int | None = None,
    layout: str = "auto",
    read_names: bool = True,
    crop_dir: str | Path | None = None,
) -> ImageResult:
    path = Path(path)
    res = ImageResult(path=path)
    try:
        img = load_image(path)
        cards, boxes = analyze_cards_with_boxes(img, expected, layout, read_names)
        decision = decide(cards, policy, watchlist)
        res.flags = _flags_for(cards)
        by_slot = {s.slot: s for s in decision.scores}
        chosen = ",".join(map(str, decision.slots))

        for card, (x, y, w, h) in zip(cards, boxes):
            f = card.frame_features
            res.rows.append({
                "image": path.name,
                "n_cards": len(cards),
                "slot": card.slot,
                "print_no": "" if card.print_no is None else card.print_no,
                "no_number": int(card.no_number),
                "ocr_conf": card.ocr_confidence,
                "ocr_text": card.ocr_text[:40],
                "frame": card.frame.value,
                "pixel_frame": f.get("pixel_frame", ""),
                "ornateness": f.get("ornateness", 0.0),
                "hue_entropy": f.get("hue_entropy", 0.0),
                "hue_diversity": f.get("hue_diversity", 0.0),
                "sat_mean": f.get("sat_mean", 0.0),
                "colored_frac": f.get("colored_frac", 0.0),
                "character": card.character[:40],
                "series": card.series[:40],
                "score_total": by_slot[card.slot].total if card.slot in by_slot else "",
                "action": decision.action.value,
                "chosen_slots": chosen,
                "flags": "|".join(res.flags),
                "true_print": "",
                "true_frame": "",
            })
            if crop_dir:
                out = Path(crop_dir) / f"{path.stem}__slot{card.slot}.png"
                cv2.imwrite(str(out), img[y:y + h, x:x + w])
    except Exception as exc:                      # keep the batch going
        res.error = f"{type(exc).__name__}: {exc}"
    return res


def _worker(job):
    cv2.setNumThreads(0)                          # avoid oversubscribing cores
    return analyze_one(*job)


# Parallelism here is opt-in, and that is deliberate.
#
# "fork" deadlocks: OpenCV and Tesseract start threads on first use, and
# forking a process that already has them leaves the children sitting at 0%
# CPU forever. "spawn" avoids that but re-imports the main module in every
# worker -- and under ``python -m gacha_vision`` that module is __main__.py,
# which multiprocessing re-runs through runpy where its relative import
# cannot resolve, so the workers die or hang instead.
#
# Serial throughput is ~150ms/card, i.e. under a minute for a few hundred
# cards, so the safe default costs little. --workers N opts in and falls back
# to serial if the pool breaks rather than leaving the run wedged.
_MP_CONTEXT = "spawn"


def analyze_folder(
    root: str | Path,
    policy: Policy,
    watchlist: dict[str, float],
    expected: int | None = None,
    layout: str = "auto",
    read_names: bool = True,
    workers: int | None = None,
    crop_dir: str | Path | None = None,
    progress=None,
) -> list[ImageResult]:
    paths = iter_images(root)
    if not paths:
        return []
    if crop_dir:
        Path(crop_dir).mkdir(parents=True, exist_ok=True)

    jobs = [(p, policy, watchlist, expected, layout, read_names, crop_dir) for p in paths]
    workers = 1 if workers is None else max(1, min(workers, os.cpu_count() or 1))

    def serial() -> list[ImageResult]:
        out: list[ImageResult] = []
        for i, job in enumerate(jobs, 1):
            out.append(_worker(job))
            if progress:
                progress(i, len(jobs))
        return out

    if workers <= 1:
        return serial()

    results: list[ImageResult] = []
    try:
        ctx = mp.get_context(_MP_CONTEXT)
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            for i, r in enumerate(ex.map(_worker, jobs), 1):
                results.append(r)
                if progress:
                    progress(i, len(jobs))
        return results
    except Exception as exc:
        print(f"  (parallel pool unavailable: {type(exc).__name__}; running serially)")
        return serial()


def write_csv(results: list[ImageResult], path: str | Path) -> int:
    rows = [r for res in results for r in res.rows]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def summarize(results: list[ImageResult]) -> dict:
    ok = [r for r in results if not r.error]
    cards = [row for r in ok for row in r.rows]
    actions: dict[str, int] = {}
    for r in ok:
        if r.rows:
            a = r.rows[0]["action"]
            actions[a] = actions.get(a, 0) + 1
    frames: dict[str, int] = {}
    for c in cards:
        frames[c["frame"]] = frames.get(c["frame"], 0) + 1

    n_low = sum(1 for c in cards if float(c["ocr_conf"] or 0) < LOW_OCR)
    n_e = sum(1 for c in cards if int(c["no_number"]))
    n_unread = sum(1 for c in cards if c["print_no"] == "" and not int(c["no_number"]))
    flagged = [r for r in ok if r.flags]
    return {
        "images": len(results),
        "failed": len(results) - len(ok),
        "cards": len(cards),
        "actions": actions,
        "frames": frames,
        "low_ocr_cards": n_low,
        "e_cards": n_e,
        "unreadable_cards": n_unread,
        "flagged_images": len(flagged),
        "errors": [(r.path.name, r.error) for r in results if r.error][:10],
    }
