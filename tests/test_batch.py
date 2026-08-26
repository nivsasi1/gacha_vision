"""Tests for folder batching, segmentation strategies and threshold fitting."""

from __future__ import annotations

import csv

import cv2
import pytest

from gacha_vision.batch import (CSV_FIELDS, analyze_one, analyze_folder,
                                iter_images, summarize, write_csv)
from gacha_vision.calibrate import (best_split, build_sheet, fit_thresholds,
                                    ocr_report, read_csv)
from gacha_vision.config import Policy
from gacha_vision.models import FrameTier
from gacha_vision.segment import find_cards, projection_split
from gacha_vision.synth import draw_spawn

P = Policy()


@pytest.fixture
def corpus(tmp_path):
    """A small folder of spawns with known card counts."""
    specs = {
        "a": [dict(tier=FrameTier.NORMAL, badge="852"), dict(tier=FrameTier.NORMAL, badge="14")],
        "b": [dict(tier=FrameTier.E, badge="7"), dict(tier=FrameTier.OTHER, badge="12"),
              dict(tier=FrameTier.NORMAL, badge="E")],
        "c": [dict(tier=FrameTier.NORMAL, badge="430"), dict(tier=FrameTier.NORMAL, badge="9999")],
    }
    for name, sp in specs.items():
        cv2.imwrite(str(tmp_path / f"{name}.png"), draw_spawn(sp))
    return tmp_path, specs


# --- segmentation --------------------------------------------------------

@pytest.mark.parametrize("n", [2, 3, 4])
def test_projection_split_finds_every_card(n):
    img = draw_spawn([dict(tier=FrameTier.NORMAL, badge=str(100 + i)) for i in range(n)])
    assert len(projection_split(img)) == n


def test_projection_handles_plain_low_contrast_frames():
    """Regression: a pale NORMAL frame yields no closed contour, so the old
    contour-only path collapsed a two-card spawn into one box."""
    img = draw_spawn([
        dict(tier=FrameTier.NORMAL, badge="900"),
        dict(tier=FrameTier.NORMAL, badge="E"),
    ])
    assert len(find_cards(img, None, "auto")) == 2


def test_projection_returns_nothing_on_a_flat_image():
    import numpy as np
    assert projection_split(np.full((100, 300, 3), 30, dtype=np.uint8)) == []


def test_boxes_are_ordered_left_to_right():
    img = draw_spawn([dict(tier=FrameTier.NORMAL, badge=str(i)) for i in (11, 22, 33)])
    xs = [b[0] for b in find_cards(img, None, "auto")]
    assert xs == sorted(xs)


# --- batch ---------------------------------------------------------------

def test_iter_images_finds_only_images(tmp_path):
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("no")
    assert [p.name for p in iter_images(tmp_path)] == ["a.png"]


def test_analyze_one_returns_a_row_per_card(corpus):
    root, specs = corpus
    res = analyze_one(root / "b.png", P, {}, read_names=False)
    assert not res.error
    assert len(res.rows) == len(specs["b"])
    assert {r["slot"] for r in res.rows} == {1, 2, 3}


def test_analyze_one_survives_a_bad_file(tmp_path):
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"not an image")
    res = analyze_one(bad, P, {}, read_names=False)
    assert res.error and res.rows == []


def test_folder_run_covers_every_image(corpus):
    root, specs = corpus
    results = analyze_folder(root, P, {}, read_names=False)
    assert len(results) == len(specs)
    assert sum(len(r.rows) for r in results) == sum(len(v) for v in specs.values())


def test_csv_roundtrip_has_every_column(corpus, tmp_path):
    root, _ = corpus
    results = analyze_folder(root, P, {}, read_names=False)
    out = tmp_path / "report.csv"
    n = write_csv(results, out)
    rows = read_csv(out)
    assert len(rows) == n
    assert set(rows[0]) == set(CSV_FIELDS)
    assert rows[0]["true_print"] == "" and rows[0]["true_frame"] == ""


def test_summary_counts_add_up(corpus):
    root, specs = corpus
    s = summarize(analyze_folder(root, P, {}, read_names=False))
    assert s["images"] == len(specs) and s["failed"] == 0
    assert s["cards"] == sum(len(v) for v in specs.values())
    assert sum(s["actions"].values()) == len(specs)


def test_crops_are_written(corpus, tmp_path):
    root, specs = corpus
    crops = tmp_path / "crops"
    analyze_folder(root, P, {}, read_names=False, crop_dir=crops)
    assert len(list(crops.glob("*.png"))) == sum(len(v) for v in specs.values())


# --- fitting -------------------------------------------------------------

def test_best_split_separates_two_clean_groups():
    t, acc = best_split([0.1, 0.2, 0.15], [0.7, 0.8, 0.75])
    assert 0.2 < t < 0.7 and acc == 1.0


def test_best_split_handles_an_empty_side():
    assert best_split([], [0.5]) == (0.0, 0.0)


def test_fit_recovers_thresholds_from_labels():
    rows = []
    for tier, vals in [("normal", [0.10, 0.15]), ("other", [0.45, 0.50]),
                       ("e", [0.85, 0.90])]:
        rows += [{"ornateness": v, "true_frame": tier} for v in vals]
    fit = fit_thresholds(rows)
    assert fit["overall_accuracy"] == 1.0
    t = fit["thresholds"]
    assert t["other"] < t["e"]


def test_fit_reports_when_labels_are_missing():
    fit = fit_thresholds([{"ornateness": 0.5, "true_frame": ""}])
    assert fit["labelled"] == 0 and fit["thresholds"] == {}


def test_ocr_report_flags_the_costly_errors():
    rows = [
        {"print_no": "", "no_number": "1", "true_print": "14", "ocr_conf": "0.3"},   # number -> E
        {"print_no": "77", "no_number": "0", "true_print": "E", "ocr_conf": "0.4"},  # E -> number
        {"print_no": "14", "no_number": "0", "true_print": "14", "ocr_conf": "0.9"}, # correct
    ]
    r = ocr_report(rows)
    assert r["checked"] == 3 and r["correct"] == 1
    assert r["number_read_as_E"] == 1 and r["E_read_as_number"] == 1
    assert r["mean_conf_correct"] > r["mean_conf_wrong"]


def test_ocr_report_ignores_unlabelled_rows():
    assert ocr_report([{"print_no": "14", "no_number": "0", "true_print": ""}])["checked"] == 0


def test_sheet_renders_one_card_per_row(tmp_path):
    rows = [{f: "" for f in CSV_FIELDS} | {"image": "a.png", "slot": 1, "print_no": "14",
                                           "no_number": "0", "frame": "rare",
                                           "ornateness": "0.7", "ocr_conf": "0.9"}]
    out = tmp_path / "sheet.html"
    assert build_sheet(rows, out, CSV_FIELDS) == 1
    html = out.read_text()
    assert html.count('class="card"') == 1
    assert 'class="tf"' in html and 'class="tp"' in html
    assert "labels.csv" in html
