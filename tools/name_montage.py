# tools/name_montage.py
"""Tile the name band of unlabelled cards into montage sheets for hand-reading."""
import csv, os
import cv2, numpy as np

CROPS = "crops_v2"
BAND_TOP = 0.55           # same generous crop as build_name_fixture.py
OUT_DIR = "tools/name_sheets"
PER_SHEET = 10
SCALE = 3
CAPTION_H = 22

def band(card_bgr):
    h = card_bgr.shape[0]
    return card_bgr[int(BAND_TOP * h):, :]

def make_tile(card_key, band_bgr):
    h, w = band_bgr.shape[:2]
    big = cv2.resize(band_bgr, (w * SCALE, h * SCALE), interpolation=cv2.INTER_NEAREST)
    tile = np.zeros((big.shape[0] + CAPTION_H, big.shape[1], 3), dtype=np.uint8)
    tile[CAPTION_H:, :] = big
    cv2.putText(tile, card_key, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return tile

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = list(csv.DictReader(open("gacha_vision/tests/data/real_labels.csv", encoding="utf-8")))
    todo = [r for r in rows if not r["true_character"].strip()]
    print(f"{len(todo)} cards need hand-reading")

    tiles = []
    for r in todo:
        card_key = f"{r['image']}#{r['slot']}"
        img = cv2.imread(f"{CROPS}/{r['image'][:-4]}__slot{r['slot']}.png")
        if img is None:
            raise SystemExit(f"missing crop for {card_key}")
        tiles.append((card_key, make_tile(card_key, band(img))))

    for i in range(0, len(tiles), PER_SHEET):
        chunk = tiles[i:i + PER_SHEET]
        width = max(t.shape[1] for _, t in chunk)
        total_h = sum(t.shape[0] for _, t in chunk) + 4 * (len(chunk) - 1)
        sheet = np.full((total_h, width, 3), 40, dtype=np.uint8)
        y = 0
        for _, t in chunk:
            sheet[y:y + t.shape[0], 0:t.shape[1]] = t
            y += t.shape[0] + 4
        out_path = f"{OUT_DIR}/sheet_{i // PER_SHEET:03d}.png"
        cv2.imwrite(out_path, sheet)
    print(f"wrote {(len(tiles) + PER_SHEET - 1) // PER_SHEET} sheets to {OUT_DIR}")

if __name__ == "__main__":
    main()
