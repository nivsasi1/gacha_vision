# tools/build_name_fixture.py
"""Cut the name band out of every corpus card and save it as a test fixture."""
import csv, sys
import cv2, numpy as np

CROPS = "crops_v2"
BAND_TOP = 0.60          # generous: the block sits higher on E frames
OUT_NPZ = "gacha_vision/tests/data/name_bands.npz"

def band(card_bgr):
    h = card_bgr.shape[0]
    return cv2.cvtColor(card_bgr[int(BAND_TOP * h):, :], cv2.COLOR_BGR2GRAY)

def main():
    rows = list(csv.DictReader(open("gacha_vision/tests/data/real_labels.csv", encoding="utf-8")))
    out = {}
    for r in rows:
        img = cv2.imread(f"{CROPS}/{r['image'][:-4]}__slot{r['slot']}.png")
        if img is None:
            sys.exit(f"missing crop for {r['image']} slot {r['slot']}")
        out[f"{r['image']}#{r['slot']}"] = band(img)
    np.savez_compressed(OUT_NPZ, **out)
    print(f"wrote {len(out)} name bands")

if __name__ == "__main__":
    main()
