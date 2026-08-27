# Name reader — progress

Plan: docs/superpowers/plans/2026-08-27-name-reader.md
Branch: feat/name-reader

Task 1: complete (commits 0165ea4..b0a4bcd, review clean after fixes)
  - Critical (resolved): 59 legacy labels were never pixel-verified; one was
    wrong (cards18 `Pokemon` -> `Pokémon`). All 59 have now been checked
    against the card images by hand. 58 were correct.
  - Fixed: 24MB of labelling montages had been committed; now gitignored.
  - Minor, for final review: BAND_TOP differs between build_name_fixture.py
    (0.60) and name_montage.py (0.55) with no shared constant.
  - Minor, for final review: no test asserts npz/csv key correspondence.
  - Data: 4745 glyphs, 70 classes. Thin (<3): + , / 3 6 7 ? @ Q X q
  - cards33.png#1 has no name text on the card; left blank deliberately.
