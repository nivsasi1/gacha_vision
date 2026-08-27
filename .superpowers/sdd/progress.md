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
Task 2: complete (commits 1c654a4..7e8310d, review clean after fix)
  - Important (resolved): both plan-supplied tests passed on pure noise --
    reviewer proved it with an 18-fake-box stub. Added
    test_top_line_component_count_tracks_the_character_name, which binds the
    top line's component count to the label; it fails 176/181 under the same
    stub and passes on the real implementation.
  - Implementer replaced the plan's absolute gray thresholds with per-card
    percentile-relative ones; the absolute values segmented noise.
  - Minor, for final review: reviewer notes percentile thresholds could
    misfire on a card whose background art is uniformly bright; no such card
    in the corpus.
  - Minor, for final review: no assertion targets _split_fused directly.
