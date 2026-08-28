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
Task 3: ABANDONED - per-glyph segmentation has a 55-62% oracle ceiling.
Font ID: NO FONT IDENTIFIED, and the calibration shows the metric cannot
  discriminate at card resolution (a correct font self-matches at 0.29-0.62
  after the game's outline+downscale; best candidate scored 0.597). Stage B
  struck from the design: rendered templates are worse than learned ones.
PIVOT (user approved): segmentation-free reading with forced alignment.
  Revised tasks R1-R4 appended to the plan.
Task R1: complete (see .superpowers/sdd/task-R1-report.md for full detail)
  - Built gacha_vision/names_align.py (normalise_line, forced_align, harvest,
    plus line_ink_bounds/match_score/mean_templates helpers) and
    tools/build_glyph_atlas.py (the EM loop: seed from 52 cards' clean
    split_line output -> align -> score-gated harvest -> re-template).
  - TDD caught a real gap the plan didn't anticipate: forced_align could
    return [] when segment_lines' line union is contaminated (e.g.
    35.png#1's line 0 fuses "Seras Victoria" + "HELLSING" + border chain
    links into one 265px-tall box) and the known text's natural template
    widths don't fit. Fixed with a proportional shrink-to-fit so alignment
    always returns a full placement, as the design requires; quality on
    contaminated lines is handled by the harvest score gate instead.
  - Atlas: 1553 glyphs, 63 classes (100% of the 63-character in-scope
    alphabet -- character names + single-line series names). Thin (<3):
    , - . / 7 ? L X f é. 7 more chars (+ 0 3 6 @ Q q) appear only in
    wrapped series text, out of scope by design (unchanged from the
    original Task 4 reasoning about un-guessable wrap points).
  - EM: mean alignment score 0.333 -> 0.380 over 5 iterations, stopped on
    <0.002 gain (see report for the full table).
  - Deleted the 2 already-failing abandoned-approach tests from
    test_names.py per the brief; the other 4 (not 3 -- the brief's count
    predates 2 tests Task 2's own review added) still pass.
  - Full suite: 1 failed (pre-existing, unrelated), 143 passed. No other
    test changed status.
