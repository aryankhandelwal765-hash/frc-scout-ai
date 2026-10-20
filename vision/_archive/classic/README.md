# Archived: v1 classic-CV tracker

Superseded by the YOLO + ByteTrack pipeline in `vision/match_pipeline/`. Kept verbatim for reference;
not maintained, not linted, not imported by anything live. Run history is in git.

`scoring_estimator.py` (the one live piece) now lives in `vision/match_pipeline/`.
`test_e2e_classic.py` runs against `track_match.py` and `make_test_clip.py` from this folder and is no
longer collected by pytest.
