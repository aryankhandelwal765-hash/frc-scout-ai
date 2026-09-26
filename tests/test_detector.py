"""Detector-dependent checks.

Everything here needs trained YOLO weights on disk, which are gitignored (they
are hundreds of MB). Each test is skipped when they are absent, so a fresh
clone and CI stay green without them.
"""

import pytest

from tests.conftest import needs_weights

pytest.importorskip("cv2", reason="the detector path needs opencv")


@needs_weights
def test_newest_weights_resolves_to_a_file():
    pytest.importorskip("ultralytics", reason="ultralytics not installed")
    import trajectories

    path = trajectories._newest_weights()
    assert path.endswith("best.pt")

    from pathlib import Path

    assert Path(path).is_file()


@needs_weights
def test_bytetrack_config_is_present_and_parsable():
    """ByteTrack's tracker config is tracked config, not a generated artefact."""
    yaml = pytest.importorskip("yaml", reason="pyyaml not installed")
    from pathlib import Path

    cfg = Path(__file__).resolve().parent.parent / "vision" / "bytetrack_frc.yaml"
    assert cfg.exists()
    data = yaml.safe_load(cfg.read_text())
    assert data.get("tracker_type") == "bytetrack"
    # track_buffer is the occlusion-tolerance knob the pipeline depends on.
    assert int(data.get("track_buffer", 0)) > 0


def test_crop_is_shared_not_copy_pasted():
    """Regression guard for phase 4: CROP lives in boxes.py only."""
    import boxes
    import trajectories

    assert trajectories.CROP is boxes.CROP


def test_boxes_can_be_overridden_from_the_environment(monkeypatch):
    import importlib

    import boxes

    monkeypatch.setenv("VISION_CROP", "5,6,7,8")
    reloaded = importlib.reload(boxes)
    try:
        assert reloaded.CROP == (5, 6, 7, 8)
    finally:
        monkeypatch.delenv("VISION_CROP", raising=False)
        importlib.reload(boxes)


def test_malformed_box_override_falls_back(monkeypatch):
    """A bad env var must not take down a batch run."""
    import importlib

    import boxes

    monkeypatch.setenv("VISION_CROP", "not,numbers,at,all")
    reloaded = importlib.reload(boxes)
    try:
        assert reloaded.CROP == (0, 100, 1280, 470)
    finally:
        monkeypatch.delenv("VISION_CROP", raising=False)
        importlib.reload(boxes)
