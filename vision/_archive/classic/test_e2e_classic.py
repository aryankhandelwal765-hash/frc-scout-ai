"""End-to-end: synthetic clip -> classic CV pipeline -> the known answer.

make_test_clip.py draws a scripted match whose outcome is known by construction:
RED robot R0 scores 2 balls, BLUE robot B0 scores 1. If the whole classic chain
(detection -> tracking -> possession -> goal-zone scoring) is intact, it must
recover exactly that.

This is the repo's one true integration test. It is marked `slow` because it
renders and re-reads 360 frames of video (a few seconds).
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("cv2", reason="the classic pipeline needs opencv")

ROOT = Path(__file__).resolve().parent.parent
CLASSIC = ROOT / "vision" / "classic"

EXPECTED_RED = 2
EXPECTED_BLUE = 1


@pytest.fixture(scope="module")
def rendered_clip(tmp_path_factory):
    """Generate test_clip.mp4 + test_zones.json in a scratch directory."""
    workdir = tmp_path_factory.mktemp("classic_e2e")
    subprocess.run(
        [sys.executable, str(CLASSIC / "make_test_clip.py")],
        cwd=workdir,
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    clip = workdir / "test_clip.mp4"
    zones = workdir / "test_zones.json"
    assert clip.exists(), "make_test_clip.py produced no clip"
    assert zones.exists(), "make_test_clip.py produced no zones"
    return workdir, clip, zones


@pytest.mark.slow
def test_pipeline_recovers_the_known_score(rendered_clip):
    workdir, clip, zones = rendered_clip
    outdir = workdir / "out"
    proc = subprocess.run(
        [
            sys.executable,
            str(CLASSIC / "track_match.py"),
            "--input",
            str(clip),
            "--zones",
            str(zones),
            "--outdir",
            str(outdir),
        ],
        cwd=workdir,
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )

    summary_path = outdir / "summary.json"
    assert summary_path.exists(), f"no summary.json\n{proc.stdout[-2000:]}"
    summary = json.loads(summary_path.read_text())

    red = summary["red_alliance_balls"]
    blue = summary["blue_alliance_balls"]
    assert red == EXPECTED_RED, f"RED {red}, expected {EXPECTED_RED}\n{proc.stdout[-2000:]}"
    assert blue == EXPECTED_BLUE, f"BLUE {blue}, expected {EXPECTED_BLUE}\n{proc.stdout[-2000:]}"
    assert summary["total_balls_scored"] == EXPECTED_RED + EXPECTED_BLUE
    assert summary["unattributed_balls"] == 0, "every ball must be attributed to a robot"

    # Per-robot attribution, not just the alliance totals. Robot ids are the
    # tracker's ints (rendered as R0/B0 in the printed summary); the clip has
    # exactly one robot per alliance, so all of each total lands on that robot.
    by_robot = {(r["alliance"], r["robot"]): r["balls_scored"] for r in summary["per_robot"]}
    assert len(by_robot) == 2, f"expected one scorer per alliance, got {by_robot}"
    assert by_robot[("red", 0)] == EXPECTED_RED
    assert by_robot[("blue", 0)] == EXPECTED_BLUE


@pytest.mark.slow
def test_pipeline_writes_its_documented_outputs(rendered_clip):
    workdir, clip, zones = rendered_clip
    outdir = workdir / "out2"
    subprocess.run(
        [
            sys.executable,
            str(CLASSIC / "track_match.py"),
            "--input",
            str(clip),
            "--zones",
            str(zones),
            "--outdir",
            str(outdir),
        ],
        cwd=workdir,
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    for name in ("summary.json", "robot_positions.csv", "ball_positions.csv", "scoring_events.csv"):
        assert (outdir / name).exists(), f"missing {name}"


def test_test_zones_are_valid_goal_rects(rendered_clip):
    _workdir, _clip, zones = rendered_clip
    data = json.loads(zones.read_text())
    assert data, "no zones written"
    # Both alliances must have a goal, or scoring can only ever be one-sided.
    blob = json.dumps(data).lower()
    assert "red" in blob and "blue" in blob
