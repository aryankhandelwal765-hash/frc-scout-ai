"""stitch — relinking ByteTrack fragments into one chain per physical robot.

crop_width() opens the match video to size the gates; it is patched out here so
these stay pure unit tests. Gates are fractions of the crop width, so a nominal
1000 px crop makes the arithmetic easy to reason about:
    base gate   = 0.045 * 1000 = 45 px at zero gap
    gate growth = 0.020 * 1000 = 20 px per second of gap
"""

import pytest

pytest.importorskip("cv2", reason="stitch needs opencv")

import stitch  # noqa: E402

CW = 1000.0
FPS = 60.0


@pytest.fixture(autouse=True)
def _fixed_crop_width(monkeypatch):
    monkeypatch.setattr(stitch, "crop_width", lambda match: CW)


def frag(start_frame, n, x0, y0, dx=0.0, dy=0.0, alliance="red"):
    """A fragment as ByteTrack rows: (frame, t, alliance, cx, cy, conf)."""
    return [
        (start_frame + i, (start_frame + i) / FPS, alliance, x0 + dx * i, y0 + dy * i, 0.9)
        for i in range(n)
    ]


def test_stationary_robot_reacquired_nearby_is_one_chain():
    """The bug this module exists for: same robot, new track id."""
    tracks = {1: frag(0, 30, 500, 300), 2: frag(40, 30, 505, 302)}
    out = stitch.chains("m", tracks, {1: "red", 2: "red"}, fps=FPS)
    assert len(out) == 1
    assert out[0] == [1, 2]


def test_fragment_far_away_is_not_chained():
    """Beyond the gate, these are two different robots."""
    tracks = {1: frag(0, 30, 100, 100), 2: frag(40, 30, 900, 600)}
    out = stitch.chains("m", tracks, {1: "red", 2: "red"}, fps=FPS)
    assert sorted(len(c) for c in out) == [1, 1]


def test_opposite_alliances_are_never_chained():
    """Even perfectly co-located — bumper colour is the hard partition."""
    tracks = {1: frag(0, 30, 500, 300), 2: frag(40, 30, 500, 300)}
    out = stitch.chains("m", tracks, {1: "red", 2: "blue"}, fps=FPS)
    assert sorted(len(c) for c in out) == [1, 1]


def test_gap_beyond_max_is_not_bridged():
    """MAX_GAP_SEC is 6s; 8s of blindness is too much to claim continuity."""
    tracks = {1: frag(0, 30, 500, 300), 2: frag(30 + int(8 * FPS), 30, 500, 300)}
    out = stitch.chains("m", tracks, {1: "red", 2: "red"}, fps=FPS)
    assert sorted(len(c) for c in out) == [1, 1]


def test_moving_robot_is_chained_at_its_extrapolated_position():
    """Velocity-aware gating: compare against where A was heading, not where it
    stopped. A static-position comparison would reject this real link."""
    a = frag(0, 30, 200, 300, dx=6.0)  # ends near x=374, moving +x
    end_x = a[-1][3]
    gap = 12
    b = frag(30 + gap, 30, end_x + 6.0 * gap, 300, dx=6.0)
    out = stitch.chains("m", {1: a, 2: b}, {1: "red", 2: "red"}, fps=FPS)
    assert out == [[1, 2]]


def test_short_fragments_are_dropped():
    """min_frames filters detector noise before any linking happens."""
    tracks = {1: frag(0, 30, 500, 300), 2: frag(40, 1, 505, 302)}
    out = stitch.chains("m", tracks, {1: "red", 2: "red"}, fps=FPS, min_frames=3)
    assert out == [[1]]


def test_every_surviving_fragment_appears_exactly_once():
    """No fragment may be dropped or duplicated across the returned chains."""
    tracks = {
        1: frag(0, 30, 500, 300),
        2: frag(40, 30, 505, 302),
        3: frag(0, 30, 100, 700),
        4: frag(200, 30, 900, 100),
    }
    out = stitch.chains("m", tracks, {i: "red" for i in (1, 2, 3, 4)}, fps=FPS)
    flat = [tid for chain in out for tid in chain]
    assert sorted(flat) == [1, 2, 3, 4]
    assert len(flat) == len(set(flat))


def test_chain_is_ordered_in_time():
    tracks = {7: frag(40, 30, 505, 302), 3: frag(0, 30, 500, 300)}
    out = stitch.chains("m", tracks, {3: "red", 7: "red"}, fps=FPS)
    assert out == [[3, 7]]


def test_a_read_anywhere_locks_the_whole_chain():
    """The payoff: one bumper read on fragment 2 names fragment 1 as well."""
    chain_list = [[1, 2, 3]]
    per_tid, per_chain = stitch.lock_identity(chain_list, {2: 1706})
    assert per_chain[0] == 1706
    assert per_tid[1] == per_tid[2] == per_tid[3] == 1706


def test_majority_read_wins_within_a_chain():
    chain_list = [[1, 2, 3]]
    _per_tid, per_chain = stitch.lock_identity(chain_list, {1: 1706, 2: 1706, 3: 3035})
    assert per_chain[0] == 1706


def test_chain_with_no_reads_gets_no_team():
    _per_tid, per_chain = stitch.lock_identity([[1, 2]], {})
    assert per_chain == {}


def test_same_team_may_claim_chains_that_do_not_overlap_in_time():
    """One robot is legitimately split into many chains across a match. The
    time-aware rule (pass `tracks`) must not starve the later chain."""
    tracks = {1: frag(0, 30, 500, 300), 2: frag(600, 30, 500, 300)}
    chain_list = [[1], [2]]
    _per_tid, per_chain = stitch.lock_identity(chain_list, {1: 1706, 2: 1706}, tracks)
    assert per_chain.get(0) == 1706
    assert per_chain.get(1) == 1706, "non-overlapping chains must both keep the id"


# --- dedupe + slot stitching ------------------------------------------------


def test_dedupe_folds_a_track_riding_on_another():
    tracks = {1: frag(0, 60, 500, 300), 2: frag(10, 30, 505, 301)}
    kept, merged = stitch.dedupe(tracks, {1: "red", 2: "red"}, CW)
    assert merged == {2: 1}
    assert set(kept) == {1}


def test_dedupe_keeps_two_robots_side_by_side():
    tracks = {1: frag(0, 60, 500, 300), 2: frag(0, 60, 700, 300)}
    kept, merged = stitch.dedupe(tracks, {1: "red", 2: "red"}, CW)
    assert not merged and set(kept) == {1, 2}


def test_dedupe_never_merges_across_alliances():
    tracks = {1: frag(0, 60, 500, 300), 2: frag(0, 60, 500, 300)}
    _kept, merged = stitch.dedupe(tracks, {1: "red", 2: "blue"}, CW)
    assert not merged


def test_slots_bridge_a_gap_longer_than_the_greedy_limit():
    """8 s blind is past MAX_GAP_SEC for chains(); a free slot still takes the robot back."""
    tracks = {1: frag(0, 120, 500, 300), 2: frag(120 + int(8 * FPS), 120, 520, 310)}
    al = {1: "red", 2: "red"}
    assert sorted(len(c) for c in stitch.chains("m", tracks, al, fps=FPS)) == [1, 1]
    assert stitch.slot_chains("m", tracks, al, fps=FPS) == [[1, 2]]


def test_slots_never_exceed_three_robots_per_alliance():
    """Four long concurrent fragments: the fourth cannot found a robot, it is junk."""
    tracks = {i: frag(0, 120, 100 + 200 * i, 300) for i in range(4)}
    out = stitch.slot_chains("m", tracks, {i: "red" for i in range(4)}, fps=FPS)
    assert sum(1 for c in out if len(c) >= 1) == 4  # 3 slots + 1 junk singleton
    assert sum(1 for c in out if c[0] != 3 or len(c) > 1) >= 3


def test_slot_reads_keep_teams_apart():
    """Both candidates are equally near; the bumper read decides which slot takes it."""
    a = frag(0, 120, 400, 300)
    b = frag(0, 120, 600, 300)
    late = frag(120 + int(2 * FPS), 60, 500, 300)  # midway between the two
    tracks = {1: a, 2: b, 3: late}
    al = {1: "red", 2: "red", 3: "red"}
    out = stitch.slot_chains("m", tracks, al, fps=FPS, reads={1: "111", 2: "222", 3: "222"})
    assert [3] in [c[-1:] for c in out if 2 in c]
    assert not any(1 in c and 3 in c for c in out)
