from robot_filters import contained_frac, eliminate_alliances, merge_nested, veto_people


def test_contained_frac():
    assert contained_frac((0, 0, 10, 10), (0, 0, 20, 20)) == 1.0
    assert contained_frac((0, 0, 10, 10), (5, 0, 20, 10)) == 0.5
    assert contained_frac((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0
    assert contained_frac((0, 0, 0, 0), (0, 0, 5, 5)) == 0.0


def test_veto_drops_box_on_a_ref_but_keeps_robot_half_behind_one():
    ref = (100, 0, 140, 200)  # tall, narrow person box
    on_ref = (105, 20, 135, 180)  # box drawn around the ref
    behind_ref = (60, 120, 160, 190)  # wide robot, ref covers only the middle ~40%
    clear = (300, 100, 400, 180)
    kept, vetoed = veto_people([on_ref, behind_ref, clear], [ref])
    assert vetoed == [on_ref]
    assert kept == [behind_ref, clear]


def test_merge_nested_keeps_larger_box_only():
    big = {"box": (0, 0, 100, 100)}
    small = {"box": (10, 10, 60, 60)}
    far = {"box": (200, 200, 260, 260)}
    assert merge_nested([small, big, far]) == [big, far]


def test_merge_nested_keeps_overlapping_but_not_nested_boxes():
    a = {"box": (0, 0, 100, 100)}
    b = {"box": (60, 0, 160, 100)}  # 40% of b inside a
    assert merge_nested([a, b]) == [a, b]


def test_elimination_fills_the_missing_alliance_colour():
    filled, how = eliminate_alliances(["blue", "blue", "blue", None, "red"])
    assert filled == ["blue", "blue", "blue", "red", "red"]
    assert how == ["read", "read", "read", "elim", "read"]
    filled, how = eliminate_alliances(["red", "red", "red", None])
    assert filled[3] == "blue" and how[3] == "elim"


def test_elimination_does_not_guess_when_count_is_undecided():
    filled, how = eliminate_alliances(["blue", "blue", None, "red"])
    assert filled == ["blue", "blue", None, "red"]
    assert how[2] == "unknown"


def test_elimination_leaves_impossible_frames_unknown():
    # 3 blue and 3 red already read, so a 7th robot cannot be either colour
    filled, how = eliminate_alliances(["blue"] * 3 + ["red"] * 3 + [None])
    assert filled[-1] is None and how[-1] == "unknown"
