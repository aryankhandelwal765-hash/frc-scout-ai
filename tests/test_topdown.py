"""topdown — camera->field homography and the polynomial field map.

Round-trip tests: project camera points forward, invert, and require the
originals back. A homography that does not round-trip has a transposed or
mis-ordered corner list, which is the classic way this breaks silently.
"""

import numpy as np
import pytest

pytest.importorskip("cv2", reason="topdown needs opencv")

import cv2  # noqa: E402

import topdown as td  # noqa: E402


# A plausible camera-side quadrilateral of the field, in CROP pixels.
# Order matches FIELD_CORNERS everywhere: TL, TR, BR, BL.
SRC = np.array([[120, 180], [1160, 175], [1265, 455], [15, 460]], np.float32)


def test_canvas_geometry_is_self_consistent():
    assert td.FX1 > td.FX0 and td.FY1 > td.FY0
    assert td.FIELD_CORNERS.shape == (4, 2)


def test_corner_order_is_tl_tr_br_bl():
    """Every consumer assumes this order; assert it rather than trust it."""
    tl, tr, br, bl = td.FIELD_CORNERS
    assert tl[0] < tr[0] and tl[1] == tr[1]  # top edge, left to right
    assert bl[0] < br[0] and bl[1] == br[1]  # bottom edge, left to right
    assert tl[1] < bl[1]  # top above bottom


def test_project_maps_the_field_quad_onto_the_canvas_rectangle():
    H = cv2.getPerspectiveTransform(SRC, td.FIELD_CORNERS)
    out = td.project(H, SRC)
    assert out == pytest.approx(np.asarray(td.FIELD_CORNERS), abs=1e-3)


def test_homography_round_trips():
    H = cv2.getPerspectiveTransform(SRC, td.FIELD_CORNERS)
    H_inv = np.linalg.inv(H)
    pts = np.array([[300, 250], [640, 320], [1000, 400], [200, 440]], np.float32)
    there = td.project(H, pts)
    back = td.project(H_inv.astype(np.float32), there)
    assert back == pytest.approx(pts, abs=1e-2)


def test_projection_preserves_interior_points():
    """A point inside the camera quad must land inside the canvas rectangle."""
    H = cv2.getPerspectiveTransform(SRC, td.FIELD_CORNERS)
    centre = SRC.mean(axis=0, keepdims=True)
    x, y = td.project(H, centre)[0]
    assert td.FX0 <= x <= td.FX1
    assert td.FY0 <= y <= td.FY1


def test_project_handles_a_single_point():
    H = cv2.getPerspectiveTransform(SRC, td.FIELD_CORNERS)
    assert td.project(H, [[300, 250]]).shape == (1, 2)


# ── Polynomial field map ─────────────────────────────────────────────────


def _landmark_src():
    """Camera points for the 9 landmarks — here, the canvas points themselves
    plus a mild quadratic warp, so the degree-2 fit has something to undo."""
    dst = np.array([t for _, t in td.POLY_LANDMARKS], float)
    warped = dst.copy()
    warped[:, 0] += 0.00004 * (dst[:, 1] - td.FY0) ** 2
    return warped


def test_fieldmap_fits_its_own_landmarks():
    """The fit must reproduce the 9 landmarks it was built from."""
    src = _landmark_src()
    fm = td.fit_fieldmap(src, degree=2)
    out = td.apply_fieldmap(fm, src)
    expected = np.array([t for _, t in td.POLY_LANDMARKS], float)
    assert out == pytest.approx(expected, abs=1e-6)


def test_fieldmap_is_serializable():
    """It is written into calib.json, so it must be plain JSON types."""
    import json

    fm = td.fit_fieldmap(_landmark_src(), degree=2)
    assert json.loads(json.dumps(fm))["degree"] == 2
    assert isinstance(fm["cx"], list) and isinstance(fm["cy"], list)


def test_design_matrix_width_matches_degree():
    """Degree 2 in two variables -> 6 columns: [1, x, y, x^2, xy, y^2]."""
    pts = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert td._design(pts, 2).shape == (3, 6)
    assert td._design(pts, 1).shape == (3, 3)


def test_there_are_nine_landmarks_in_canvas_bounds():
    assert len(td.POLY_LANDMARKS) == 9
    for _name, (x, y) in td.POLY_LANDMARKS:
        assert td.FX0 <= x <= td.FX1
        assert td.FY0 <= y <= td.FY1
