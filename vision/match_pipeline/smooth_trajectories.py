"""
Phase 3 of the overnight pipeline: apply 2D field correction + Kalman filter
to every match's trajectory, then re-bake the auto paths.

Two outputs per match:
  smoothed_trajectory.csv  -- full-match field-inch positions (Kalman-smoothed)
  smoothed_auto_paths.json -- like auto_paths.json but paths in corrected field
                              inches; topdown_paths.py uses this automatically

Kalman state: [x, y, vx, vy] in field inches, constant-velocity model.
  - Predict-only during occlusion gaps (robot carried a ball, crossing paths, etc.)
  - Gap-fill adds interpolated rows between observations (up to MAX_GAP_FRAMES)
  - Hard speed gate: measurements implying > MAX_SPEED_IN_S are treated as outliers
    and skipped (avoids a blip from a mis-detected crop at the wrong robot)

Run:
  python smooth_trajectories.py                    # all matches
  python smooth_trajectories.py --only 2026hop_qm1 2026hop_qm2
  python smooth_trajectories.py --force            # re-smooth even if already done
"""

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import topdown as td
import zones

RESULTS = os.path.join(HERE, "results")
CALIB = os.path.join(HERE, "calib.json")

# Kalman tuning
PROCESS_ACCEL = 150.0  # FRC robot max accel estimate, in/s²
MEAS_STD = 12.0  # expected position noise after Phase 1+2, inches
MAX_GAP_FRAMES = 180  # don't fill gaps longer than this (3s at 60fps)
MAX_SPEED = 200.0  # in/s; measurements implying faster → outlier

# Canvas -> field-inch scale (matches topdown.py constants)
_FX0, _FY0, _FX1, _FY1 = td.FX0, td.FY0, td.FX1, td.FY1
_FIELD_LEN = zones.FIELD_LEN
_FIELD_WID = zones.FIELD_WID


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def _canvas_to_field(canvas_pts):
    """topdown canvas pixels -> field inches (linear scale, no correction)."""
    a = np.asarray(canvas_pts, float).reshape(-1, 2)
    x = (a[:, 0] - _FX0) * _FIELD_LEN / (_FX1 - _FX0)
    y = (a[:, 1] - _FY0) * _FIELD_WID / (_FY1 - _FY0)
    return np.column_stack([x, y])


def _field_to_canvas(field_pts):
    """Field inches -> topdown canvas pixels (inverse of _canvas_to_field)."""
    a = np.asarray(field_pts, float).reshape(-1, 2)
    cx = a[:, 0] * (_FX1 - _FX0) / _FIELD_LEN + _FX0
    cy = a[:, 1] * (_FY1 - _FY0) / _FIELD_WID + _FY0
    return np.column_stack([cx, cy])


def _load_correction():
    """Load 2D correction coefficients from calib.json."""
    calib = json.load(open(CALIB)) if os.path.exists(CALIB) else {}
    return calib.get("correction_2d")


def _apply_correction(raw_x, raw_y, corr):
    """Apply correction_2d polynomial to (raw_X, raw_Y) arrays -> (corr_X, corr_Y)."""
    if corr is None:
        return np.asarray(raw_x, float), np.asarray(raw_y, float)
    deg = corr["degree"]
    cx, cy = np.asarray(raw_x, float), np.asarray(raw_y, float)
    cols = [np.ones_like(cx)]
    for d in range(1, deg + 1):
        for i in range(d + 1):
            cols.append(cx ** (d - i) * cy**i)
    A = np.column_stack(cols)
    dx = A @ np.array(corr["coef_x"])
    dy = A @ np.array(corr["coef_y"])
    return cx + dx, cy + dy


# ---------------------------------------------------------------------------
# Kalman filter
# ---------------------------------------------------------------------------


class KalmanFilter:
    """2D constant-velocity Kalman. State: [x, y, vx, vy] in field inches."""

    def __init__(self, dt, accel_std=PROCESS_ACCEL, meas_std=MEAS_STD):
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], np.float64)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float64)
        g = np.array([dt**2 / 2, dt**2 / 2, dt, dt], np.float64)
        self.Q = np.outer(g, g) * accel_std**2
        self.R = np.eye(2) * meas_std**2
        self.x = np.zeros(4)
        self.P = np.diag([meas_std**2, meas_std**2, (accel_std * dt) ** 2, (accel_std * dt) ** 2])

    def init(self, x, y):
        self.x[:] = [x, y, 0.0, 0.0]
        self.P = np.diag([self.R[0, 0], self.R[1, 1], 1000.0, 1000.0])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, meas_x, meas_y):
        z = np.array([meas_x, meas_y])
        innov = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innov
        self.P = (np.eye(4) - K @ self.H) @ self.P

    @property
    def pos(self):
        return self.x[0], self.x[1]

    @property
    def vel(self):
        return self.x[2], self.x[3]


# ---------------------------------------------------------------------------
# Per-match processing
# ---------------------------------------------------------------------------


def _load_tracks_json(match):
    path = os.path.join(RESULTS, match, "tracks.json")
    if os.path.exists(path):
        d = json.load(open(path))
        return float(d.get("fps", 60.0)), int(d.get("stride", 2))
    return 60.0, 2


def smooth_match(match, mapper, correction):
    """Smooth trajectory.csv for one match; write smoothed_trajectory.csv.

    Returns a dict tid -> list of smoothed rows:
      (frame, t, alliance, field_x, field_y, vx, vy, interpolated, conf)
    """
    traj_path = os.path.join(RESULTS, match, "trajectory.csv")
    if not os.path.exists(traj_path):
        return None

    fps, stride = _load_tracks_json(match)
    dt = stride / fps

    # group by tid
    raw = defaultdict(list)
    with open(traj_path) as f:
        for row in csv.DictReader(f):
            raw[int(row["tid"])].append(
                (
                    int(row["frame"]),
                    float(row["t"]),
                    row["alliance"],
                    float(row["cx"]),
                    float(row["cy"]),
                    float(row.get("conf", 0.9)),
                )
            )
    for tid in raw:
        raw[tid].sort()

    smoothed_tracks = {}
    for tid, rows in raw.items():
        # Map all observations to corrected field inches
        cx_arr = np.array([r[3] for r in rows])
        cy_arr = np.array([r[4] for r in rows])
        canvas_pts = mapper(list(zip(cx_arr, cy_arr)))
        raw_fi = _canvas_to_field(canvas_pts)
        corr_x, corr_y = _apply_correction(raw_fi[:, 0], raw_fi[:, 1], correction)

        kf = KalmanFilter(dt)
        kf.init(corr_x[0], corr_y[0])

        out_rows = []
        prev_frame = rows[0][0]
        prev_t = rows[0][1]

        for i, (frame, t, alliance, _, _, conf) in enumerate(rows):
            fx, fy = corr_x[i], corr_y[i]

            # Predict through any gap since last observation
            gap = frame - prev_frame
            if gap > stride:
                steps = gap // stride
                # Check if this gap is bridgeable (robot occluded, not gone)
                if gap <= MAX_GAP_FRAMES:
                    for s in range(1, steps):
                        kf.predict()
                        interp_frame = prev_frame + s * stride
                        interp_t = prev_t + s * dt
                        px, py = kf.pos
                        vx, vy = kf.vel
                        out_rows.append((interp_frame, interp_t, alliance, px, py, vx, vy, 1, 0.0))
                else:
                    # Long gap: re-init Kalman at this observation
                    kf.init(fx, fy)
                    steps = 0  # no intermediate predictions

                for _ in range(steps):
                    kf.predict()
            else:
                kf.predict()

            # Speed gate: skip measurement if it implies impossible speed
            px, py = kf.pos
            speed = ((fx - px) ** 2 + (fy - py) ** 2) ** 0.5 / dt
            if speed < MAX_SPEED:
                kf.update(fx, fy)
            # else: keep Kalman prediction, don't update

            px, py = kf.pos
            vx, vy = kf.vel
            out_rows.append((frame, t, alliance, px, py, vx, vy, 0, conf))
            prev_frame = frame
            prev_t = t

        smoothed_tracks[tid] = out_rows

    # Write smoothed_trajectory.csv
    out_path = os.path.join(RESULTS, match, "smoothed_trajectory.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "frame",
                "t",
                "tid",
                "alliance",
                "field_x",
                "field_y",
                "vx",
                "vy",
                "interpolated",
                "conf",
            ]
        )
        for tid, rows in sorted(smoothed_tracks.items()):
            for frame, t, alliance, fx, fy, vx, vy, interp, conf in rows:
                w.writerow(
                    [
                        frame,
                        round(t, 3),
                        tid,
                        alliance,
                        round(fx, 2),
                        round(fy, 2),
                        round(vx, 3),
                        round(vy, 3),
                        interp,
                        round(conf, 3),
                    ]
                )

    return smoothed_tracks


def smooth_auto_paths(match, smoothed_tracks):
    """Re-bake auto_paths.json paths using smoothed field-inch positions.

    Reads the existing auto_paths.json (which has team IDs, station assignments,
    etc., all determined from the raw data) and replaces each robot's path with
    the Kalman-smoothed version -- including gap-filled interpolated frames.

    Output: results/<match>/smoothed_auto_paths.json
      Same structure as auto_paths.json but:
        path: [[t, field_x, field_y], ...]
        coords: "field_in"   <- tells topdown_paths.py to skip the mapper
    """
    ap_path = os.path.join(RESULTS, match, "auto_paths.json")
    if not os.path.exists(ap_path):
        return
    data = json.load(open(ap_path))
    teams = data.get("teams", {})
    if not teams:
        return

    # Auto window from original run (filter smoothed frames to match)
    dbg = data.get("debug", {})
    start_frame = dbg.get("start_frame", 0)
    end_frame = dbg.get("end_frame", 10**9)

    # Build fast lookup: tid -> {frame -> (t, field_x, field_y)}, auto window only
    frame_lookup = defaultdict(dict)
    for tid, rows in smoothed_tracks.items():
        for frame, t, alliance, fx, fy, *_ in rows:
            if start_frame <= frame <= end_frame:
                frame_lookup[tid][frame] = (t, fx, fy)

    new_teams = {}
    for station, info in teams.items():
        tids = info.get("tids") or [info.get("tid")]
        new_path = []
        for tid in tids:
            if tid not in frame_lookup:
                continue
            for frame, (t, fx, fy) in sorted(frame_lookup[tid].items()):
                new_path.append([round(t, 3), round(fx, 2), round(fy, 2)])
        # Sort by time, deduplicate
        new_path.sort(key=lambda r: r[0])
        seen_t = set()
        deduped = []
        for r in new_path:
            if r[0] not in seen_t:
                deduped.append(r)
                seen_t.add(r[0])

        entry = dict(info)
        entry["path"] = deduped if deduped else info["path"]
        entry["coords"] = "field_in" if deduped else "crop_px"
        new_teams[station] = entry

    out = {"debug": dbg, "teams": new_teams, "smoothed": True}
    out_path = os.path.join(RESULTS, match, "smoothed_auto_paths.json")
    json.dump(out, open(out_path, "w"), indent=2)


def process_all(matches=None, force=False):
    calib = json.load(open(CALIB)) if os.path.exists(CALIB) else {}
    if "fieldmap" not in calib:
        raise SystemExit("No fieldmap in calib.json -- run multi_video_apriltag_calib.py first")

    mapper = td.load_mapper()
    if mapper is None:
        raise SystemExit("load_mapper() returned None")
    correction = _load_correction()
    if correction:
        print("Using 2D correction from calib.json")
    else:
        print("No correction_2d in calib.json -- using Phase 1 mapper only")

    if matches is None:
        traj_dirs = sorted(glob.glob(os.path.join(RESULTS, "*", "trajectory.csv")))
        matches = [os.path.basename(os.path.dirname(p)) for p in traj_dirs]

    ok = fail = skip = 0
    for match in matches:
        out_path = os.path.join(RESULTS, match, "smoothed_trajectory.csv")
        if not force and os.path.exists(out_path):
            skip += 1
            continue
        try:
            st = smooth_match(match, mapper, correction)
            if st is None:
                fail += 1
                print(f"  {match}: no trajectory.csv, skipped")
                continue
            smooth_auto_paths(match, st)
            n_interp = sum(1 for rows in st.values() for r in rows if r[7] == 1)
            print(
                f"  {match}: {sum(len(r) for r in st.values())} rows "
                f"({n_interp} interpolated), {len(st)} tracks"
            )
            ok += 1
        except Exception as e:
            import traceback

            fail += 1
            print(f"  {match}: FAILED -- {e}")
            traceback.print_exc()

    print(f"\nDone: {ok} smoothed, {skip} skipped (already done), {fail} failed")
    if fail:
        print("  Re-run with --force to retry failed matches")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", help="limit to these match keys")
    p.add_argument(
        "--force",
        action="store_true",
        help="re-smooth even if smoothed_trajectory.csv already exists",
    )
    a = p.parse_args()
    process_all(matches=a.only, force=a.force)


if __name__ == "__main__":
    main()
