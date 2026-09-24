"""
Solve the camera->ground homography by FUSING two independent sources.

WHY THE PREVIOUS MAPS WERE WRONG. Projecting the field grid back onto the camera
image shows the active map draws X=0, 158.6, 325.6, 492.6, 651.2 as straight,
parallel, vertical lines. A real perspective view of a plane does not do that -- the
map is effectively affine, so it is right in the middle (where it was anchored) and
wrong at the edges, exactly matching the measured error pattern: 3.0% centre-line
violations (a coarse, mid-field question) but 47in zone bands unusable.

THE TWO SOURCES, AND WHY NEITHER WORKS ALONE.
  AprilTags   6 are visible in the top view with exact published 3D positions and
              0.23-0.42px corner jitter. But solvePnP needs the focal length, the
              broadcast has no EXIF, and a focal sweep is degenerate: long-focal
              solutions fit the tags nearly as well while putting the camera in
              physically impossible places (one landed 2.4m BELOW the floor).
  Robots      the manual requires BUMPERS to overlap the ROBOT STARTING LINE at
              match start, so ~300 tracked robots sit at a known X. That pins the
              ground plane but says nothing about the camera.

FUSING THEM. Sweep the focal length; for each, solve the tag pose, derive the
implied ground-plane homography, and score it by how well the starting-line robots
land on their known X. The tags fix the geometry, the robots fix the scale. Neither
constraint is free to be satisfied by an absurd camera.

Validation is the same objective test used throughout: CENTER LINE rule violations
and the split-half reliability of the zone columns. Written only if it wins.

PyCharm: click Run.
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PARENT)
import auto_paths as ap
import topdown as td
import topdown_paths as tp

VIDEOS = os.path.join(HERE, "videos")
RESULTS = os.path.join(HERE, "results")
CALIB = os.path.join(HERE, "calib.json")
LAYOUT = os.path.join(HERE, "apriltags_2026_welded.json")

# ===========================================================================
# EDIT ME (PyCharm one-click).
# ===========================================================================
TAG_MATCH = "2026hop_qm27"   # any match; the camera is fixed for the event
FOCAL_LO, FOCAL_HI, FOCAL_STEP = 600, 6000, 25
APPLY = True                 # write the winner into calib.json
# ===========================================================================

IN = 0.0254
FIELD_LEN, FIELD_WID = 651.2, 317.7
START_X = 158.6
HALF = 8.125 / 2 * IN


def _quat_R(q):
    w, x, y, z = q["W"], q["X"], q["Y"], q["Z"]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def tag_correspondences(match=TAG_MATCH):
    """(obj 3D metres, img 2D px) from tags detected in the top view."""
    import cv2.aruco as A
    lay = {t["ID"]: t for t in json.load(open(LAYOUT))["tags"]}
    det = A.ArucoDetector(A.getPredefinedDictionary(A.DICT_APRILTAG_36H11),
                          A.DetectorParameters())
    acc = defaultdict(list)
    cap = cv2.VideoCapture(os.path.join(VIDEOS, f"{match}.mp4"))
    ok, f0 = cap.read()
    if not ok:
        raise SystemExit(f"cannot read {match}")
    H, W = f0.shape[:2]
    for f in range(0, 9000, 60):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        if not ok:
            break
        cs, ids, _ = det.detectMarkers(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
        if ids is None:
            continue
        for c, i in zip(cs, ids.ravel()):
            acc[int(i)].append(c.reshape(4, 2))
    cap.release()

    # local corner layout that measured best (zy_ccw); tag +X is its outward normal
    loc = np.array([[0, +1, -1], [0, +1, +1], [0, -1, +1], [0, -1, -1]], float) * HALF
    obj, img, used = [], [], []
    for tid, v in sorted(acc.items()):
        if tid not in lay or len(v) < 5:
            continue
        p = lay[tid]["pose"]
        t, R = p["translation"], _quat_R(p["rotation"]["quaternion"])
        obj.append((R @ loc.T).T + np.array([t["x"], t["y"], t["z"]]))
        img.append(np.median(np.stack(v), 0))
        used.append(tid)
    return np.concatenate(obj), np.concatenate(img), used, (W, H)


def line_observations():
    """(camera_px, known_field_X_inches) for robots on a ROBOT STARTING LINE."""
    obs = []
    for apf in sorted(glob.glob(os.path.join(RESULTS, "*", "auto_paths.json"))):
        if not tp._fresh(apf):
            continue
        m = os.path.basename(os.path.dirname(apf))
        sf = ap._detected_start(m)
        if sf is None:
            continue
        sf = max(0, sf)
        tracks = ap._load_trajectory(m)
        if not tracks:
            continue
        try:
            blue_left = tp._match_blue_left(json.load(open(apf))["teams"])
        except Exception:
            continue
        for tid, tr in tracks.items():
            if len(tr) < 80:
                continue
            near = [r for r in tr if abs(r[0] - sf) <= 3]
            if not near:
                continue
            r = min(near, key=lambda z: abs(z[0] - sf))
            al = ap._voted_alliance(m, tid, tr)
            own_left = (al == "blue") == blue_left
            obs.append((r[3], r[4], START_X if own_left else FIELD_LEN - START_X))
    return np.array(obs)


def ground_homography(rvec, tvec, K, dist, crop_off):
    """Camera image -> field inches, via four known ground points."""
    g = np.array([[0, 0, 0], [FIELD_LEN * IN, 0, 0],
                  [FIELD_LEN * IN, FIELD_WID * IN, 0], [0, FIELD_WID * IN, 0]], float)
    pr, _ = cv2.projectPoints(g, rvec, tvec, K, dist)
    pr = pr.reshape(-1, 2) - np.array(crop_off, float)   # full frame -> crop px
    dst = np.array([[0, 0], [FIELD_LEN, 0], [FIELD_LEN, FIELD_WID], [0, FIELD_WID]],
                   np.float32)
    if not np.isfinite(pr).all():
        return None
    return cv2.getPerspectiveTransform(pr.astype(np.float32), dst)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true", default=APPLY)
    a = p.parse_args()

    obj, img, tags, (W, H) = tag_correspondences()
    print(f"tags used: {tags}  ({len(obj)} corner correspondences)")
    obs = line_observations()
    print(f"starting-line observations: {len(obs)} robots")

    import trajectories as tj
    cx0, cy0, _cx1, _cy1 = tj.field_crop((H, W, 3))
    crop_off = (cx0, cy0)

    best = None
    for f in range(FOCAL_LO, FOCAL_HI, FOCAL_STEP):
        K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], float)
        dist = np.zeros(5)
        ok, rv, tv = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            continue
        R, _ = cv2.Rodrigues(rv)
        cam = (-R.T @ tv).ravel()
        # The camera is a broadcast rig inside the arena, not a satellite. Bound it
        # to a physically real position, otherwise the sweep converges on a
        # long-focal solution that fits the tags while sitting tens of metres away.
        if not (1.5 <= cam[2] <= 14.0):                 # height above the floor
            continue
        if not (-14.0 <= cam[1] <= FIELD_WID * IN + 14.0):   # beside the field
            continue
        if not (-8.0 <= cam[0] <= FIELD_LEN * IN + 8.0):     # along the field
            continue
        Hm = ground_homography(rv, tv, K, dist, crop_off)
        if Hm is None:
            continue
        got = cv2.perspectiveTransform(
            obs[:, :2].reshape(-1, 1, 2).astype(np.float32), Hm).reshape(-1, 2)
        if not np.isfinite(got).all():
            continue
        err = float(np.median(np.abs(got[:, 0] - obs[:, 2])))
        inb = float(((got[:, 1] >= -30) & (got[:, 1] <= FIELD_WID + 30)).mean())
        score = err + (1.0 - inb) * 100.0
        if best is None or score < best[0]:
            best = (score, f, err, inb, Hm, cam)

    if best is None:
        raise SystemExit("no physically valid solution found across the focal sweep")
    score, f, err, inb, Hm, cam = best
    print(f"\nbest focal {f}px")
    print(f"  camera at x={cam[0]:.2f} y={cam[1]:.2f} z={cam[2]:.2f} m "
          f"(field {FIELD_LEN * IN:.1f} x {FIELD_WID * IN:.1f} m)")
    print(f"  median |X error| on the starting line: {err:.1f} in")
    print(f"  robots landing within the field width: {inb * 100:.0f}%")

    if not a.apply:
        print("\n[report only] nothing written")
        return
    calib = json.load(open(CALIB)) if os.path.exists(CALIB) else {}
    calib.setdefault("fieldmap_prev", calib.get("fieldmap"))
    calib["ground_homography"] = {
        "H": Hm.tolist(), "focal_px": f, "cam_xyz_m": [float(v) for v in cam],
        "start_line_err_in": err, "n_tags": len(tags), "n_line_obs": int(len(obs)),
        "note": "camera crop px -> FIELD INCHES directly",
    }
    json.dump(calib, open(CALIB, "w"), indent=2)
    print(f"\nsaved ground_homography into {CALIB}")
    print("Validate before trusting it: validate_paths.py, then zone reliability.")


if __name__ == "__main__":
    main()
