"""
Phase 1 of the overnight pipeline: accumulate AprilTag observations from ALL match
videos and solve the best-fit camera pose across the event.

calib_solve.py only processes one video (TAG_MATCH = "2026hop_qm27"). With one
video you get ~20-40 tag detections. With the full event you get 300-600+ -- the
pose converges much tighter and the reprojection error drops significantly.

Writes to calib.json:
  ["ground_homography"]  -- H mapping crop_px -> field INCHES directly
  ["fieldmap"]           -- canvas_H mapping crop_px -> canvas_px (replaces polynomial,
                            picked up automatically by topdown.load_mapper())

Run: python multi_video_apriltag_calib.py
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import trajectories as tj

VIDEOS_DIR = os.path.join(HERE, "videos")
CALIB = os.path.join(HERE, "calib.json")
LAYOUT = os.path.join(HERE, "apriltags_2026_welded.json")

IN = 0.0254                     # metres per inch
FIELD_LEN_IN = 651.2
FIELD_WID_IN = 317.7
HALF_TAG_M = (8.125 / 2) * IN  # AprilTag half-size in metres

# Canvas constants from topdown.py (must match exactly)
CW, CH = 1640, 860
MARGIN_X, MARGIN_Y = 100, 80
FX0, FY0 = MARGIN_X, MARGIN_Y
FX1, FY1 = CW - MARGIN_X, CH - MARGIN_Y

# Focal sweep range. camera_calib.json has focal_px=1005, so sweep around that.
FOCAL_LO, FOCAL_HI, FOCAL_STEP = 700, 1400, 10


def _quat_R(q):
    w, x, y, z = q["W"], q["X"], q["Y"], q["Z"]
    return np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def _local_corners():
    """4 tag corners in the tag's LOCAL frame (metres). Layout: zy_ccw."""
    return np.array([
        [0, +1, -1],
        [0, +1, +1],
        [0, -1, +1],
        [0, -1, -1]], float) * HALF_TAG_M


def collect_observations(videos_dir, layout_file,
                          frames_per_video=300, frame_step=60):
    """Detect AprilTags in every .mp4 in videos_dir.

    frames_per_video caps one video's contribution so a long event opener
    doesn't drown out the others. frame_step=60 → one frame per second at 60fps.

    Returns (obj_all, img_all, used_tags, (W, H), crop_off).
    """
    lay = {t["ID"]: t for t in json.load(open(layout_file))["tags"]}
    det_params = cv2.aruco.DetectorParameters()
    det = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11),
        det_params)
    loc = _local_corners()

    acc = defaultdict(list)   # tag_id -> [4×2 corner arrays]
    frame_shape = None
    crop_off = (0, 0)

    videos = sorted(f for f in os.listdir(videos_dir) if f.endswith(".mp4"))
    if not videos:
        raise SystemExit(f"No .mp4 files found in {videos_dir}")

    for vid_name in videos:
        path = os.path.join(videos_dir, vid_name)
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        seen = 0
        for fi in range(0, total, frame_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            if frame_shape is None:
                frame_shape = frame.shape
                x1, y1, x2, y2 = tj.field_crop(frame_shape)
                crop_off = (x1, y1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cs, ids, _ = det.detectMarkers(gray)
            if ids is not None:
                for c, i in zip(cs, ids.ravel()):
                    if int(i) in lay:
                        acc[int(i)].append(c.reshape(4, 2))
            seen += 1
            if seen >= frames_per_video:
                break
        cap.release()
        print(f"  {vid_name}: {seen} frames, "
              f"{sum(1 for tid in acc if any(True for _ in acc[tid]))} tags seen")

    if frame_shape is None:
        raise SystemExit("Could not read any video frames")

    H_img, W_img = frame_shape[:2]
    obj_all, img_all, used_tags = [], [], []
    for tag_id in sorted(acc):
        obs = acc[tag_id]
        if len(obs) < 5:
            continue
        p = lay[tag_id]["pose"]
        R = _quat_R(p["rotation"]["quaternion"])
        t_trans = p["translation"]
        t_vec = np.array([t_trans["x"], t_trans["y"], t_trans["z"]])
        corners_world = (R @ loc.T).T + t_vec
        img_median = np.median(np.stack(obs), axis=0)
        obj_all.append(corners_world)
        img_all.append(img_median)
        used_tags.append(tag_id)
        print(f"    tag {tag_id:3d}: {len(obs)} observations")

    return (np.concatenate(obj_all).astype(np.float64),
            np.concatenate(img_all).astype(np.float64),
            used_tags, (W_img, H_img), crop_off)


def _ground_H_from_pose(rvec, tvec, K, dist, crop_off):
    """Compute H that maps crop_px -> field INCHES, via four floor corners."""
    corners_world_m = np.array([
        [0,                    0,                   0],
        [FIELD_LEN_IN * IN,    0,                   0],
        [FIELD_LEN_IN * IN,    FIELD_WID_IN * IN,   0],
        [0,                    FIELD_WID_IN * IN,   0]], float)
    proj, _ = cv2.projectPoints(corners_world_m, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2) - np.array(crop_off, float)
    dst = np.array([[0, 0], [FIELD_LEN_IN, 0],
                    [FIELD_LEN_IN, FIELD_WID_IN], [0, FIELD_WID_IN]], np.float32)
    if not np.isfinite(proj).all():
        return None
    return cv2.getPerspectiveTransform(proj.astype(np.float32), dst)


def _canvas_H_from_ground(H_ground):
    """Compose field-inch -> canvas-px scale onto ground_H to get crop_px -> canvas_px."""
    sx = (FX1 - FX0) / FIELD_LEN_IN
    sy = (FY1 - FY0) / FIELD_WID_IN
    S = np.array([[sx, 0, FX0],
                  [0, sy, FY0],
                  [0,  0,  1.0]])
    return (S @ H_ground).astype(np.float32)


def solve(obj, img, W, H, crop_off):
    """Focal-length sweep, return (reproj_px, focal, rvec, tvec, H_ground, cam_xyz)."""
    # Seed k1 from camera_calib.json if available
    cam_calib = os.path.join(HERE, "camera_calib.json")
    k1 = k2 = 0.0
    if os.path.exists(cam_calib):
        cc = json.load(open(cam_calib))
        k1, k2 = cc.get("k1", 0.0), cc.get("k2", 0.0)
    dist = np.array([k1, k2, 0, 0, 0], float)

    best = None
    for focal in range(FOCAL_LO, FOCAL_HI + 1, FOCAL_STEP):
        K = np.array([[focal, 0, W / 2.0],
                      [0, focal, H / 2.0],
                      [0, 0, 1.0]], float)
        ok, rv, tv = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            continue
        R, _ = cv2.Rodrigues(rv)
        cam_xyz = (-R.T @ tv).ravel()
        # Physical bounds: broadcast camera is above the arena, beside the field.
        # camera_calib.json reference: x=8.33m, y=-3.35m, z=3.41m.
        # Allow ±5m tolerance around those known bounds.
        if not (2.0 <= cam_xyz[2] <= 8.0):           # height: 2-8m above floor
            continue
        if not (-6.0 <= cam_xyz[1] <= FIELD_WID_IN * IN + 6.0):  # beside field
            continue
        if not (0.0 <= cam_xyz[0] <= FIELD_LEN_IN * IN):          # along field length
            continue
        proj, _ = cv2.projectPoints(obj, rv, tv, K, dist)
        reproj = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)))
        # Reject if reprojection error is implausibly high -- the old single-video
        # camera_calib.json achieved 24.8px; anything above 80px is a bad local min.
        if reproj > 80.0:
            continue
        H_ground = _ground_H_from_pose(rv, tv, K, dist, crop_off)
        if H_ground is None:
            continue
        if best is None or reproj < best[0]:
            best = (reproj, focal, rv.ravel(), tv.ravel(), H_ground, cam_xyz)

    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", default=VIDEOS_DIR)
    p.add_argument("--layout", default=LAYOUT)
    p.add_argument("--frame-step", type=int, default=60,
                   help="check every Nth frame per video (60fps → 60 = 1fps)")
    p.add_argument("--frames-per-video", type=int, default=300,
                   help="max frames per video (caps one video from dominating)")
    p.add_argument("--max-reproj", type=float, default=80.0,
                   help="reject any solution with reprojection error above this (px)")
    p.add_argument("--no-apply", action="store_true",
                   help="print results but do not write calib.json")
    a = p.parse_args()

    print(f"Collecting AprilTag observations across all videos in {a.videos} ...")
    obj, img, used_tags, (W, H), crop_off = collect_observations(
        a.videos, a.layout, a.frames_per_video, a.frame_step)
    print(f"\nUsed tags: {used_tags}  ({len(obj)} corner correspondences)")

    print(f"Sweeping focal length {FOCAL_LO}-{FOCAL_HI}px (step {FOCAL_STEP}) ...")
    result = solve(obj, img, W, H, crop_off)
    if result is None:
        raise SystemExit(
            "No physically valid camera pose found in focal sweep.\n"
            "All solutions either failed bounds or had reproj > --max-reproj.\n"
            "Try: --max-reproj 150 to loosen, or check that AprilTags are visible.")

    reproj, focal, rvec, tvec, H_ground, cam_xyz = result
    print(f"\nBest focal: {focal}px")
    print(f"Camera at  x={cam_xyz[0]:.2f}m  y={cam_xyz[1]:.2f}m  z={cam_xyz[2]:.2f}m")
    print(f"Mean reprojection error: {reproj:.2f}px  "
          f"({'GOOD' if reproj < 30 else 'OK' if reproj < 60 else 'POOR'})")
    if reproj > a.max_reproj:
        raise SystemExit(
            f"Reprojection error {reproj:.1f}px exceeds --max-reproj {a.max_reproj}px.\n"
            "Camera pose is unreliable. Not writing calib.json.\n"
            f"Try: --max-reproj {reproj+20:.0f} to accept this solution anyway.")

    if a.no_apply:
        print("\n[--no-apply] nothing written")
        return

    calib = json.load(open(CALIB)) if os.path.exists(CALIB) else {}
    # Back up the existing polynomial fieldmap before replacing it
    if "fieldmap" in calib and calib["fieldmap"].get("degree"):
        calib["fieldmap_poly_backup"] = calib["fieldmap"]

    calib["ground_homography"] = {
        "H": H_ground.tolist(),
        "focal_px": focal,
        "cam_xyz_m": cam_xyz.tolist(),
        "reproj_px": reproj,
        "crop_off": list(crop_off),
        "n_tags": len(used_tags),
        "n_corners": int(len(obj)),
        "note": "crop_px -> field INCHES (Phase 1, multi-video)",
    }
    # Replace fieldmap so load_mapper() picks up the better homography
    canvas_H = _canvas_H_from_ground(H_ground)
    calib["fieldmap"] = {
        "kind": "homography",
        "H": canvas_H.tolist(),
        "note": "crop_px -> canvas_px via Phase 1 camera pose; overwrites polynomial",
    }
    json.dump(calib, open(CALIB, "w"), indent=2)
    print(f"\nWritten to {CALIB}")
    print("  calib['ground_homography'] -- crop_px -> field inches")
    print("  calib['fieldmap']          -- crop_px -> canvas_px (replaces polynomial)")
    print("\nNext: fit_2d_correction.py")


if __name__ == "__main__":
    main()
