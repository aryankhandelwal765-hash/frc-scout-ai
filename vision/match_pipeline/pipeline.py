"""
End-to-end per-match pipeline with overlap.

Instead of running each STAGE across all matches (download-all, then preprocess-all,
...), this runs the whole chain PER MATCH, many matches in flight at once. So a
match that finishes downloading is preprocessed + tracked + scored while other
matches are still downloading -- nothing waits on a whole previous stage.

Chain per match:  download -> preprocess -> trajectory(GPU) -> scoring(OCR) -> auto
Each stage is skipped if its output already exists (idempotent). Put a stage name
in REDO to force it (e.g. REDO=["scoring","auto"] re-runs just those with new logic).

After every match is done, the aggregations run once: preferred_paths + compare_tba.

PyCharm: set REDO / WORKERS below, click Run.
"""

import glob
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PARENT)
VIDEOS = os.path.join(HERE, "videos")
RESULTS = os.path.join(HERE, "results")
TRUTH = os.path.join(HERE, "truth")

# ===========================================================================
# EDIT ME (PyCharm one-click).
# ===========================================================================
# Re-running IDENTITY across every current-geometry match after the teleop fix.
# identify.py now budgets OCR per TRACK instead of by a flat stride, and BOTH
# identify and stitch enforce team-uniqueness only between tracks alive AT THE
# SAME TIME. The old global "six teams, six tracks" cap left every later fragment
# unnamed however clearly its bumper was read. Measured on qm27: 11 -> 40 tracks
# identified, steal attribution 17% -> 54%.
# Trajectory is NOT redone -- this is identify + auto only, ~1.1h at WORKERS=5.
ONLY = [
    "2026hop_f1m2",
    "2026hop_f1m3",
    "2026hop_qm1",
    "2026hop_qm10",
    "2026hop_qm100",
    "2026hop_qm101",
    "2026hop_qm102",
    "2026hop_qm103",
    "2026hop_qm104",
    "2026hop_qm105",
    "2026hop_qm106",
    "2026hop_qm108",
    "2026hop_qm109",
    "2026hop_qm11",
    "2026hop_qm111",
    "2026hop_qm112",
    "2026hop_qm113",
    "2026hop_qm114",
    "2026hop_qm115",
    "2026hop_qm116",
    "2026hop_qm117",
    "2026hop_qm118",
    "2026hop_qm119",
    "2026hop_qm12",
    "2026hop_qm120",
    "2026hop_qm121",
    "2026hop_qm122",
    "2026hop_qm123",
    "2026hop_qm13",
    "2026hop_qm14",
    "2026hop_qm15",
    "2026hop_qm16",
    "2026hop_qm17",
    "2026hop_qm18",
    "2026hop_qm19",
    "2026hop_qm2",
    "2026hop_qm20",
    "2026hop_qm21",
    "2026hop_qm22",
    "2026hop_qm23",
    "2026hop_qm24",
    "2026hop_qm25",
    "2026hop_qm26",
    "2026hop_qm27",
    "2026hop_qm28",
    "2026hop_qm29",
    "2026hop_qm3",
    "2026hop_qm30",
    "2026hop_qm31",
    "2026hop_qm32",
    "2026hop_qm33",
    "2026hop_qm34",
    "2026hop_qm35",
    "2026hop_qm36",
    "2026hop_qm37",
    "2026hop_qm39",
    "2026hop_qm4",
    "2026hop_qm40",
    "2026hop_qm41",
    "2026hop_qm42",
    "2026hop_qm43",
    "2026hop_qm44",
    "2026hop_qm46",
    "2026hop_qm47",
    "2026hop_qm48",
    "2026hop_qm49",
    "2026hop_qm5",
    "2026hop_qm50",
    "2026hop_qm51",
    "2026hop_qm52",
    "2026hop_qm53",
    "2026hop_qm54",
    "2026hop_qm55",
    "2026hop_qm56",
    "2026hop_qm57",
    "2026hop_qm58",
    "2026hop_qm59",
    "2026hop_qm6",
    "2026hop_qm60",
    "2026hop_qm61",
    "2026hop_qm62",
    "2026hop_qm63",
    "2026hop_qm64",
    "2026hop_qm65",
    "2026hop_qm66",
    "2026hop_qm67",
    "2026hop_qm68",
    "2026hop_qm69",
    "2026hop_qm7",
    "2026hop_qm70",
    "2026hop_qm71",
    "2026hop_qm72",
    "2026hop_qm73",
    "2026hop_qm74",
    "2026hop_qm75",
    "2026hop_qm77",
    "2026hop_qm78",
    "2026hop_qm8",
    "2026hop_qm80",
    "2026hop_qm81",
    "2026hop_qm82",
    "2026hop_qm83",
    "2026hop_qm84",
    "2026hop_qm85",
    "2026hop_qm86",
    "2026hop_qm87",
    "2026hop_qm88",
    "2026hop_qm89",
    "2026hop_qm9",
    "2026hop_qm90",
    "2026hop_qm91",
    "2026hop_qm92"
]
REDO = ["identify", "auto"]  # stages to force re-run; [] = only fill in missing
WORKERS = 5            # matches in flight. ~4-5 if trajectory(GPU) will run;
                       # higher (8-10) is fine when only CPU stages (scoring/auto) redo
STRIDE = 2             # frame stride for trajectory + scoring
HEIGHT = 1080          # download max height (1080 = readable bumper numbers)
SKIP_NO_VIDEO = True   # skip matches with no local video instead of re-attempting the
                       # download. 11 of the 140 have a youtube_key that yt-dlp can no
                       # longer fetch ("Sign in to confirm you're not a bot"), and each
                       # retry burns a worker slot on a guaranteed failure. Set False
                       # once you have cookies wired into download_matches.py.
# ===========================================================================
STAGES = ["download", "preprocess", "trajectory", "scoring", "identify", "auto"]


def _weights():
    c = glob.glob(os.path.join(PARENT, "runs", "detect", "*", "weights", "best.pt"))
    if not c:
        raise SystemExit("No trained weights under runs/detect/*/weights/best.pt")
    return max(c, key=os.path.getmtime)


def process_match(match, redo, stride, height):
    """Run the whole chain for one match. Returns (match, [stage:status,...])."""
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(v, "1")
    sys.path.insert(0, HERE)
    sys.path.insert(0, PARENT)
    import cv2
    cv2.setNumThreads(1)

    redo = set(redo)
    vid = os.path.join(VIDEOS, f"{match}.mp4")
    outdir = os.path.join(RESULTS, match)
    os.makedirs(outdir, exist_ok=True)
    log = []
    try:
        # 1. download ------------------------------------------------------
        if not os.path.exists(vid) or "download" in redo:
            import download_matches as dl
            t = json.load(open(os.path.join(TRUTH, f"{match}.json")))
            yt = t.get("youtube_key")
            if not yt:
                return match, ["download:no-video"]
            _, st = dl.ytdlp_download(match, yt, height)
            log.append(f"download:{st}")

        # 2. preprocess (trim + crop + h264) -------------------------------
        import preprocess as pp
        need_split = not os.path.exists(os.path.join(VIDEOS, "revL", f"{match}.mp4"))
        h = pp._height(vid) if os.path.exists(vid) else 0
        stale = (h > pp.KEEP_BOTTOM) if not pp.MULTIVIEW else need_split
        if os.path.exists(vid) and (stale or "preprocess" in redo):
            _, st = pp.clean_one(match, pp.KEEP_BOTTOM, pp.PAD, None,
                                 force="preprocess" in redo)
            log.append(f"preprocess:{st.split('(')[0].strip()}")

        # 3. trajectory (GPU) ----------------------------------------------
        traj = os.path.join(outdir, "trajectory.csv")
        if not os.path.exists(traj) or "trajectory" in redo:
            import trajectories as tj
            r = tj.run(vid, outdir, _weights(), stride=stride, device="mps")
            log.append(f"trajectory:{r['tracks']}trk")

        # 4. scoring (OCR -> per-alliance + per-robot) ---------------------
        est = os.path.join(outdir, "estimate.json")
        if not os.path.exists(est) or "scoring" in redo:
            import scoring_estimator as se
            s = se.run(vid, outdir, start_frame=0, max_frames=None, ocr_every=10,
                       field_mid=None, k_slots=3, write_video=False, frame_stride=stride)
            sc = s.get("alliance_scored_from_board", {})
            log.append(f"scoring:b{sc.get('blue')}/r{sc.get('red')}")

        # 5. identity by READ bumper numbers (top + both side views) --------
        # NON-FATAL on purpose: identify has only ever been proven on qm27, and a
        # failure here must not cost us the auto stage -- auto_paths falls back to
        # rank order when identity.json is missing.
        idp = os.path.join(outdir, "identity.json")
        if not os.path.exists(idp) or "identify" in redo:
            try:
                import identify
                ident = identify.run(match)
                log.append(f"identify:{len(ident or {})}read")
            except Exception as e:
                log.append(f"identify:ERR:{e.__class__.__name__}")

        # 6. auto paths + classification -----------------------------------
        ap = os.path.join(outdir, "auto_paths.json")
        if not os.path.exists(ap) or "auto" in redo:
            import auto_paths
            o = auto_paths.run(match)
            log.append(f"auto:{len(o)}robots")

        return match, (log or ["skip:all-done"])
    except MemoryError:
        return match, log + ["FAIL:MemoryError"]
    except Exception as e:
        log.append(f"FAIL:{e.__class__.__name__}:{e}")
        return match, log + [traceback.format_exc(limit=1).strip().splitlines()[-1]]


def identity_summary():
    """How each robot in the auto book actually got its team number: READ off the
    bumper, or inferred from rank order. This is the honest scoreboard for
    identify.py -- rank order is the fallback, not a result."""
    rows = []
    for ap in sorted(glob.glob(os.path.join(RESULTS, "*", "auto_paths.json"))):
        match = os.path.basename(os.path.dirname(ap))
        try:
            teams = json.load(open(ap)).get("teams", {})
        except Exception:
            continue
        idp = os.path.join(RESULTS, match, "identity.json")
        read_tids = set()
        if os.path.exists(idp):
            try:
                read_tids = {int(k) for k, v in
                             json.load(open(idp)).get("tracks", {}).items() if v.get("team")}
            except Exception:
                pass
        # auto_paths records "source" now; fall back to the tid<->identity.json
        # join for auto_paths.json written before that field existed.
        if any("source" in o for o in teams.values()):
            read = sum(1 for o in teams.values() if o.get("source") == "read")
            rank = sum(1 for o in teams.values() if o.get("source") == "rank")
        else:
            read = sum(1 for o in teams.values() if o.get("tid") in read_tids)
            rank = sum(1 for o in teams.values() if o.get("team")) - read
        rows.append((match, read, rank, len(teams)))
    if not rows:
        print("\nno auto_paths.json yet -- nothing to summarise")
        return
    tot_read = sum(r[1] for r in rows)
    tot_rank = sum(r[2] for r in rows)
    tot_bots = sum(r[3] for r in rows)
    full = sum(1 for r in rows if r[1] == 6)
    none = sum(1 for r in rows if r[1] == 0)
    print(f"\nIDENTITY over {len(rows)} matches ({tot_bots} robots):")
    print(f"  by READ bumper number : {tot_read} ({100.0 * tot_read / max(tot_bots, 1):.1f}%)")
    print(f"  by rank-order fallback: {tot_rank} ({100.0 * tot_rank / max(tot_bots, 1):.1f}%)")
    print(f"  unresolved            : {tot_bots - tot_read - tot_rank}")
    print(f"  matches all 6 by read : {full}    matches 0 by read: {none}")
    worst = sorted(rows, key=lambda r: r[1])[:10]
    print("  weakest matches (read/robots): "
          + ", ".join(f"{m} {r}/{n}" for m, r, _, n in worst))
    with open(os.path.join(RESULTS, "identity_summary.csv"), "w") as f:
        f.write("match,by_read,by_rank,robots\n")
        for m, r, k, n in rows:
            f.write(f"{m},{r},{k},{n}\n")


def main():
    matches = sorted(os.path.splitext(os.path.basename(p))[0]
                     for p in glob.glob(os.path.join(TRUTH, "*.json")))
    if ONLY:
        matches = [m for m in matches if m in set(ONLY)]
    if SKIP_NO_VIDEO:
        have = [m for m in matches if os.path.exists(os.path.join(VIDEOS, f"{m}.mp4"))]
        missing = len(matches) - len(have)
        if missing:
            print(f"skipping {missing} match(es) with no local video "
                  f"(SKIP_NO_VIDEO=True)")
        matches = have
    print(f"{len(matches)} matches, {WORKERS} in flight, redo={REDO or 'none'}")
    ok = fail = 0
    t0 = time.time()
    # max_tasks_per_child=1 is NOT optional here. Each match loads a fresh YOLO
    # model twice (trajectories.run + identify.run) plus an EasyOCR reader, and on
    # MPS those are UNIFIED memory -- they count against the same RAM. Worker
    # processes are reused for ~26 matches each, nothing forces a release between
    # them, and the run climbed to 94GB on a 48GB machine and went to swap.
    # Recycling the process after every match hands the memory back to the OS.
    # Costs one model reload per match (~5s) against ~80s of real work. Worth it.
    with ProcessPoolExecutor(max_workers=WORKERS, max_tasks_per_child=1) as ex:
        futs = {ex.submit(process_match, m, REDO, STRIDE, HEIGHT): m for m in matches}
        for i, fut in enumerate(as_completed(futs), 1):
            m, log = fut.result()
            bad = any(s.startswith("FAIL") for s in log)
            fail += bad
            ok += not bad
            print(f"  [{i}/{len(matches)}] {m:16s} {'  '.join(log)}")
    print(f"\nchain done: {ok} ok, {fail} failed in {time.time()-t0:.0f}s")

    # aggregations (run once, after every match) ---------------------------
    try:
        identity_summary()
    except Exception as e:
        print("identity_summary:", e)
    try:
        import preferred_paths
        preferred_paths.run()
    except Exception as e:
        print("preferred_paths:", e)
    try:
        import compare_tba
        sys.argv = ["compare_tba"]
        compare_tba.main()
    except SystemExit:
        pass
    except Exception as e:
        print("compare_tba:", e)


if __name__ == "__main__":
    main()
