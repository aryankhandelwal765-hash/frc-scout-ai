"""
Baseline for occlusion-aware tracking work: how fragmented is a match AFTER stitch.py?

    python stitch_baseline.py --matches 30

Per alliance there are 3 robots, so ideal = 3 long chains. Reports, per match:
  tracks          raw track ids with >= min_frames frames
  chains          after stitch.chains (the production linker)
  long chains     chains lasting >= 10 s
  top3 coverage   share of an alliance's tracked frames held by its 3 longest chains
                  (1.0 = every detection belongs to one of three robots)
"""

import argparse
import glob
import os
import statistics as st

import auto_paths as ap
import stitch

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matches", type=int, default=30)
    p.add_argument("--long-sec", type=float, default=10.0)
    args = p.parse_args()

    names = sorted(
        os.path.basename(os.path.dirname(f))
        for f in glob.glob(os.path.join(HERE, "results", "2026hop_qm*", "trajectory.csv"))
    )
    step = max(1, len(names) // args.matches)
    names = names[::step][: args.matches]
    fps = 60.0
    rows = []
    for m in names:
        try:
            tracks = ap._load_trajectory(m)
            al_of = {t: ap._voted_alliance(m, t, r) for t, r in tracks.items()}
            last = max(r[-1][0] for r in tracks.values())
            ch = stitch.chains(m, tracks, al_of, fps=fps, window=(0, last))
        except Exception as e:  # missing video / odd match: skip, report count
            print(m, "skipped:", repr(e)[:80])
            continue
        n_tr = sum(1 for r in tracks.values() if len(r) >= 3)
        by_al = {"blue": [], "red": []}
        for c in ch:
            frames = sum(len(tracks[t]) for t in c)
            by_al.setdefault(al_of.get(c[0]), []).append(frames)
        cov = []
        for al in ("blue", "red"):
            fr = sorted(by_al.get(al, []), reverse=True)
            if fr:
                cov.append(sum(fr[:3]) / sum(fr))
        long_n = sum(1 for c in ch if sum(len(tracks[t]) for t in c) >= args.long_sec * fps)
        rows.append((m, n_tr, len(ch), long_n, st.mean(cov) if cov else 0))
    print(f"{'match':22s} tracks chains long(>={args.long_sec:.0f}s) top3-coverage")
    for r in rows:
        print(f"{r[0]:22s} {r[1]:6d} {r[2]:6d} {r[3]:8d}      {r[4]:.2f}")
    print(
        f"\nMEAN over {len(rows)} matches: tracks {st.mean(r[1] for r in rows):.0f}  "
        f"chains {st.mean(r[2] for r in rows):.0f}  long chains {st.mean(r[3] for r in rows):.1f} "
        f"(ideal 6)  top3 coverage {st.mean(r[4] for r in rows):.2f}"
    )


if __name__ == "__main__":
    main()
