"""
Batch-download every match video for a TBA event, and dump TBA "truth" alongside.

For event 2026hop (Hopper division, 2026):
  1. Pull all matches from TBA (includes YouTube video keys + score_breakdown).
  2. Save per-match TBA truth to truth/<match_key>.json and a combined truth.json.
  3. Download each match's YouTube video to videos/<match_key>.mp4 (<=720p),
     several in parallel. Skips files already present.

Usage:
  python download_matches.py --event 2026hop
  python download_matches.py --event 2026hop --height 720 --parallel 5
  python download_matches.py --event 2026hop --list-only     # no downloads

The vision batch (batch_run.py) then consumes videos/, and the checker
(compare_tba.py) consumes truth/.
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # so `import tba` works from PyCharm's Run button
import tba

VIDEOS = os.path.join(HERE, "videos")
TRUTH = os.path.join(HERE, "truth")

# ===========================================================================
# EDIT ME (PyCharm: just click Run — no terminal args needed). CLI args, if
# given, override these. Defaults = the small test.
# ===========================================================================
EVENT = "2026hop"  # TBA event key
LIMIT = None  # None = ALL matches; set N for a small test
QUALS_ONLY = False  # True = skip playoff matches
HEIGHT = 1080  # max video height (p). 1080 makes bumper NUMBERS
# readable (720 does not) -- that's what identity relies on.
PARALLEL = 9  # concurrent downloads
LIST_ONLY = False  # True = save truth + print list, download nothing
# ===========================================================================


def match_truth(m):
    """Distill a TBA match object into the fields the checker needs (keep the
    raw score_breakdown so we can pick the right game-piece field later)."""
    al = m.get("alliances") or {}
    red = al.get("red") or {}
    blue = al.get("blue") or {}
    return {
        "key": m["key"],
        "comp_level": m.get("comp_level"),
        "match_number": m.get("match_number"),
        "set_number": m.get("set_number"),
        "youtube_key": tba.youtube_key(m),
        "winning_alliance": m.get("winning_alliance"),
        "red_teams": red.get("team_keys"),
        "blue_teams": blue.get("team_keys"),
        "red_score": red.get("score"),
        "blue_score": blue.get("score"),
        "score_breakdown": m.get("score_breakdown"),  # raw, game-specific
    }


def ytdlp_download(match_key, yt_key, height):
    out_tmpl = os.path.join(VIDEOS, f"{match_key}.%(ext)s")
    final_mp4 = os.path.join(VIDEOS, f"{match_key}.mp4")
    if os.path.exists(final_mp4) and os.path.getsize(final_mp4) > 0:
        return match_key, "skip (exists)"
    cmd = [
        "yt-dlp",
        "-f",
        f"bv*[height<={height}]+ba/b[height<={height}]/b",
        "--merge-output-format",
        "mp4",
        "-o",
        out_tmpl,
        "--no-playlist",
        "--no-progress",
        "--quiet",
        "--retries",
        "5",
        "--fragment-retries",
        "5",
        f"https://www.youtube.com/watch?v={yt_key}",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "").strip().splitlines()
        return match_key, "FAIL: " + (tail[-1] if tail else "unknown")
    size = os.path.getsize(final_mp4) if os.path.exists(final_mp4) else 0
    return match_key, f"ok ({size // (1024 * 1024)} MB)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--event", default=EVENT, help="TBA event key")
    p.add_argument("--key", help="TBA auth key (else TBA_AUTH_KEY / .tba_key)")
    p.add_argument("--height", type=int, default=HEIGHT, help="max video height")
    p.add_argument("--parallel", type=int, default=PARALLEL, help="concurrent downloads")
    p.add_argument("--limit", type=int, default=LIMIT, help="only download first N videos")
    p.add_argument(
        "--quals-only", action="store_true", default=QUALS_ONLY, help="skip playoff matches"
    )
    p.add_argument(
        "--list-only", action="store_true", default=LIST_ONLY, help="fetch+save truth, no downloads"
    )
    args = p.parse_args()

    os.makedirs(VIDEOS, exist_ok=True)
    os.makedirs(TRUTH, exist_ok=True)
    key = tba.load_key(args.key)

    print(f"Fetching matches for {args.event} ...")
    matches = tba.event_matches(args.event, key)
    matches.sort(key=tba.match_sort_key)
    print(f"  {len(matches)} matches total")

    truths = [match_truth(m) for m in matches]
    with open(os.path.join(HERE, "truth.json"), "w") as f:
        json.dump(truths, f, indent=2)
    for t in truths:
        with open(os.path.join(TRUTH, f"{t['key']}.json"), "w") as f:
            json.dump(t, f, indent=2)

    with_video = [t for t in truths if t["youtube_key"]]
    if args.quals_only:
        with_video = [t for t in with_video if t["comp_level"] == "qm"]
    print(
        f"  {len(with_video)} matches have a YouTube video"
        + (" (quals only)" if args.quals_only else "")
    )
    if args.limit:
        with_video = with_video[: args.limit]
        print(f"  limiting to first {len(with_video)} (--limit)")
    if args.list_only:
        for t in with_video:
            print(f"    {t['key']:16s} yt={t['youtube_key']}")
        return

    print(f"Downloading {len(with_video)} videos (<= {args.height}p, {args.parallel} parallel) ...")
    ok = fail = skip = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = {
            ex.submit(ytdlp_download, t["key"], t["youtube_key"], args.height): t
            for t in with_video
        }
        for fut in as_completed(futs):
            mk, status = fut.result()
            print(f"    {mk:16s} {status}")
            if status.startswith("ok"):
                ok += 1
            elif status.startswith("skip"):
                skip += 1
            else:
                fail += 1
    print(f"\nDone: {ok} downloaded, {skip} skipped, {fail} failed. Videos in {VIDEOS}")


if __name__ == "__main__":
    main()
