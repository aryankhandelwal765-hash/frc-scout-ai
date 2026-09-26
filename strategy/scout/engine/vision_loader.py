"""Load vision scouting output into strategy.

Vision produces per-match, per-robot measurements the TBA aggregates miss:
  - estimate.json     est_balls_scored per robot, activity heatmap value
  - identity.json     track_id → team (with confidence + source: read/elim/rank)
  - auto_paths.json   per-station auto trajectory in field coords

This module reads those artefacts, joins them by team number, and produces
per-team vision stats aggregated across matches. Cached in SQLite alongside
TBA / prediction / synthesis outputs.

Vision data is *complementary* to TBA — OPR is aggregate/offensive-only, vision
gives verified per-robot activity + specific match behaviors the LLM can cite.
"""

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from scout.db import cache_get, cache_set

# vision output root — mirrors the SCOUTING_ROOT in api/main.py.
VISION_ROOT = (
    Path(__file__).resolve().parent.parent.parent.parent / "vision" / "match_pipeline" / "results"
)

# Cache TTL — vision output rarely changes, but bump if user re-runs the pipeline.
TTL_VISION = 6 * 3600  # 6 h


def _parse_team(raw: str | int | None) -> int | None:
    """Parse '7287' or 'frc7287' or 7287 → 7287. None on failure."""
    if raw is None:
        return None
    s = str(raw).lower().replace("frc", "").strip()
    try:
        return int(s)
    except ValueError:
        return None


def _load_match(match_dir: Path) -> dict | None:
    """Read one match's artefacts and return {match_key, teams: {team: stats}}.

    Returns None if the match has no usable data (missing estimate + identity).
    """
    estimate_path = match_dir / "estimate.json"
    identity_path = match_dir / "identity.json"
    if not estimate_path.exists() or not identity_path.exists():
        return None
    try:
        estimate = json.loads(estimate_path.read_text())
        identity = json.loads(identity_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    # Map (alliance, slot) → team via identity.json's per-track resolution.
    # identity.tracks: {track_id: {alliance, team, confidence, source}}
    # estimate.robots: [{alliance, slot, activity, hits, pos, est_balls_scored}]
    # No direct slot↔track linkage, so we approximate: for each alliance, the
    # top-3-confidence tracks map to slots 0/1/2 in order of confidence descending.
    alliance_tracks = defaultdict(list)
    for _tid, info in identity.get("tracks", {}).items():
        team = _parse_team(info.get("team"))
        if team is None:
            continue
        alliance_tracks[info.get("alliance", "?")].append(
            {
                "team": team,
                "confidence": info.get("confidence", 0.0),
                "source": info.get("source", "?"),
            }
        )

    # Highest-confidence tracks first — those are the "sure" identity reads.
    for alliance, tracks in alliance_tracks.items():
        tracks.sort(key=lambda t: t["confidence"], reverse=True)

    teams: dict[int, dict] = {}
    for robot in estimate.get("robots", []):
        alliance = robot.get("alliance")
        slot = robot.get("slot", 0)
        candidates = alliance_tracks.get(alliance, [])
        if slot >= len(candidates):
            continue
        team = candidates[slot]["team"]
        teams[team] = {
            "match_key": match_dir.name,
            "alliance": alliance,
            "est_balls_scored": robot.get("est_balls_scored", 0.0),
            "activity": robot.get("activity", 0),
            "hits": robot.get("hits", 0),
            "position": robot.get("pos", [0, 0]),
            "id_confidence": candidates[slot]["confidence"],
            "id_source": candidates[slot]["source"],
        }

    return {
        "match_key": match_dir.name,
        "duration_sec": estimate.get("duration_sec", 0),
        "alliance_scored": estimate.get("alliance_scored_from_board", {}),
        "teams": teams,
    }


def get_team_vision(team: int, event_prefix: str = "2026hop") -> dict | None:
    """Aggregate vision stats for one team across all matches in an event.

    Returns None if the team has no vision data. Cached 6h.
    """
    cache_key = f"vision:team:{team}:{event_prefix}"
    hit = cache_get(cache_key)
    if hit is not None:
        return hit

    if not VISION_ROOT.exists():
        return None

    matches_seen = []
    balls_scored = []
    activities = []
    positions = []
    alliances = []
    id_confidences = []

    for match_dir in VISION_ROOT.iterdir():
        if not match_dir.is_dir() or not match_dir.name.startswith(event_prefix):
            continue
        data = _load_match(match_dir)
        if data is None or team not in data["teams"]:
            continue
        t = data["teams"][team]
        matches_seen.append(match_dir.name)
        balls_scored.append(t["est_balls_scored"])
        activities.append(t["activity"])
        positions.append(t["position"])
        alliances.append(t["alliance"])
        id_confidences.append(t["id_confidence"])

    if not matches_seen:
        cache_set(cache_key, {"team": team, "matches_count": 0}, TTL_VISION)
        return {"team": team, "matches_count": 0}

    avg_balls = sum(balls_scored) / len(balls_scored)
    max_balls = max(balls_scored)
    avg_activity = sum(activities) / len(activities)
    # Position "center of gravity" — average field-pixel position across matches.
    avg_x = sum(p[0] for p in positions) / len(positions)
    avg_y = sum(p[1] for p in positions) / len(positions)
    avg_confidence = sum(id_confidences) / len(id_confidences)

    result = {
        "team": team,
        "matches_count": len(matches_seen),
        "matches_seen": matches_seen,
        "avg_balls_scored": round(avg_balls, 1),
        "max_balls_scored": round(max_balls, 1),
        "avg_activity": round(avg_activity, 0),
        "avg_position": [round(avg_x, 0), round(avg_y, 0)],
        "id_confidence": round(avg_confidence, 2),
    }
    cache_set(cache_key, result, TTL_VISION)
    return result


def get_match_vision(match_key: str) -> dict | None:
    """Return one match's full vision output (all six robots). Cached 6h."""
    cache_key = f"vision:match:{match_key}"
    hit = cache_get(cache_key)
    if hit is not None:
        return hit

    match_dir = VISION_ROOT / match_key
    if not match_dir.exists():
        return None
    data = _load_match(match_dir)
    if data is None:
        return None
    # Cast team keys to str for JSON compatibility.
    data["teams"] = {str(k): v for k, v in data["teams"].items()}
    cache_set(cache_key, data, TTL_VISION)
    return data


def get_team_auto_routes(team: int, event_prefix: str = "2026hop") -> list[dict]:
    """Return a team's distinct auto routines across all matches in an event.

    Reads auto_interp.json (written by interpret_auto.py). Each entry has:
      name, description, start_side, delayed, zone_seq, match, station.
    Sorted by frequency (most-run routine first). Cached 6h.
    """
    cache_key = f"vision:auto_routes:{team}:{event_prefix}"
    hit = cache_get(cache_key)
    if hit is not None:
        return hit

    if not VISION_ROOT.exists():
        cache_set(cache_key, [], TTL_VISION)
        return []

    from collections import Counter

    raw_entries: list[dict] = []
    for match_dir in VISION_ROOT.iterdir():
        if not match_dir.is_dir() or not match_dir.name.startswith(event_prefix):
            continue
        interp_path = match_dir / "auto_interp.json"
        if not interp_path.exists():
            continue
        try:
            data = json.loads(interp_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for station, info in data.items():
            t = info.get("team", "")
            t_num = None
            try:
                t_num = int(str(t).lower().replace("frc", ""))
            except (ValueError, AttributeError):
                pass
            if t_num != team:
                continue
            if "name" not in info or "error" in info:
                continue
            raw_entries.append(
                {
                    "match": match_dir.name,
                    "station": station,
                    "name": info.get("name", "Unknown"),
                    "description": info.get("description", ""),
                    "start_side": info.get("start_side", ""),
                    "delayed": info.get("delayed", False),
                    "zone_seq": info.get("zone_seq", []),
                }
            )

    if not raw_entries:
        cache_set(cache_key, [], TTL_VISION)
        return []

    # Group by routine name, pick most-common descriptions
    name_groups: dict[str, list] = {}
    for e in raw_entries:
        name_groups.setdefault(e["name"], []).append(e)

    result = []
    for name, entries in sorted(name_groups.items(), key=lambda kv: len(kv[1]), reverse=True):
        # Modal description
        desc_counts = Counter(e["description"] for e in entries if e["description"])
        modal_desc = desc_counts.most_common(1)[0][0] if desc_counts else ""
        zone_seqs = [e["zone_seq"] for e in entries if e["zone_seq"]]
        modal_zones = zone_seqs[0] if zone_seqs else []
        result.append(
            {
                "name": name,
                "count": len(entries),
                "description": modal_desc,
                "start_side": entries[0]["start_side"],
                "delayed": entries[0]["delayed"],
                "zone_seq": modal_zones,
                "matches": [e["match"] for e in entries],
            }
        )

    cache_set(cache_key, result, TTL_VISION)
    return result


def build_scouting_notes(
    red_teams: list[int], blue_teams: list[int], event_prefix: str = "2026hop"
) -> str:
    """Auto-generate scouting notes from vision stats for both alliances.

    Fed into MatchSetup's `notes` field as a starting point — user can edit/append.
    Returns a plain-text summary; empty string if no vision data available.
    """
    lines = []
    lines.append("--- Vision-derived observations (edit/append) ---")

    for side_name, team_list in (("RED", red_teams), ("BLUE", blue_teams)):
        lines.append(f"\n{side_name}:")
        for team in team_list:
            v = get_team_vision(team, event_prefix)
            if v is None or v.get("matches_count", 0) == 0:
                lines.append(f"  {team}: no vision data.")
                continue
            n = v["matches_count"]
            avg = v["avg_balls_scored"]
            peak = v["max_balls_scored"]
            confidence_note = ""
            if v["id_confidence"] < 0.7:
                confidence_note = " (id uncertain — bumper OCR low-confidence)"
            lines.append(
                f"  {team}: {avg} avg balls/match (peak {peak}) across {n} matches{confidence_note}"
            )
            # Auto route summary from interpret_auto.py output
            routes = get_team_auto_routes(team, event_prefix)
            if routes:
                total_runs = sum(r["count"] for r in routes)
                route_parts = []
                for r in routes[:3]:  # top 3 routines
                    delayed_tag = " (delayed start)" if r["delayed"] else ""
                    route_parts.append(f"{r['name']}{delayed_tag} ×{r['count']}/{total_runs}")
                lines.append(f"    AUTO: {' | '.join(route_parts)}")
                # Zone sequence for primary routine
                if routes[0]["zone_seq"]:
                    zs = " → ".join(routes[0]["zone_seq"])
                    lines.append(f"    Primary route zones: {zs}")

    return "\n".join(lines)
