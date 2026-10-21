"""
Team ratings learned from every past match, game-agnostic (only who won and by how much).

    .venv/bin/python -m scout.training.history_elo --backtest     # tune and report, writes nothing
    .venv/bin/python -m scout.training.history_elo --write        # data/history/team_ratings.json

Why not the existing EPA diff: a rating that only uses results does not care what the game's
scoring was, so one model learns from ten seasons instead of one event, and a team carries its
history into the next season (which is what a first-week prediction needs).

Method: Elo over all played qualification and playoff matches in time order, alliance rating =
mean of its teams, update shared equally. A margin multiplier scales surprise by the margin
relative to that year's average alliance score. At each season boundary a team keeps CARRY of its
rating and the rest regresses to the mean. Every prediction is made BEFORE its own update, so the
backtest has no leakage.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

HISTORY = Path(__file__).resolve().parent.parent / "data" / "history"
BASE = 1500.0


def load_year(year):
    rows = []
    for p in sorted((HISTORY / str(year)).glob("*.json")):
        rows += json.loads(p.read_text())
    rows = [
        m
        for m in rows
        if m["time"] and len(m["red"]["teams"]) >= 2 and len(m["blue"]["teams"]) >= 2
    ]
    rows.sort(key=lambda m: (m["time"], m["key"]))
    return rows


def mean_score(rows):
    tot = [m["red"]["score"] + m["blue"]["score"] for m in rows]
    return (sum(tot) / (2 * len(tot))) if tot else 1.0


class Elo:
    def __init__(self, k=16.0, carry=0.7, margin=1.0, new_team=BASE):
        self.k, self.carry, self.margin, self.new_team = k, carry, margin, new_team
        self.r = {}
        self.games = defaultdict(int)  # per-season games played, to find early-event matches

    def rating(self, team):
        return self.r.get(team, self.new_team)

    def alliance(self, teams):
        return sum(self.rating(t) for t in teams) / len(teams)

    def new_season(self):
        self.r = {t: BASE + self.carry * (v - BASE) for t, v in self.r.items()}
        self.games = defaultdict(int)

    def diff(self, m):
        return self.alliance(m["red"]["teams"]) - self.alliance(m["blue"]["teams"])

    def p_red(self, m, scale=400.0):
        return 1.0 / (1.0 + 10 ** (-self.diff(m) / scale))

    def update(self, m, avg):
        p = self.p_red(m)
        diff = m["red"]["score"] - m["blue"]["score"]
        s = 1.0 if diff > 0 else 0.0 if diff < 0 else 0.5
        mult = 1.0 + self.margin * min(abs(diff) / max(avg, 1.0), 1.0)
        delta = self.k * mult * (s - p)
        for side, sign in (("red", 1), ("blue", -1)):
            for t in m[side]["teams"]:
                self.r[t] = self.rating(t) + sign * delta
        for side in ("red", "blue"):
            for t in m[side]["teams"]:
                self.games[t] += 1


def run(years_data, **params):
    """Walk the years in order. Returns (elo, preds): (year, rating_diff, red_won, early)."""
    elo, preds = Elo(**params), []
    for i, (year, rows) in enumerate(years_data):
        if i:
            elo.new_season()
        avg = mean_score(rows)
        for m in rows:
            diff = m["red"]["score"] - m["blue"]["score"]
            if diff != 0:
                early = min(elo.games[t] for side in ("red", "blue") for t in m[side]["teams"]) < 3
                preds.append((year, elo.diff(m), diff > 0, early))
            elo.update(m, avg)
    return elo, preds


def logistic(d, scale):
    return 1.0 / (1.0 + 10 ** (-d / scale))


def fit_scale(preds, years):
    """Logistic scale that minimises log loss on `years` (the seasons BEFORE the test years)."""
    rows = [p for p in preds if p[0] in years]
    best = None
    for scale in range(100, 2001, 50):
        ll = -sum(
            math.log(max(min(logistic(d, scale) if w else 1 - logistic(d, scale), 1 - 1e-9), 1e-9))
            for _y, d, w, _e in rows
        ) / len(rows)
        if best is None or ll < best[0]:
            best = (ll, scale)
    return best[1]


def score(preds, years, scale):
    sel = [(y, logistic(d, scale), w, e) for y, d, w, e in preds if y in years]
    if not sel:
        return None

    def ll(rows):
        return -sum(
            math.log(max(min(p if w else 1 - p, 1 - 1e-9), 1e-9)) for _y, p, w, _e in rows
        ) / len(rows)

    def acc(rows):
        return sum((p > 0.5) == w for _y, p, w, _e in rows) / len(rows)

    early = [p for p in sel if p[3]]
    return {
        "n": len(sel),
        "logloss": round(ll(sel), 4),
        "acc": round(acc(sel), 4),
        "early_n": len(early),
        "early_logloss": round(ll(early), 4) if early else None,
        "early_acc": round(acc(early), 4) if early else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--k", type=float, default=32.0)
    ap.add_argument("--carry", type=float, default=0.85)
    ap.add_argument("--margin", type=float, default=4.0)
    a = ap.parse_args()

    years = sorted(int(p.name) for p in HISTORY.iterdir() if p.is_dir() and p.name.isdigit())
    data = [(y, load_year(y)) for y in years]
    data = [(y, r) for y, r in data if r]
    print("years:", {y: len(r) for y, r in data})
    test_years = {y for y, _ in data if y >= max(years) - 2}

    if a.backtest:
        print(f"no information: logloss 0.6931   (test years {sorted(test_years)})")
        best = None
        fit_years = {y for y, _ in data if y < min(test_years)}
        for carry in (0.0, 0.85, 0.92, 0.97):
            for k in (16.0, 32.0, 64.0):
                for margin in (2.0, 4.0):
                    _e, preds = run(data, k=k, carry=carry, margin=margin)
                    scale = fit_scale(preds, fit_years)
                    sc = score(preds, test_years, scale)
                    sc["scale"] = scale
                    print(f"carry {carry:.2f} K {k:3.0f} margin {margin:.0f}: {sc}", flush=True)
                    if best is None or sc["logloss"] < best[0]["logloss"]:
                        best = (sc, dict(k=k, carry=carry, margin=margin))
        print("BEST", best)
        return

    elo, preds = run(data, k=a.k, carry=a.carry, margin=a.margin)
    fit_years = {y for y, _ in data if y < min(test_years)}
    scale = fit_scale(preds, fit_years)
    print("test:", score(preds, test_years, scale), "scale", scale)
    if a.write:
        out = HISTORY / "team_ratings.json"
        ratings = {t.replace("frc", ""): round(v, 1) for t, v in sorted(elo.r.items())}
        out.write_text(
            json.dumps(
                {
                    "through_year": max(y for y, _ in data),
                    "params": {"k": a.k, "carry": a.carry, "margin": a.margin},
                    "logistic_scale": scale,
                    "ratings": ratings,
                },
                separators=(",", ":"),
            )
        )
        print(f"wrote {out} ({len(ratings)} teams)")


if __name__ == "__main__":
    main()
