#!/usr/bin/env python3
"""Bolão Layer 1c: aggregate elo.csv + specific.csv into odds.csv.

Pipeline position:
    1-fetch-elo.py        -> resource/<date>/elo.csv
    Claude context skill  -> resource/<date>/specific.csv   (pasted manually)
    this script           -> resource/<date>/odds.csv
    2-probabilities-to-score.py reads odds.csv

Model: adjusted Elo difference -> win expectancy We = 1 / (1 + 10^(-dr/400)),
then solve for Poisson goal rates (lam_home, lam_away) constrained by
lam_home + lam_away = total_goals and P(home) + 0.5 * P(draw) = We.
The resulting outcome probabilities are written as margin-free decimal odds.

specific.csv format (header required; empty numeric fields take defaults):
    home_team,away_team,stage,elo_adj_home,elo_adj_away,home_adv_elo,total_goals,rho,notes
    Mexico,South Africa,group,0,-25,75,2.9,-0.05,SA missing first-choice GK; Azteca crowd

stage is one of: group, r32, r16, qf, sf, third, final (default group); it is
passed through to odds.csv for the Layer 3 stage multiplier.

Usage:
    python 3-aggregate-odds.py                  # resource/<today>/
    python 3-aggregate-odds.py --date 20260611

Zero dependencies: standard library only.
"""

import argparse
import csv
import difflib
import math
import sys
import unicodedata
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

MAX_GOALS = 10
DEFAULT_TOTAL_GOALS = 2.9
DEFAULT_RHO = -0.05

# Mismatch goal model: a lopsided Elo gap means the favourite piles on goals that
# a fixed tempo-only total_goals cannot express (the old 3.2 ceiling made blowouts
# impossible). Add goals proportional to the gap; the win-expectancy split below
# then hands most of those extra goals to the favourite.
GAP_GOALS_PER_ELO = 0.0025   # +1.0 expected goal per 400 Elo of mismatch
TOTAL_GOALS_CAP = 5.0        # effective total never exceeds this

ALLOWED_STAGES = {"group", "r32", "r16", "qf", "sf", "third", "final"}


@dataclass(frozen=True)
class Fixture:
    home_team: str
    away_team: str
    stage: str
    elo_adj_home: float
    elo_adj_away: float
    home_adv_elo: float
    total_goals: float
    rho: float
    notes: str


def poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * lam ** k / math.factorial(k)


def score_matrix(lam_home: float, lam_away: float, rho: float) -> list[list[float]]:
    """Same model as 2-probabilities-to-score.py: independent Poisson with
    Dixon-Coles low-score correction, kept identical so the layers agree."""
    home_pmf = [poisson_pmf(i, lam_home) for i in range(MAX_GOALS + 1)]
    away_pmf = [poisson_pmf(j, lam_away) for j in range(MAX_GOALS + 1)]
    grid = [[home_pmf[i] * away_pmf[j] for j in range(MAX_GOALS + 1)]
            for i in range(MAX_GOALS + 1)]
    if rho != 0.0:
        grid[0][0] *= 1.0 - lam_home * lam_away * rho
        grid[0][1] *= 1.0 + lam_home * rho
        grid[1][0] *= 1.0 + lam_away * rho
        grid[1][1] *= 1.0 - rho
    total = sum(sum(row) for row in grid)
    return [[p / total for p in row] for row in grid]


def outcome_probs(grid: list[list[float]]) -> tuple[float, float, float]:
    home = sum(grid[i][j] for i in range(MAX_GOALS + 1) for j in range(MAX_GOALS + 1) if i > j)
    draw = sum(grid[k][k] for k in range(MAX_GOALS + 1))
    return home, draw, 1.0 - home - draw


def win_expectancy_from_elo(elo_diff: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))


def mismatch_goal_bonus(elo_diff: float) -> float:
    """Extra expected goals contributed by a one-sided fixture (GAP_GOALS_PER_ELO)."""
    return abs(elo_diff) * GAP_GOALS_PER_ELO


def solve_lambdas(win_expectancy: float, total_goals: float, rho: float) -> tuple[float, float]:
    """Bisection on lam_home: with total fixed, We rises monotonically in lam_home."""
    lo, hi = 0.05, total_goals - 0.05

    def we(lam_home: float) -> float:
        ph, pd, _ = outcome_probs(score_matrix(lam_home, total_goals - lam_home, rho))
        return ph + 0.5 * pd

    target = min(max(win_expectancy, we(lo)), we(hi))  # clamp to achievable range
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if we(mid) < target:
            lo = mid
        else:
            hi = mid
    lam_home = (lo + hi) / 2.0
    return lam_home, total_goals - lam_home


def normalise(name: str) -> str:
    decomposed = unicodedata.normalize("NFKD", name)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold().strip()


def load_elo(path: Path) -> dict[str, float]:
    """Maps normalised team name AND team code to rating."""
    ratings: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ratings[normalise(row["team"])] = float(row["rating"])
            ratings[normalise(row["code"])] = float(row["rating"])
    if not ratings:
        sys.exit(f"{path}: no teams found")
    return ratings


def resolve_rating(team: str, ratings: dict[str, float]) -> float:
    key = normalise(team)
    if key in ratings:
        return ratings[key]
    suggestions = difflib.get_close_matches(key, ratings.keys(), n=3, cutoff=0.6)
    hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
    sys.exit(f"Team '{team}' not found in elo.csv.{hint}")


def parse_float(raw: str | None, default: float) -> float:
    raw = (raw or "").strip()
    return float(raw) if raw else default


def load_fixtures(path: Path) -> list[Fixture]:
    fixtures: list[Fixture] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"home_team", "away_team"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            sys.exit(f"specific.csv is missing columns: {', '.join(sorted(missing))}")
        for line_number, row in enumerate(reader, start=2):
            try:
                stage = (row.get("stage") or "group").strip().casefold()
                if stage not in ALLOWED_STAGES:
                    sys.exit(f"{path}, line {line_number}: unknown stage '{stage}' "
                             f"(allowed: {', '.join(sorted(ALLOWED_STAGES))})")
                fixtures.append(Fixture(
                    home_team=row["home_team"].strip(),
                    away_team=row["away_team"].strip(),
                    stage=stage,
                    elo_adj_home=parse_float(row.get("elo_adj_home"), 0.0),
                    elo_adj_away=parse_float(row.get("elo_adj_away"), 0.0),
                    home_adv_elo=parse_float(row.get("home_adv_elo"), 0.0),
                    total_goals=parse_float(row.get("total_goals"), DEFAULT_TOTAL_GOALS),
                    rho=parse_float(row.get("rho"), DEFAULT_RHO),
                    notes=(row.get("notes") or "").strip(),
                ))
            except ValueError as error:
                sys.exit(f"{path}, line {line_number}: {error}")
    if not fixtures:
        sys.exit(f"{path}: no fixture rows found")
    return fixtures


def resource_dir(date_str: str) -> Path:
    relative = Path("resource") / date_str
    project_root = Path(__file__).resolve().parent.parent
    for candidate in (project_root / relative, Path.cwd() / relative):
        if candidate.exists():
            return candidate
    return project_root / relative


def main() -> None:
    parser = argparse.ArgumentParser(description="elo.csv + specific.csv -> odds.csv")
    date_to_parse = date.today() #- timedelta(days=1)
    parser.add_argument("--date", default=date_to_parse.strftime("%Y%m%d"),
                        help="matchday folder as YYYYMMDD (default: today)")
    args = parser.parse_args()

    folder = resource_dir(args.date)
    elo_path, specific_path, out_path = (folder / "elo.csv", folder / "specific.csv",
                                         folder / "odds.csv")
    for path in (elo_path, specific_path):
        if not path.exists():
            sys.exit(f"Missing input: {path}")

    ratings = load_elo(elo_path)
    fixtures = load_fixtures(specific_path)

    rows: list[list[str]] = []
    for fixture in fixtures:
        elo_home = resolve_rating(fixture.home_team, ratings) + fixture.elo_adj_home
        elo_away = resolve_rating(fixture.away_team, ratings) + fixture.elo_adj_away
        elo_diff = elo_home + fixture.home_adv_elo - elo_away
        target_we = win_expectancy_from_elo(elo_diff)

        effective_total = min(fixture.total_goals + mismatch_goal_bonus(elo_diff),
                              TOTAL_GOALS_CAP)
        lam_home, lam_away = solve_lambdas(target_we, effective_total, fixture.rho)
        prob_home, prob_draw, prob_away = outcome_probs(
            score_matrix(lam_home, lam_away, fixture.rho))

        odds = tuple(1.0 / max(p, 1e-6) for p in (prob_home, prob_draw, prob_away))
        rows.append([fixture.home_team, fixture.away_team,
                     f"{odds[0]:.3f}", f"{odds[1]:.3f}", f"{odds[2]:.3f}",
                     f"{fixture.rho:.2f}", fixture.stage])

        print(f"{fixture.home_team} vs {fixture.away_team}: "
              f"Elo diff {elo_diff:+.0f} -> We {target_we:.1%}  "
              f"H {prob_home:.1%} / D {prob_draw:.1%} / A {prob_away:.1%}  "
              f"(λ {lam_home:.2f}-{lam_away:.2f}, total {effective_total:.2f} "
              f"= base {fixture.total_goals:.1f} + gap {mismatch_goal_bonus(elo_diff):.2f})"
              + (f"  [{fixture.notes}]" if fixture.notes else ""))

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["home_team", "away_team", "odds_home", "odds_draw",
                         "odds_away", "rho", "stage"])
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} matches to {out_path}")


if __name__ == "__main__":
    main()
