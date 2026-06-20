#!/usr/bin/env python3
"""Bolão Layer 3: rank score predictions by expected points, not by likelihood.

Reads resource/<YYYYMMDD>/odds.csv (the Layer 1 output, including the stage
column) and for every match ranks all candidate scorelines by expected points
under the bolão tier table:

    exact score 25 · correct winner same goal difference 18 · correct winner
    other margin 12 · correct draw any score 15 · miss 0
    Only the highest tier counts; the stage multiplier applies to ALL tiers.

Knockout rules (per the app): the result is judged AFTER extra time including
penalties, so a knockout match always has a winner and draw predictions score 0.
Model assumptions for knockouts, tune the constants below if reality disagrees:
  - a 90' draw goes to a 30' extra time modelled as Poisson with the same team
    rates scaled by ET_RATE (30/90 minutes x 0.75 tempo = 0.25);
  - if still level, penalties are a 50/50 coin flip (PEN_HOME_PROB);
  - a match decided on penalties records a level scoreline with a winner, so
    only the 12-point correct-winner tier is reachable for it.

Usage:
    python 4-points-ev.py                  # resource/<today>/odds.csv
    python 4-points-ev.py --date 20260611
    python 4-points-ev.py --csv path/to/odds.csv
    python 4-points-ev.py --top 15

Zero dependencies: standard library only.
"""

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

MAX_GOALS = 10
CANDIDATE_GOALS = 6          # candidate predictions swept over 0..6 x 0..6
ET_RATE = 0.25               # extra-time goal rate as a fraction of the 90' rate
ET_MAX_GOALS = 4
PEN_HOME_PROB = 0.5

POINTS_EXACT = 25
POINTS_SAME_DIFF = 18
POINTS_DRAW_ANY = 15
POINTS_WINNER = 12

STAGE_MULTIPLIERS = {"group": 1, "r32": 2, "r16": 3, "qf": 4,
                     "sf": 5, "third": 5, "final": 6}
STAGE_ALIASES = {"group stage": "group", "round of 32": "r32", "round of 16": "r16",
                 "quarter": "qf", "quarterfinal": "qf", "quarter-final": "qf",
                 "semi": "sf", "semifinal": "sf", "semi-final": "sf",
                 "3rd": "third", "third place": "third", "third-place": "third"}


@dataclass(frozen=True)
class MatchInput:
    home_team: str
    away_team: str
    odds: tuple[float, float, float]
    rho: float
    stage: str


@dataclass(frozen=True)
class Candidate:
    score: tuple[int, int]
    expected_points: float
    p_exact: float
    p_same_diff: float
    p_winner_other: float
    p_draw_other: float


# ---------------------------------------------------------------- shared model

def poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * lam ** k / math.factorial(k)


def score_matrix(lam_home: float, lam_away: float, rho: float) -> list[list[float]]:
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


def fit_lambdas(target: tuple[float, float, float], rho: float) -> tuple[float, float]:
    def loss(lh: float, la: float) -> float:
        ph, pd, pa = outcome_probs(score_matrix(lh, la, rho))
        return (ph - target[0]) ** 2 + (pd - target[1]) ** 2 + (pa - target[2]) ** 2

    best = (1.3, 1.1)
    lo_h, hi_h, lo_a, hi_a = 0.1, 4.5, 0.1, 4.5
    for step in (0.1, 0.01, 0.001):
        best_loss = float("inf")
        h = lo_h
        while h <= hi_h:
            a = lo_a
            while a <= hi_a:
                l = loss(h, a)
                if l < best_loss:
                    best_loss, best = l, (h, a)
                a += step
            h += step
        lo_h, hi_h = max(0.05, best[0] - step), best[0] + step
        lo_a, hi_a = max(0.05, best[1] - step), best[1] + step
    return best


def strip_margin(odds: tuple[float, float, float]) -> tuple[float, float, float]:
    implied = [1.0 / o for o in odds]
    total = sum(implied)
    return tuple(p / total for p in implied)


# -------------------------------------------------------- knockout final model

def knockout_distribution(grid: list[list[float]], lam_home: float, lam_away: float,
                          rho: float) -> tuple[dict[tuple[int, int], float], float, float]:
    """Returns (decisive final scores -> prob, P(home wins on pens), P(away wins on pens)).

    90' draws are extended with an independent-Poisson extra time; rho is a
    low-score 90-minute effect, so it is deliberately not applied to extra time.
    """
    et_home = [poisson_pmf(i, lam_home * ET_RATE) for i in range(ET_MAX_GOALS + 1)]
    et_away = [poisson_pmf(j, lam_away * ET_RATE) for j in range(ET_MAX_GOALS + 1)]
    et_total = sum(et_home) * sum(et_away)

    decisive: dict[tuple[int, int], float] = {}
    pens_home = pens_away = 0.0
    for h in range(MAX_GOALS + 1):
        for a in range(MAX_GOALS + 1):
            p = grid[h][a]
            if p == 0.0:
                continue
            if h != a:
                decisive[(h, a)] = decisive.get((h, a), 0.0) + p
                continue
            for i in range(ET_MAX_GOALS + 1):
                for j in range(ET_MAX_GOALS + 1):
                    q = p * et_home[i] * et_away[j] / et_total
                    if i != j:
                        final = (h + i, a + j)
                        decisive[final] = decisive.get(final, 0.0) + q
                    else:
                        pens_home += q * PEN_HOME_PROB
                        pens_away += q * (1.0 - PEN_HOME_PROB)
    return decisive, pens_home, pens_away


# ------------------------------------------------------------- expected points

def evaluate_group(pred: tuple[int, int], grid: list[list[float]],
                   multiplier: int) -> Candidate:
    h, a = pred
    p_exact = grid[h][a]
    if h == a:
        p_draw_total = sum(grid[k][k] for k in range(MAX_GOALS + 1))
        p_draw_other = p_draw_total - p_exact
        ev = multiplier * (POINTS_EXACT * p_exact + POINTS_DRAW_ANY * p_draw_other)
        return Candidate(pred, ev, p_exact, 0.0, 0.0, p_draw_other)

    sign = 1 if h > a else -1
    diff = h - a
    p_same_diff = sum(
        grid[i][j] for i in range(MAX_GOALS + 1) for j in range(MAX_GOALS + 1)
        if (i - j) == diff and (i, j) != pred)
    p_winner = sum(
        grid[i][j] for i in range(MAX_GOALS + 1) for j in range(MAX_GOALS + 1)
        if (1 if i > j else -1 if i < j else 0) == sign)
    p_winner_other = p_winner - p_same_diff - p_exact
    ev = multiplier * (POINTS_EXACT * p_exact + POINTS_SAME_DIFF * p_same_diff
                       + POINTS_WINNER * p_winner_other)
    return Candidate(pred, ev, p_exact, p_same_diff, p_winner_other, 0.0)


def evaluate_knockout(pred: tuple[int, int], decisive: dict[tuple[int, int], float],
                      pens_home: float, pens_away: float, multiplier: int) -> Candidate:
    h, a = pred
    if h == a:
        # The result always has a winner; a draw prediction can never score.
        return Candidate(pred, 0.0, 0.0, 0.0, 0.0, 0.0)

    sign = 1 if h > a else -1
    diff = h - a
    p_exact = p_same_diff = p_winner_other = 0.0
    for (fh, fa), p in decisive.items():
        final_sign = 1 if fh > fa else -1
        if final_sign != sign:
            continue
        if (fh, fa) == pred:
            p_exact += p
        elif (fh - fa) == diff:
            p_same_diff += p
        else:
            p_winner_other += p
    p_winner_other += pens_home if sign == 1 else pens_away   # pens: winner tier only

    ev = multiplier * (POINTS_EXACT * p_exact + POINTS_SAME_DIFF * p_same_diff
                       + POINTS_WINNER * p_winner_other)
    return Candidate(pred, ev, p_exact, p_same_diff, p_winner_other, 0.0)


# ------------------------------------------------------------------------- io

def normalise_stage(raw: str) -> str:
    key = (raw or "group").strip().casefold()
    key = STAGE_ALIASES.get(key, key)
    if key not in STAGE_MULTIPLIERS:
        sys.exit(f"Unknown stage '{raw}'. Allowed: {', '.join(STAGE_MULTIPLIERS)} "
                 f"(or aliases like 'round of 16').")
    return key


def default_csv_path(date_str: str) -> Path:
    relative = Path("resource") / date_str / "odds.csv"
    project_root = Path(__file__).resolve().parent.parent
    for candidate in (project_root / relative, Path.cwd() / relative):
        if candidate.exists():
            return candidate
    return project_root / relative


def load_matches(csv_path: Path) -> list[MatchInput]:
    matches: list[MatchInput] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"home_team", "away_team", "odds_home", "odds_draw", "odds_away"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            sys.exit(f"odds.csv is missing columns: {', '.join(sorted(missing))}")
        if "stage" not in (reader.fieldnames or []):
            print("WARNING: no 'stage' column found, assuming group stage (x1) for all matches.")
        for line_number, row in enumerate(reader, start=2):
            try:
                odds = (float(row["odds_home"]), float(row["odds_draw"]), float(row["odds_away"]))
                rho_raw = (row.get("rho") or "").strip()
                rho = float(rho_raw) if rho_raw else 0.0
            except ValueError as error:
                sys.exit(f"{csv_path}, line {line_number}: {error}")
            matches.append(MatchInput(row["home_team"].strip(), row["away_team"].strip(),
                                      odds, rho, normalise_stage(row.get("stage", "group"))))
    if not matches:
        sys.exit(f"{csv_path}: no match rows found")
    return matches


def print_match_report(match: MatchInput, top_n: int) -> None:
    multiplier = STAGE_MULTIPLIERS[match.stage]
    target = strip_margin(match.odds)
    lam_home, lam_away = fit_lambdas(target, match.rho)
    grid = score_matrix(lam_home, lam_away, match.rho)
    knockout = match.stage != "group"

    print(f"\n{'=' * 68}")
    print(f"  {match.home_team}  vs  {match.away_team}"
          f"   [{match.stage}, x{multiplier}]   λ {lam_home:.2f}-{lam_away:.2f}")
    print(f"{'=' * 68}")

    candidates: list[Candidate] = []
    if knockout:
        decisive, pens_home, pens_away = knockout_distribution(grid, lam_home, lam_away,
                                                               match.rho)
        p_extra_time = sum(grid[k][k] for k in range(MAX_GOALS + 1))
        p_pens = pens_home + pens_away
        print(f"After-ET model: P(extra time) {p_extra_time:.1%}, P(penalties) {p_pens:.1%} "
              f"— draws cannot be predicted; a shootout pays the winner tier only.")
        for h in range(CANDIDATE_GOALS + 1):
            for a in range(CANDIDATE_GOALS + 1):
                if h != a:
                    candidates.append(evaluate_knockout((h, a), decisive,
                                                        pens_home, pens_away, multiplier))
    else:
        for h in range(CANDIDATE_GOALS + 1):
            for a in range(CANDIDATE_GOALS + 1):
                candidates.append(evaluate_group((h, a), grid, multiplier))

    candidates.sort(key=lambda c: c.expected_points, reverse=True)
    if knockout:
        # The 90' modal score can be a draw, which is not a predictable result
        # after ET+pens — take the mode of the decisive final-score distribution.
        modal = max(decisive, key=decisive.get)
    else:
        modal = max(((i, j) for i in range(MAX_GOALS + 1) for j in range(MAX_GOALS + 1)),
                    key=lambda s: grid[s[0]][s[1]])

    print(f"\n{'score':>7} {'EV pts':>8} {'P(25)':>8} {'P(18)':>8} {'P(12)':>8} {'P(15)':>8}")
    for candidate in candidates[:top_n]:
        h, a = candidate.score
        marker = "  <- most likely score" if (h, a) == modal else ""
        print(f"{h:>4}-{a:<2} {candidate.expected_points:>8.2f} "
              f"{candidate.p_exact:>8.2%} {candidate.p_same_diff:>8.2%} "
              f"{candidate.p_winner_other:>8.2%} {candidate.p_draw_other:>8.2%}{marker}")

    best = candidates[0]
    print(f"\nRecommendation: {best.score[0]}-{best.score[1]} "
          f"(EV {best.expected_points:.2f} pts)", end="")
    if best.score != modal:
        modal_ev = next((c.expected_points for c in candidates if c.score == modal), None)
        comparison = f" (EV {modal_ev:.2f})" if modal_ev is not None else ""
        print(f" — beats the most likely score {modal[0]}-{modal[1]}"
              f"{comparison} because of how the consolation tiers stack.")
    else:
        print(" — here the most likely score is also the EV-optimal pick.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank score predictions by expected points")
    date_to_parse = date.today() # - timedelta(days=1)
    parser.add_argument("--date", default=date_to_parse.strftime("%Y%m%d"),
                        help="matchday folder as YYYYMMDD (default: today)")
    parser.add_argument("--csv", type=Path, default=None,
                        help="explicit odds.csv path, overrides --date resolution")
    parser.add_argument("--top", type=int, default=10,
                        help="how many candidates to show per match (default 10)")
    args = parser.parse_args()

    csv_path = args.csv if args.csv else default_csv_path(args.date)
    if not csv_path.exists():
        sys.exit(f"Odds file not found: {csv_path} — run 3-aggregate-odds.py first.")

    print(f"Reading matches from {csv_path}")
    for match in load_matches(csv_path):
        print_match_report(match, args.top)


if __name__ == "__main__":
    main()