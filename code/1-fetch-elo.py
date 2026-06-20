#!/usr/bin/env python3
"""Bolão Layer 1a: fetch World Football Elo ratings into resource/<YYYYMMDD>/elo.csv.

eloratings.net renders its table client-side, but the underlying data is served
as plain TSV — so we skip the browser entirely and hit the data endpoints:

    World.tsv      ratings table, no header; rank=col 0, team code=col 2, rating=col 3
    en.teams.tsv   team code -> English display name

Usage:
    python 1-fetch-elo.py                  # writes resource/<today>/elo.csv
    python 1-fetch-elo.py --date 20260611
    python 1-fetch-elo.py --out path/to/elo.csv

Dependencies: requests
"""

import argparse
import csv
import sys
from datetime import date
from pathlib import Path

import requests

RATINGS_URL = "https://www.eloratings.net/World.tsv"
TEAM_NAMES_URL = "https://www.eloratings.net/en.teams.tsv"
HEADERS = {"User-Agent": "Mozilla/5.0 (personal bolao research; one request per matchday)"}
MIN_EXPECTED_TEAMS = 150
RATING_RANGE = (500, 2500)


def fetch_text(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.text


def parse_ratings(tsv_text: str) -> list[tuple[int, str, int]]:
    """Returns (rank, code, rating) per team, skipping malformed rows."""
    rows: list[tuple[int, str, int]] = []
    for line in tsv_text.splitlines():
        fields = line.split("\t")
        if len(fields) < 4:
            continue
        try:
            rank = int(fields[0])
            rating = int(fields[3])
        except ValueError:
            continue
        code = fields[2].strip()
        if code and RATING_RANGE[0] <= rating <= RATING_RANGE[1]:
            rows.append((rank, code, rating))
    return rows


def parse_team_names(tsv_text: str) -> dict[str, str]:
    """Returns code -> display name; first field is the code, the next
    non-empty field is the name. Defensive: unknown extra columns are ignored."""
    names: dict[str, str] = {}
    for line in tsv_text.splitlines():
        fields = [f.strip() for f in line.split("\t")]
        if len(fields) < 2 or not fields[0]:
            continue
        display = next((f for f in fields[1:] if f), "")
        if display:
            names[fields[0]] = display
    return names


def default_out_path(date_str: str) -> Path:
    project_root = Path(__file__).resolve().parent.parent
    return project_root / "resource" / date_str / "elo.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Elo ratings to CSV")
    parser.add_argument("--date", default=date.today().strftime("%Y%m%d"),
                        help="matchday folder as YYYYMMDD (default: today)")
    parser.add_argument("--out", type=Path, default=None,
                        help="explicit output path, overrides --date resolution")
    args = parser.parse_args()

    print(f"Fetching {RATINGS_URL}")
    ratings = parse_ratings(fetch_text(RATINGS_URL))
    if len(ratings) < MIN_EXPECTED_TEAMS:
        sys.exit(f"Only parsed {len(ratings)} teams — the TSV format may have changed; "
                 f"inspect {RATINGS_URL} manually.")

    print(f"Fetching {TEAM_NAMES_URL}")
    try:
        names = parse_team_names(fetch_text(TEAM_NAMES_URL))
    except requests.RequestException as error:
        print(f"WARNING: could not fetch team names ({error}); falling back to codes.")
        names = {}

    out_path = args.out if args.out else default_out_path(args.date)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "code", "team", "rating"])
        for rank, code, rating in ratings:
            writer.writerow([rank, code, names.get(code, code), rating])

    top = ratings[0]
    print(f"Wrote {len(ratings)} teams to {out_path}")
    print(f"Sanity check — #1: {names.get(top[1], top[1])} at {top[2]}")


if __name__ == "__main__":
    main()