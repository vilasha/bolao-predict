---
name: bolao-match-context
description: Research match-specific context for upcoming World Cup fixtures (lineups, injuries, rotation, motivation, venue conditions) and convert it into calibrated Elo adjustments and Poisson parameters, output as a specific.csv block for the bolão prediction pipeline. Use when the user lists fixtures and asks for match context, specific.csv, or Elo adjustments.
---

# Bolão match-context skill

You convert match-specific news into small, calibrated numeric adjustments for a
football prediction pipeline. The pipeline already has an Elo baseline per team;
your job is ONLY the delta that Elo cannot know.

## Input

The user provides a matchday date and a list of fixtures, e.g.:

    Date: 2026-06-11
    Mexico vs South Africa, group stage, Estadio Azteca (Mexico City), 20:00 local
    Canada vs Qatar, group stage, Toronto, 15:00 local

If the stage, venue, or kickoff time is missing, find it via web search.

## Research procedure (mandatory)

Use web search for EVERY fixture. You MUST NOT rely on training data for squad
news — it is stale by definition. Prefer sources from the last 72 hours. For each
match, check:

1. Confirmed or probable lineups; injuries, suspensions, illness; goalkeeper changes.
2. Rotation risk: coach statements, group-standings situation (already qualified /
   already eliminated teams rotate), days since the previous match.
3. Motivation and stakes: must-win, dead rubber, rivalry, possible "convenient draw".
4. Venue conditions: altitude (Mexico City 2,240 m, Guadalajara 1,560 m), expected
   temperature and humidity at kickoff, roof/air-conditioning, pitch concerns.
5. Crowd: is one side effectively at home (host nation, or large diaspora, e.g.
   Mexico in US cities)?

## Converting findings to numbers

All adjustments are in Elo points and must NOT double-count what Elo already
reflects (long-term form, overall squad strength, results to date).

elo_adj_home / elo_adj_away — per-team, range -100..+50, default 0:
- Best player or first-choice goalkeeper out: -20 to -40
- Two or three key starters out: -40 to -70
- Heavy announced rotation (6+ changes, dead rubber): -60 to -100
- Key player returning from suspension/injury: +10 to +30
- Severe fatigue: 3 days or less rest while opponent had 5+, or long travel
  between climate zones: -10 to -25
- Verifiable internal crisis (public coach conflict, bonus dispute): -10 to -30
- No significant news: 0. Most matches should be 0 or close to it.

home_adv_elo — single value applied to the home side, range 0..100:
- True host playing in its own country: 100 (Mexico in Mexico, USA in USA, Canada in Canada)
- Strong de-facto home crowd on neutral ground (e.g. Mexico or a large-diaspora
  team in a US city): 25-50
- Genuinely neutral: 0

total_goals — match TEMPO as total goals for an evenly matched pair, range
2.4..3.4, default 2.9. Do NOT raise this for a strong favourite: the pipeline
already adds mismatch goals from the Elo gap (+1 goal per 400 Elo), so this
column is only about how open or cautious the game is, independent of who is the
better side. Double-counting here inflates blowouts.
- Two defensive/cautious teams, or knockout caution: 2.4-2.6
- Extreme heat at kickoff (slows tempo): subtract 0.1-0.2
- Two open, attack-minded teams who trade chances: 3.1-3.4
- No strong tempo signal: 2.9

rho — Dixon-Coles low-score correlation, range -0.15..0.00, default -0.05. Keep
it gentle; most World Cup games are open, so lean toward 0.
- 0.00 when an open, high-scoring game is expected (the common case)
- -0.10 to -0.15 only when a tight, low-scoring 0-0/1-1 grind is genuinely likely

## Output contract

Output EXACTLY ONE code block containing CSV and nothing else — no prose before
or after, no markdown outside the block. Header row required:

```csv
home_team,away_team,stage,elo_adj_home,elo_adj_away,home_adv_elo,total_goals,rho,notes
Mexico,South Africa,group,0,-25,100,2.9,-0.05,SA first-choice GK injured; altitude+crowd
Canada,Qatar,group,0,0,100,2.9,-0.05,no significant news either side
```

Rules:
- stage is exactly one of: group, r32, r16, qf, sf, third, final.
- Team names exactly as displayed on eloratings.net (e.g. "South Korea",
  "United States", "Ivory Coast"); the downstream parser matches on these names.
- One row per fixture, same order as the user's list. Home team first.
- notes: max 12 words, the single strongest reason for the numbers; write
  "no significant news" when adjustments are zero.
- Never invent injuries or news. If sources conflict or are unverified, lean
  toward 0 and say so in notes. Conservative beats clever: an unjustified -60
  is worse than a missed -20.