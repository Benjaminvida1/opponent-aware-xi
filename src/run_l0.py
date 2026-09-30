"""Run the level-0 baseline (methodology §7 + §10.1) through the rolling-origin harness (§15.1).

Run:  python src/run_l0.py [--grid]

Splits (fixed before looking at results):
  evaluation  = 2025/26 rounds 11-38 + all of 2026/27 (330 matches)
  tuning      = 2025/26 rounds 11-24 (140 matches): used to pick half-life and ridge
  test        = 2025/26 rounds 25-38 + 2026/27 rounds 1-5 (190 matches): reported
Rounds 1-10 of 2025/26 are warm-up only (no earlier season in the data).
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from team_strength import (PROC, ROOT, StrengthModel, base_rate_predictions, load, log_loss,  # noqa: E402
                           paired_bootstrap, rolling_predictions, rps, score, season_start_prior)

REPORTS = ROOT / "reports"
CHOSEN = dict(half_life_days=270.0, ridge=4.0)          # from the tuning grid (see findings log)
CHOSEN_GOALS = dict(half_life_days=120.0, ridge=4.0, response="goals")


def splits(matches: pd.DataFrame):
    s25 = matches.season.eq("2025/2026")
    ev = (s25 & matches["round"].ge(11)) | matches.season.eq("2026/2027")
    tune_ids = set(matches.loc[s25 & matches["round"].between(11, 24), "match_id"])
    return ev, tune_ids


def grid(matches, sides):
    ev, tune_ids = splits(matches)
    rows = []
    for resp in ["npxg", "goals"]:
        for hl, rg in itertools.product([30, 60, 90, 120, 180, 270, 365, np.inf], [0.5, 1, 2, 4, 8]):
            p = rolling_predictions(matches, sides, dict(half_life_days=hl, ridge=rg, response=resp), ev,
                                    dixon_coles=False)
            t = score(p[p.match_id.isin(tune_ids)])
            rows.append(dict(response=resp, half_life=hl, ridge=rg, tune_rps=t["rps"], tune_ll=t["log_loss"],
                             tune_npxg_dev=t["npxg_deviance"]))
    g = pd.DataFrame(rows)
    g.to_csv(REPORTS / "team_strength_grid.csv", index=False)
    return g


def current_ratings(matches, sides, as_of=None):
    """Ratings as of the day after the last match in the data, in npxG-per-match terms."""
    as_of = as_of or matches.date.max() + pd.Timedelta(days=1)
    season = matches.loc[matches.date < as_of, "season"].iloc[-1]
    m = StrengthModel(**CHOSEN).fit(sides, as_of, season_start_prior(sides, matches, CHOSEN, season))
    teams = matches.loc[matches.season == season, ["home_team_id", "home_team"]].drop_duplicates()
    p = m.params
    r = pd.DataFrame({
        "team": teams.home_team.values,
        "att": [p["att"][t] for t in teams.home_team_id],
        "def": [p["defence"][t] for t in teams.home_team_id],
    })
    base = np.exp(p["mu"] + p["home"] / 2)   # neutral venue, average opponent
    r["npxg_for_per_match"] = base * np.exp(r.att)
    r["npxg_against_per_match"] = base * np.exp(-r["def"])
    r["npxg_diff_per_match"] = r.npxg_for_per_match - r.npxg_against_per_match
    return r.sort_values("npxg_diff_per_match", ascending=False).reset_index(drop=True), as_of


def rolling_five_check(matches, sides, preds):
    """§7.2 sanity check: model rating difference vs last-5 npxG difference, per team-match."""
    s = sides.sort_values("date").copy()
    s["npxg_diff"] = s.npxg - s.merge(
        s[["match_id", "team_id", "npxg"]].rename(columns={"team_id": "opponent_id", "npxg": "npxg_opp"}),
        on=["match_id", "opponent_id"]).npxg_opp.values
    s["roll5"] = s.groupby("team_id").npxg_diff.transform(lambda v: v.shift(1).rolling(5).mean())
    home = preds[["match_id", "npxg_home_pred", "npxg_away_pred"]]
    k = s[s.home == 1].merge(home, on="match_id")
    k["model_diff"] = k.npxg_home_pred - k.npxg_away_pred
    k = k.dropna(subset=["roll5"])
    return float(np.corrcoef(k.model_diff, k.roll5)[0, 1]), len(k)


def main(run_grid: bool = False):
    REPORTS.mkdir(exist_ok=True)
    matches, sides = load()
    if run_grid:
        grid(matches, sides)
    ev, tune_ids = splits(matches)

    preds = {
        "L0 (npxG ratings, Dixon–Coles)": rolling_predictions(matches, sides, CHOSEN, ev, dixon_coles=True),
        "L0 without Dixon–Coles": rolling_predictions(matches, sides, CHOSEN, ev, dixon_coles=False),
        "Goals ratings": rolling_predictions(matches, sides, CHOSEN_GOALS, ev, dixon_coles=True),
    }
    l0 = preds["L0 (npxG ratings, Dixon–Coles)"]
    preds["Base rates"] = base_rate_predictions(matches, matches[ev]).merge(l0[["match_id", "outcome"]], on="match_id")

    rows, per_match = [], {}
    for name, p in preds.items():
        q = p[~p.match_id.isin(tune_ids)].sort_values("match_id")
        s = score(q)
        rows.append(dict(model=name, **s))
        P = q[["p_home", "p_draw", "p_away"]].to_numpy()
        per_match[name] = (rps(P, q.outcome.to_numpy()), log_loss(P, q.outcome.to_numpy()))
    table = pd.DataFrame(rows)

    comps = []
    ref = "L0 (npxG ratings, Dixon–Coles)"
    for other in [k for k in preds if k != ref]:
        for i, metric in enumerate(["rps", "log_loss"]):
            d, lo, hi = paired_bootstrap(per_match[ref][i], per_match[other][i])
            comps.append(dict(comparison=f"{ref} minus {other}", metric=metric, diff=d, ci_low=lo, ci_high=hi))
    comps = pd.DataFrame(comps)

    q = l0[~l0.match_id.isin(tune_ids)]
    P = q[["p_home", "p_draw", "p_away"]].to_numpy()
    O = np.eye(3)[q.outcome.to_numpy()]
    calib = pd.DataFrame({"p": P.ravel(), "o": O.ravel()})
    calib["bin"] = pd.cut(calib.p, [0, .15, .25, .35, .45, .55, 1.0])
    calib = calib.groupby("bin", observed=True).agg(predicted=("p", "mean"), observed=("o", "mean"),
                                                    n=("o", "size")).reset_index()
    calib["bin"] = calib["bin"].astype(str)
    mix = dict(predicted=dict(zip(["home", "draw", "away"], P.mean(0).round(3))),
               observed=dict(zip(["home", "draw", "away"], O.mean(0).round(3))))

    ratings, as_of = current_ratings(matches, sides)
    corr, n_corr = rolling_five_check(matches, sides, l0)

    l0.to_parquet(PROC / "l0_predictions.parquet", index=False)
    table.to_csv(REPORTS / "l0_test_scores.csv", index=False)
    comps.to_csv(REPORTS / "l0_comparisons.csv", index=False)
    calib.to_csv(REPORTS / "l0_calibration.csv", index=False)
    ratings.to_csv(REPORTS / "team_ratings_current.csv", index=False)
    summary = dict(chosen=CHOSEN, n_test=int((~l0.match_id.isin(tune_ids)).sum()), outcome_mix=mix,
                   rolling_five_corr=corr, rolling_five_n=n_corr, ratings_as_of=str(as_of),
                   dc_rho_range=[float(l0.dc_rho.min()), float(l0.dc_rho.max())],
                   calib_c_range=[float(l0.calib_c.min()), float(l0.calib_c.max())],
                   home_advantage_multiplier=float(np.exp(l0.home_adv).mean()))
    (REPORTS / "l0_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return dict(table=table, comps=comps, calib=calib, ratings=ratings, summary=summary, preds=preds)


if __name__ == "__main__":
    r = main(run_grid="--grid" in sys.argv)
    pd.set_option("display.width", 200)
    print(r["table"].round(4).to_string(index=False))
    print(r["comps"].round(4).to_string(index=False))
    print(r["calib"].round(3).to_string(index=False))
    print(json.dumps(r["summary"], indent=2, default=str))
    print(r["ratings"].round(3).to_string())
