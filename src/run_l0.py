"""Level-0 baseline (methodology §7 + §10.1) through the rolling-origin harness (§15.1), data v2.

Run:  python src/run_l0.py --grid      (tuning grid on 2024/25, then the test report)
      python src/run_l0.py             (test report with the settings chosen from the saved grid)

Split (v0.4, fixed before looking at test results):
  warm-up  = 2023/24 (fitting only; never predicted)
  tuning   = 2024/25 (380 matches): every setting is chosen here
  test     = 2025/26 + 2026/27 to date (430 matches): reported, never used to choose settings

Settings chosen on the tuning season (npxG model): half-life, ridge, season carry-over factor.
Rule: lowest out-of-sample npxG deviance; configurations within 0.5% of the best deviance are
treated as tied and the one with the lowest RPS is chosen (methodology §16: "RPS tie-break").
The goals-response comparison model is tuned on the same season by RPS.

Outputs (reports/): team_strength_grid.csv, l0_chosen.json, season_carryover.csv,
  l0_test_scores.csv, l0_comparisons.csv, l0_calibration.csv, l0_by_season.csv,
  team_ratings_current.csv, l0_summary.json; data/processed/l0_predictions.parquet
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
WARMUP, TUNE = "2023/2024", "2024/2025"
TEST = ("2025/2026", "2026/2027")
HALF_LIVES = [60, 90, 120, 180, 270, 365, 540, 730, np.inf]
RIDGES = [0.5, 1, 2, 4, 8, 16]
CARRIES = [1.0, 0.75, 0.5, 0.25]
TIE_TOL = 0.005


def masks(matches: pd.DataFrame):
    return matches.season.eq(TUNE), matches.season.isin(TEST)


# --------------------------------------------------------------------------- tuning
def grid(matches, sides):
    tune, _ = masks(matches)
    rows = []
    for resp, hl, rg, cy in itertools.product(["npxg", "goals"], HALF_LIVES, RIDGES, CARRIES):
        kw = dict(half_life_days=hl, ridge=rg, carry=cy, response=resp)
        p = rolling_predictions(matches, sides, kw, tune, dixon_coles=False)
        s = score(p)
        rows.append(dict(response=resp, half_life=hl, ridge=rg, carry=cy, tune_rps=s["rps"],
                         tune_ll=s["log_loss"], tune_npxg_dev=s["npxg_deviance"]))
        print(resp, hl, rg, cy, round(s["npxg_deviance"], 5), round(s["rps"], 5), flush=True)
    g = pd.DataFrame(rows)
    g.to_csv(REPORTS / "team_strength_grid.csv", index=False)
    return g


def choose(g: pd.DataFrame) -> dict:
    def kw(r, resp):
        return dict(half_life_days=float(r.half_life), ridge=float(r.ridge), carry=float(r.carry), response=resp)
    n = g[g.response == "npxg"]
    best = n.tune_npxg_dev.min()
    tied = n[n.tune_npxg_dev <= best * (1 + TIE_TOL)]
    r = tied.sort_values("tune_rps").iloc[0]
    by_dev = n.sort_values("tune_npxg_dev").iloc[0]
    gl = g[g.response == "goals"].sort_values("tune_rps").iloc[0]
    out = dict(npxg=kw(r, "npxg"), goals=kw(gl, "goals"), n_tied=int(len(tied)),
               best_by_deviance_alone=kw(by_dev, "npxg"),
               tune_scores=dict(npxg_dev=float(r.tune_npxg_dev), rps=float(r.tune_rps), log_loss=float(r.tune_ll)),
               tune_scores_goals_model=dict(rps=float(gl.tune_rps), log_loss=float(gl.tune_ll)))
    (REPORTS / "l0_chosen.json").write_text(json.dumps(out, indent=2, default=str))
    return out


# --------------------------------------------------------------------------- carry-over (omega)
def season_carryover(matches, sides, n_boot=2000, seed=0):
    """How much of a team's season rating survives into the next season.

    Each season is fitted on its own (no decay, light ridge) and the rating is the npxG difference
    per match against an average opponent at a neutral venue. omega = OLS slope of next-season
    rating on this-season rating, for teams in both seasons. Uses full seasons, so it is a
    descriptive check on the tuned carry factor, not an input to any prediction.
    """
    seasons = sorted(matches.season.unique())
    full = [s for s in seasons if matches.season.eq(s).sum() == 380]
    rat = {}
    for s in full:
        d = sides[sides.season == s]
        end = d.date.max() + pd.Timedelta(days=1)
        m = StrengthModel(half_life_days=np.inf, ridge=1.0).fit(d, end)
        p = m.params
        base = np.exp(p["mu"] + p["home"] / 2)
        rat[s] = {t: base * (np.exp(p["att"][t]) - np.exp(-p["defence"][t])) for t in p["att"]}
    rows = []
    for a, b in zip(full[:-1], full[1:]):
        for t in set(rat[a]) & set(rat[b]):
            rows.append(dict(season_from=a, season_to=b, team_id=t, rating_from=rat[a][t], rating_to=rat[b][t]))
    df = pd.DataFrame(rows)
    names = pd.concat([matches[["home_team_id", "home_team"]].set_axis(["team_id", "team"], axis=1)]).drop_duplicates("team_id")
    df = df.merge(names, on="team_id")
    df.to_csv(REPORTS / "season_carryover.csv", index=False)

    def slope(x, y):
        x = x - x.mean()
        return float((x * (y - y.mean())).sum() / (x ** 2).sum())
    rng = np.random.default_rng(seed)
    res = {}
    for key, d in list(df.groupby(["season_from", "season_to"])) + [(("pooled", ""), df)]:
        x, y = d.rating_from.to_numpy(), d.rating_to.to_numpy()
        boots = [slope(x[i], y[i]) for i in rng.integers(0, len(x), (n_boot, len(x)))]
        res[" -> ".join(k for k in key if k)] = dict(
            n=len(d), omega=slope(x, y), ci=[float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
            r=float(np.corrcoef(x, y)[0, 1]))
    return res


# --------------------------------------------------------------------------- report helpers
def market_predictions(matches, target):
    o = pd.read_parquet(PROC / "odds.parquet")
    return target[["match_id"]].merge(o[["match_id", "p_home", "p_draw", "p_away"]], on="match_id")


def current_ratings(matches, sides, kw, as_of=None):
    """Ratings as of the day after the last match in the data, in npxG-per-match terms."""
    as_of = as_of or matches.date.max() + pd.Timedelta(days=1)
    seasons = sorted(matches.season.unique())
    season = matches.loc[matches.date < as_of, "season"].iloc[-1]
    m = StrengthModel(**kw).fit(sides, as_of, season_start_prior(sides, matches, kw, season), seasons.index(season))
    teams = matches.loc[matches.season == season, ["home_team_id", "home_team"]].drop_duplicates()
    p = m.params
    r = pd.DataFrame({"team": teams.home_team.values,
                      "att": [p["att"][t] for t in teams.home_team_id],
                      "def": [p["defence"][t] for t in teams.home_team_id]})
    base = np.exp(p["mu"] + p["home"] / 2)   # neutral venue, average opponent
    r["npxg_for_per_match"] = base * np.exp(r.att)
    r["npxg_against_per_match"] = base * np.exp(-r["def"])
    r["npxg_diff_per_match"] = r.npxg_for_per_match - r.npxg_against_per_match
    return r.sort_values("npxg_diff_per_match", ascending=False).reset_index(drop=True), as_of


def rolling_five_check(sides, preds):
    """§7.2 sanity check: model npxG difference vs the previous five matches' npxG difference."""
    s = sides.sort_values("date").copy()
    opp = s[["match_id", "team_id", "npxg"]].rename(columns={"team_id": "opponent_id", "npxg": "npxg_opp"})
    s = s.merge(opp, on=["match_id", "opponent_id"])
    s["npxg_diff"] = s.npxg - s.npxg_opp
    s = s.sort_values("date")
    s["roll5"] = s.groupby("team_id").npxg_diff.transform(lambda v: v.shift(1).rolling(5).mean())
    k = s[s.home == 1].merge(preds[["match_id", "npxg_home_pred", "npxg_away_pred"]], on="match_id")
    k["model_diff"] = k.npxg_home_pred - k.npxg_away_pred
    k = k.dropna(subset=["roll5"])
    return float(np.corrcoef(k.model_diff, k.roll5)[0, 1]), len(k)


def calibration(q):
    P = q[["p_home", "p_draw", "p_away"]].to_numpy()
    O = np.eye(3)[q.outcome.to_numpy()]
    c = pd.DataFrame({"p": P.ravel(), "o": O.ravel()})
    c["bin"] = pd.cut(c.p, [0, .15, .25, .35, .45, .55, .7, 1.0])
    c = c.groupby("bin", observed=True).agg(predicted=("p", "mean"), observed=("o", "mean"), n=("o", "size")).reset_index()
    se = np.sqrt(c.observed * (1 - c.observed) / c.n)
    c["ci_low"], c["ci_high"] = c.observed - 1.96 * se, c.observed + 1.96 * se
    c["bin"] = c["bin"].astype(str)
    mix = dict(predicted=dict(zip(["home", "draw", "away"], P.mean(0).round(3).tolist())),
               observed=dict(zip(["home", "draw", "away"], O.mean(0).round(3).tolist())))
    return c, mix


def main(run_grid: bool = False):
    REPORTS.mkdir(exist_ok=True)
    matches, sides = load()
    tune, test = masks(matches)
    g = grid(matches, sides) if run_grid else pd.read_csv(REPORTS / "team_strength_grid.csv")
    chosen = choose(g)
    kw, kw_goals = chosen["npxg"], chosen["goals"]
    carry = season_carryover(matches, sides)

    preds = {
        "L0": rolling_predictions(matches, sides, kw, test, dixon_coles=True),
        "L0 without Dixon–Coles": rolling_predictions(matches, sides, kw, test, dixon_coles=False),
        "Goals ratings": rolling_predictions(matches, sides, kw_goals, test, dixon_coles=True),
    }
    l0 = preds["L0"]
    outcome = l0[["match_id", "outcome", "season"]]
    preds["Base rates"] = base_rate_predictions(matches, matches[test]).merge(outcome, on="match_id")
    preds["Market (Bet365 closing)"] = market_predictions(matches, matches[test]).merge(outcome, on="match_id")

    rows, per_match = [], {}
    for name, p in preds.items():
        q = p.sort_values("match_id")
        rows.append(dict(model=name, **score(q)))
        P = q[["p_home", "p_draw", "p_away"]].to_numpy()
        per_match[name] = (rps(P, q.outcome.to_numpy()), log_loss(P, q.outcome.to_numpy()))
    table = pd.DataFrame(rows)

    pairs = [("L0", "Base rates"), ("L0", "Goals ratings"), ("L0", "L0 without Dixon–Coles"),
             ("L0", "Market (Bet365 closing)"), ("Market (Bet365 closing)", "Base rates")]
    comps = []
    for a, b in pairs:
        for i, metric in enumerate(["rps", "log_loss"]):
            d, lo, hi = paired_bootstrap(per_match[a][i], per_match[b][i])
            comps.append(dict(comparison=f"{a} minus {b}", metric=metric, diff=d, ci_low=lo, ci_high=hi))
    comps = pd.DataFrame(comps)
    gap = comps[(comps.comparison == "L0 minus Market (Bet365 closing)") & (comps.metric == "rps")].iloc[0]
    gap_base = comps[(comps.comparison == "Market (Bet365 closing) minus Base rates") & (comps.metric == "rps")].iloc[0]
    share_of_market_gain = float(1 - gap["diff"] / -gap_base["diff"])

    by_season = []
    for name, p in preds.items():
        for s, q in p.groupby("season"):
            by_season.append(dict(model=name, season=s, **score(q)))
    by_season = pd.DataFrame(by_season)

    calib, mix = calibration(l0)
    calib_mkt, mix_mkt = calibration(preds["Market (Bet365 closing)"])
    ratings, as_of = current_ratings(matches, sides, kw)
    corr, n_corr = rolling_five_check(sides, l0)

    # Tuning-season reference for the chosen settings (with Dixon–Coles), next to the market.
    tune_l0 = rolling_predictions(matches, sides, kw, tune, dixon_coles=True)
    tune_mkt = market_predictions(matches, matches[tune]).merge(tune_l0[["match_id", "outcome"]], on="match_id")

    l0.to_parquet(PROC / "l0_predictions.parquet", index=False)
    table.to_csv(REPORTS / "l0_test_scores.csv", index=False)
    comps.to_csv(REPORTS / "l0_comparisons.csv", index=False)
    calib.assign(model="L0").pipe(lambda c: pd.concat([c, calib_mkt.assign(model="Market")])).to_csv(
        REPORTS / "l0_calibration.csv", index=False)
    by_season.to_csv(REPORTS / "l0_by_season.csv", index=False)
    ratings.to_csv(REPORTS / "team_ratings_current.csv", index=False)
    summary = dict(
        split=dict(warmup=WARMUP, tuning=TUNE, test=list(TEST), n_tune=int(tune.sum()), n_test=int(test.sum())),
        chosen=chosen, season_carryover_omega=carry,
        test_outcome_mix=dict(L0=mix, market=mix_mkt),
        l0_share_of_market_gain_over_base_rates_rps=share_of_market_gain,
        tuning_season_reference=dict(L0=score(tune_l0), market=score(tune_mkt)),
        rolling_five_corr=corr, rolling_five_n=n_corr, ratings_as_of=str(as_of),
        dc_rho_range=[float(l0.dc_rho.min()), float(l0.dc_rho.max())],
        calib_c_range=[float(l0.calib_c.min()), float(l0.calib_c.max())],
        home_advantage_multiplier=float(np.exp(l0.home_adv).mean()))
    (REPORTS / "l0_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return dict(table=table, comps=comps, calib=calib, ratings=ratings, summary=summary, by_season=by_season)


if __name__ == "__main__":
    r = main(run_grid="--grid" in sys.argv)
    pd.set_option("display.width", 200)
    print(r["table"].round(4).to_string(index=False))
    print(r["comps"].round(4).to_string(index=False))
    print(r["by_season"].round(4).to_string(index=False))
    print(r["calib"].round(3).to_string(index=False))
    print(json.dumps(r["summary"], indent=2, default=str))
    print(r["ratings"].round(3).to_string())
