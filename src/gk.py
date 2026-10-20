"""Step 3: goalkeeper shot-stopping prior (methodology §8).

For every on-target shot (strictly on target: goal or saved, not blocked; penalties and own goals
excluded) the keeper on the pitch at that moment is found from the stints. Per keeper, as of each date:

    G_k = goals conceded,  X_k = sum of xGOT faced,   raw ratio r_k = G_k / X_k  (1 = average)

shrunk with the §4.4 credibility rule, exposure = xGOT faced:

    r_hat_k = (G_k + b * r_bar) / (X_k + b),   b = nu / tau2
    nu      = sum xGOT (1 - xGOT) / sum xGOT                (binomial noise per unit of xGOT)
    tau2    = (SS - (n - 1) nu) / (sum X - sum X^2 / sum X), SS = sum_k X_k (r_k - r_bar)^2

over every keeper with any shots before the date; no time decay (shot-stopping is treated as a
steady trait, like shooting in Step 2). If tau2 <= 0 every keeper gets the league ratio.

The model target is npxG, which a keeper cannot change, so shot-stopping enters where it acts: the
goals the opponent is expected to score are multiplied by m_k = r_hat_k / r_bar for the starting keeper.
Goals prevented GP_k = sum (xGOT - goal) is reported alongside.

Run:  python src/gk.py
Outputs: data/processed/gk_asof.parquet (one row per team-match: starting keeper, G, X, r_hat, mult)
         reports/gk_summary.json, reports/gk_current.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"


def keeper_shots() -> pd.DataFrame:
    shots = pd.read_parquet(PROC / "shots.parquet")
    ss = pd.read_parquet(PROC / "shot_stints.parquet")
    spl = pd.read_parquet(PROC / "stint_players.parquet")
    s = shots[(shots.on_target_strict == 1) & (shots.situation_group != "penalty") & (shots.is_own_goal == 0)]
    s = s.merge(ss[["shot_id", "stint"]], on="shot_id")
    gk = spl[spl.is_gk][["match_id", "stint", "team_id", "player_id"]].rename(
        columns={"team_id": "opponent_id", "player_id": "keeper_id"})
    gk = gk.drop_duplicates(["match_id", "stint", "opponent_id"])
    s = s.merge(gk, on=["match_id", "stint", "opponent_id"], how="left")
    s["date"] = s.date.dt.tz_convert(None) if s.date.dt.tz is not None else s.date
    return s


def credibility(G, X, nu):
    """Shrunk ratios r_hat for arrays of goals G and xGOT X (one entry per keeper)."""
    keep = X > 0
    if keep.sum() < 3:
        return np.ones_like(X), 1.0, np.inf
    rbar = G[keep].sum() / X[keep].sum()
    r = G[keep] / X[keep]
    SS = np.sum(X[keep] * (r - rbar) ** 2)
    denom = X[keep].sum() - np.sum(X[keep] ** 2) / X[keep].sum()
    tau2 = (SS - (keep.sum() - 1) * nu) / denom
    if tau2 <= 0:
        return np.full_like(X, rbar), rbar, np.inf
    b = nu / tau2
    return (G + b * rbar) / (X + b), rbar, b


def build() -> tuple[pd.DataFrame, dict]:
    s = keeper_shots().dropna(subset=["keeper_id"])
    s["keeper_id"] = s.keeper_id.astype(int)
    pm = pd.read_parquet(PROC / "player_match.parquet")
    m = pd.read_parquet(PROC / "matches.parquet")
    m["date"] = m.date.dt.tz_convert(None) if m.date.dt.tz is not None else m.date
    starters = pm[pm.started & pm.is_gk][["match_id", "team_id", "player_id", "player_name"]].drop_duplicates(
        ["match_id", "team_id"]).merge(m[["match_id", "date", "season"]], on="match_id")
    rows, bs = [], []
    for day, g in starters.groupby(starters.date.dt.floor("D")):
        as_of = g.date.min()
        past = s[s.date < as_of]
        agg = past.groupby("keeper_id").agg(G=("is_goal", "sum"), X=("xgot", "sum"))
        nu = float((past.xgot * (1 - past.xgot)).sum() / past.xgot.sum()) if len(past) else 1.0
        rh, rbar, b = credibility(agg.G.to_numpy(float), agg.X.to_numpy(float), nu)
        rhat = pd.Series(rh, index=agg.index)
        bs.append(dict(date=as_of, b=b, rbar=rbar, nu=nu, n_keepers=len(agg)))
        for r in g.itertuples():
            Gk, Xk = (agg.G.get(r.player_id, 0.0), agg.X.get(r.player_id, 0.0))
            rk = rhat.get(r.player_id, rbar if np.isfinite(rbar) else 1.0)
            rows.append(dict(match_id=r.match_id, team_id=r.team_id, season=r.season, date=r.date,
                             keeper_id=r.player_id, keeper=r.player_name, G=Gk, X=Xk, gp=Xk - Gk,
                             r_hat=rk, rbar=rbar, mult=rk / rbar if rbar else 1.0))
    out = pd.DataFrame(rows)
    bs = pd.DataFrame(bs)
    return out, bs


def wide(gk: pd.DataFrame) -> pd.DataFrame:
    """One row per match: home and away keepers' multipliers on the goals they concede."""
    m = pd.read_parquet(PROC / "matches.parquet")[["match_id", "home_team_id", "away_team_id"]]
    k = gk.set_index(["match_id", "team_id"]).mult
    m["mult_home"] = k.reindex(pd.MultiIndex.from_frame(m[["match_id", "home_team_id"]])).fillna(1).to_numpy()
    m["mult_away"] = k.reindex(pd.MultiIndex.from_frame(m[["match_id", "away_team_id"]])).fillna(1).to_numpy()
    return m[["match_id", "mult_home", "mult_away"]]


def main():
    gk, bs = build()
    gk.to_parquet(PROC / "gk_asof.parquet", index=False)
    last = gk.sort_values("date").groupby("keeper_id").tail(1)
    cur = last[last.date >= gk.date.max() - pd.Timedelta(days=120)].sort_values("mult")
    cur[["keeper", "team_id", "G", "X", "gp", "r_hat", "mult"]].to_csv(REPORTS / "gk_current.csv", index=False)
    test = gk[gk.season.isin(["2025/2026", "2026/2027"])]
    summ = dict(
        b_xgot_range_test=[float(bs.loc[bs.date >= test.date.min(), "b"].min()), float(bs.loc[bs.date >= test.date.min(), "b"].max())],
        rbar_range_test=[float(bs.loc[bs.date >= test.date.min(), "rbar"].min()), float(bs.loc[bs.date >= test.date.min(), "rbar"].max())],
        mult_quantiles_test=test.mult.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(3).tolist(),
        n_team_matches=len(gk))
    (REPORTS / "gk_summary.json").write_text(json.dumps(summ, indent=2, default=str))
    print(json.dumps(summ, indent=2, default=str))
    print(cur[["keeper", "G", "X", "gp", "mult"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
