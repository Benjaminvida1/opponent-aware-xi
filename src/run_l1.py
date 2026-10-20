"""Step 3: L1 (player effects) against L0 through the rolling-origin harness (methodology §9, §15).

Run:  python src/run_l1.py --grid    (tuning grid on 2024/25, then the test report)
      python src/run_l1.py           (test report with the settings chosen from the saved grid)

Every setting is chosen on 2024/25: lowest out-of-sample npxG deviance (team-match sides); settings
within 0.5% of the best are tied and the lowest RPS wins (the L0 rule). Test = 2025/26 + 2026/27.

Goals and W/D/L use L0's as-of conversion for the same date (c, penalty xG per team-match, Dixon–Coles
rho), so L1 and L0 differ only in the npxG forecast, plus the goalkeeper factor if kept (gk.py).

Outputs (reports/): l1_grid.csv, l1_chosen.json, l1_test_scores.csv, l1_comparisons.csv,
  l1_by_season.csv, l1_summary.json; data/processed/l0_predictions_all.parquet,
  data/processed/l1_predictions.parquet
"""
from __future__ import annotations

import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import team_strength as ts_  # noqa: E402
from outcome import PROC, REPORTS, TEST, TUNE, Config, StintData, rolling  # noqa: E402
from team_strength import log_loss, outcome_probs, paired_bootstrap, poisson_deviance, rps  # noqa: E402

TIE_TOL = 0.005
GRID = dict(lam_tau=[4.0, 8.0, 16.0], lam_u=[300.0, 1000.0, np.inf], lam_b=[3000.0, 10000.0, 30000.0],
            feats=["full"])
L0_KW = dict(half_life_days=540.0, ridge=4.0, carry=0.75, response="npxg")

_DATA = None


def data():
    global _DATA
    if _DATA is None:
        _DATA = StintData()
    return _DATA


# --------------------------------------------------------------------------- L0 reference
def l0_predictions() -> pd.DataFrame:
    f = PROC / "l0_predictions_all.parquet"
    if f.exists():
        return pd.read_parquet(f)
    matches, sides = ts_.load()
    mask = matches.season.isin(TUNE + TEST)
    p = ts_.rolling_predictions(matches, sides, L0_KW, mask, dixon_coles=True)
    p["pen_home"] = p.goals_home_pred / p.calib_c - p.npxg_home_pred
    p["pen_away"] = p.goals_away_pred / p.calib_c - p.npxg_away_pred
    p.to_parquet(f, index=False)
    return p


def to_match(pred_sides: pd.DataFrame, l0: pd.DataFrame, gk: pd.DataFrame | None = None) -> pd.DataFrame:
    """Side-level npxG forecasts -> match rows with W/D/L, using L0's conversion for that date."""
    h = pred_sides[pred_sides.home == 1][["match_id", "npxg_pred"]].rename(columns={"npxg_pred": "npxg_home_pred"})
    a = pred_sides[pred_sides.home == 0][["match_id", "npxg_pred"]].rename(columns={"npxg_pred": "npxg_away_pred"})
    base = l0[["match_id", "date", "season", "round", "home_team", "away_team", "npxg_home", "npxg_away", "home_score",
               "away_score", "outcome", "calib_c", "dc_rho", "pen_home", "pen_away"]]
    q = base.merge(h, on="match_id").merge(a, on="match_id")
    gh = q.calib_c * (q.npxg_home_pred + q.pen_home)
    ga = q.calib_c * (q.npxg_away_pred + q.pen_away)
    if gk is not None:   # goals the home side scores depend on the away keeper, and vice versa
        k = gk.set_index("match_id")
        gh = gh * q.match_id.map(k.mult_away).fillna(1).to_numpy()
        ga = ga * q.match_id.map(k.mult_home).fillna(1).to_numpy()
    P = np.array([outcome_probs(x, y, r) for x, y, r in zip(gh, ga, q.dc_rho)])
    q["goals_home_pred"], q["goals_away_pred"] = gh, ga
    q["p_home"], q["p_draw"], q["p_away"] = P[:, 0], P[:, 1], P[:, 2]
    return q


def scores(q: pd.DataFrame) -> dict:
    P = q[["p_home", "p_draw", "p_away"]].to_numpy()
    o = q.outcome.to_numpy()
    dev = np.r_[poisson_deviance(q.npxg_home, q.npxg_home_pred), poisson_deviance(q.npxg_away, q.npxg_away_pred)]
    return dict(n=len(q), rps=rps(P, o).mean(), log_loss=log_loss(P, o).mean(), npxg_deviance=dev.mean())


def per_match(q: pd.DataFrame) -> dict:
    q = q.sort_values("match_id")
    P = q[["p_home", "p_draw", "p_away"]].to_numpy()
    o = q.outcome.to_numpy()
    dev = (poisson_deviance(q.npxg_home, q.npxg_home_pred) + poisson_deviance(q.npxg_away, q.npxg_away_pred)) / 2
    return dict(rps=rps(P, o), log_loss=log_loss(P, o), npxg_deviance=dev)


# --------------------------------------------------------------------------- tuning
def _run(args):
    cfg, seasons = args
    return cfg, rolling(data(), cfg, seasons)


REFINE = [dict(lam_tau=8.0, lam_u=100.0, lam_b=10000.0), dict(lam_tau=8.0, lam_u=300.0, lam_b=1000.0),
          dict(lam_tau=32.0, lam_u=300.0, lam_b=10000.0), dict(lam_tau=8.0, lam_u=300.0, lam_b=10000.0, feats="compact")]


def configs():
    out = [Config(lam_tau=t, lam_u=u, lam_b=b, feats=f) for t, u, b, f in itertools.product(*GRID.values())
           if not (np.isinf(b) and f == "compact") and not (np.isinf(u) and np.isinf(b))]
    out += [Config(lam_tau=t, use_players=False) for t in [2.0, 4.0, 8.0, 16.0]]
    return out


def grid(l0: pd.DataFrame, cfgs=None, append=False) -> pd.DataFrame:
    """Tuning grid on 2024/25. The first pass is GRID; REFINE extends it where the best settings sat on an edge."""
    tune_l0 = l0[l0.season.isin(TUNE)]
    rows = []
    with Pool(2) as pool:
        for cfg, p in pool.imap_unordered(_run, [(c, TUNE) for c in (cfgs or configs())]):
            s = scores(to_match(p, tune_l0))
            rows.append(dict(key=cfg.key(), half_life=cfg.half_life, carry=cfg.carry, lam_tau=cfg.lam_tau,
                             lam_u=cfg.lam_u, lam_b=cfg.lam_b, feats=cfg.feats, use_players=cfg.use_players,
                             tune_npxg_dev=s["npxg_deviance"], tune_rps=s["rps"], tune_ll=s["log_loss"]))
            print(rows[-1], flush=True)
    g = pd.DataFrame(rows)
    if append:
        g = pd.concat([pd.read_csv(REPORTS / "l1_grid.csv"), g], ignore_index=True)
    g = g.sort_values("tune_npxg_dev")
    g.to_csv(REPORTS / "l1_grid.csv", index=False)
    return g


def choose(g: pd.DataFrame) -> dict:
    pl = g[g.use_players]
    best = pl.tune_npxg_dev.min()
    tied = pl[pl.tune_npxg_dev <= best * (1 + TIE_TOL)]
    r = tied.sort_values("tune_rps").iloc[0]
    out = dict(lam_tau=float(r.lam_tau), lam_u=float(r.lam_u), lam_b=float(r.lam_b), feats=r.feats,
               half_life=float(r.half_life), carry=float(r.carry), n_tied=int(len(tied)),
               tune=dict(npxg_dev=float(r.tune_npxg_dev), rps=float(r.tune_rps), log_loss=float(r.tune_ll)))
    (REPORTS / "l1_chosen.json").write_text(json.dumps(out, indent=2))
    return out


# --------------------------------------------------------------------------- report
def player_ratings(d: StintData, cfg: Config, as_of=None, min_minutes=900) -> pd.DataFrame:
    """o and d of every player as of the date, with his latest profile; minutes over the previous 365 days."""
    from outcome import fit, player_effects, unpack, ts
    as_of = ts(as_of) if as_of is not None else d.matches.date.max() + pd.Timedelta(days=1)
    th, info = fit(d, as_of, cfg)
    q = unpack(d, th)
    pt = d.pt[d.pt.date < as_of]
    rec = pt[pt.date >= as_of - pd.Timedelta(days=365)]
    mins = rec.groupby("player_id").share.sum() * 90
    last = pt.sort_values("date").groupby("player_id").tail(1)
    last = last[last.player_id.isin(mins[mins >= min_minutes].index)]
    o, dd, xo, xd = player_effects(d, th, info["zs"], last.match_id.to_numpy(), last.player_id.to_numpy())
    pm = pd.read_parquet(PROC / "player_match.parquet")[["match_id", "player_id", "player_name", "team"]]
    out = last[["match_id", "player_id", "team_id", "is_gk"]].merge(pm, on=["match_id", "player_id"])
    k = out.set_index("player_id")
    r = pd.DataFrame({"player_id": last.player_id.to_numpy(), "o": o, "d": dd, "o_prior": xo, "d_prior": xd})
    r = r.merge(out[["player_id", "player_name", "team", "is_gk"]], on="player_id")
    r["minutes_365d"] = r.player_id.map(mins).round(0)
    prof = d.prof.reset_index()
    grp = prof.sort_values("date").groupby("player_id").group.last()
    r["group"] = np.where(r.is_gk, "GK", r.player_id.map(grp))
    base = float(np.exp(q["mu"] + q["home"] / 2 + info["G"]))      # npxG per team-match, average teams, neutral
    r["att_npxg_per90"] = base * (np.exp(r.o) - 1)                    # change in own team's npxG per 90
    r["def_npxg_per90"] = base * (1 - np.exp(-r.d))                   # npxG per 90 kept from the opponent
    r["net_npxg_per90"] = r.att_npxg_per90 + r.def_npxg_per90
    return r.sort_values("net_npxg_per90", ascending=False).reset_index(drop=True), as_of


def main(run_grid=False):
    l0 = l0_predictions()
    if run_grid:
        grid(l0)
        grid(l0, [Config(**k) for k in REFINE], append=True)
    g = pd.read_csv(REPORTS / "l1_grid.csv")
    ch = choose(g)
    cfg = Config(lam_tau=ch["lam_tau"], lam_u=ch["lam_u"], lam_b=ch["lam_b"], feats=ch["feats"])
    team_only = g[~g.use_players].sort_values("tune_npxg_dev").iloc[0]
    cfg_team = Config(lam_tau=float(team_only.lam_tau), use_players=False)
    variants = {"L1": cfg, "Team effects on stints": cfg_team,
                "L1 profile prior only": Config(lam_tau=cfg.lam_tau, lam_u=np.inf, lam_b=cfg.lam_b if np.isfinite(cfg.lam_b) else 30000.0, feats=cfg.feats),
                "L1 without profile prior": Config(lam_tau=cfg.lam_tau, lam_u=cfg.lam_u if np.isfinite(cfg.lam_u) else 1000.0, lam_b=np.inf)}
    import gk as GK
    gkw = GK.wide(pd.read_parquet(PROC / "gk_asof.parquet"))
    # keeper factor: keep it if it improves the tuning-season RPS for L1
    tune_l0 = l0[l0.season.isin(TUNE)]
    p_tune = rolling(data(), cfg, TUNE)
    s_no, s_gk = scores(to_match(p_tune, tune_l0)), scores(to_match(p_tune, tune_l0, gkw))
    use_gk = s_gk["rps"] < s_no["rps"]
    ch["gk_factor"] = dict(tune_without=s_no, tune_with=s_gk, kept=bool(use_gk))

    test_l0 = l0[l0.season.isin(TEST)]
    with Pool(2) as pool:
        res = dict(pool.map(_run_named, [(n, c, TEST) for n, c in variants.items()]))
    preds = {"L0": test_l0.assign()}
    for n, p in res.items():
        preds[n] = to_match(p, test_l0, gkw if (use_gk and n == "L1") else None)
    preds["L1 with keeper factor" if not use_gk else "L1 without keeper factor"] = to_match(
        res["L1"], test_l0, None if use_gk else gkw)
    rows, pm_ = [], {}
    for n, q in preds.items():
        rows.append(dict(model=n, **scores(q)))
        pm_[n] = per_match(q)
    table = pd.DataFrame(rows)
    comps = []
    for a, b in [("L1", "L0"), ("Team effects on stints", "L0"), ("L1", "Team effects on stints"),
                 ("L1", "L1 profile prior only"), ("L1", "L1 without profile prior"),
                 ("L1", "L1 with keeper factor" if not use_gk else "L1 without keeper factor")]:
        for metric in ["rps", "log_loss", "npxg_deviance"]:
            dlt, lo, hi = paired_bootstrap(pm_[a][metric], pm_[b][metric])
            comps.append(dict(comparison=f"{a} minus {b}", metric=metric, diff=dlt, ci_low=lo, ci_high=hi))
    comps = pd.DataFrame(comps)
    by_season = pd.DataFrame([dict(model=n, season=s_, **scores(q[q.season == s_])) for n, q in preds.items()
                              for s_ in TEST])
    ratings, as_of = player_ratings(data(), cfg)
    preds["L1"].to_parquet(PROC / "l1_predictions.parquet", index=False)
    table.to_csv(REPORTS / "l1_test_scores.csv", index=False)
    comps.to_csv(REPORTS / "l1_comparisons.csv", index=False)
    by_season.to_csv(REPORTS / "l1_by_season.csv", index=False)
    ratings.to_csv(REPORTS / "player_ratings_current.csv", index=False)
    (REPORTS / "l1_chosen.json").write_text(json.dumps(ch, indent=2, default=float))
    summary = dict(chosen=ch, variants={n: c.key() for n, c in variants.items()}, ratings_as_of=str(as_of),
                   test=table.set_index("model").to_dict("index"))
    (REPORTS / "l1_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return table, comps, by_season, ratings


def _run_named(args):
    n, cfg, seasons = args
    return n, rolling(data(), cfg, seasons)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    t, c, b, r = main(run_grid="--grid" in sys.argv)
    print(t.round(4).to_string(index=False))
    print(c.round(4).to_string(index=False))
    print(b.round(4).to_string(index=False))
    print(r.head(20).round(3).to_string())
    print(r.tail(20).round(3).to_string())
