"""Step 6: two robustness checks for the write-up (methodology §15.2).

Run:  python src/robustness.py              (both checks; needs Steps 0-4 rebuilt, incl. run_l23.py for L2/L3)
      python src/robustness.py --no-match   (round-level bootstrap only, reusing a saved match-level L1)

1. Round-level bootstrap. The paired bootstrap of §15.2 resamples single matches. Matches in the same round share
   conditions (weather, fixture congestion, a round of surprising results), so their score differences may be
   correlated. Here whole rounds (season x official round, 39 clusters of about 10 matches in the test period) are
   resampled instead, for every comparison in the model ladder. Same 5,000 resamples and 95% percentile intervals.

2. Match-level L1 (no stints). The same player-effects model fitted on one row per team-match: the response is the
   team's npxG over the whole match, the exposure the match length, and each player enters with his share of the
   match on the pitch (starters and substitutes alike, weighted by minutes). Game state and man count are dropped
   (they change within a match and need stints). The pre-match forecast, the level calibration and the goals
   conversion are exactly L1's, so the two differ only in how the player effects are fitted.
   Settings: L1's (team 4, personal 300, box-score 10,000), and, as a check, the best of a small grid on 2024/25 (extended toward weaker box-score penalties when the best sits on that edge)
   chosen with the L1 rule (npxG deviance, ties within 0.5% by RPS).

Outputs (reports/): robustness_round_bootstrap.csv, robustness_match_level.csv, robustness_match_level_grid.csv,
  robustness_summary.json; data/processed/l1_match_level_predictions.parquet
"""
from __future__ import annotations

import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from outcome import N_DENSE, PROC, REPORTS, TEST, TUNE, Config, StintData, rolling  # noqa: E402
from run_l1 import l0_predictions, per_match, scores, to_match  # noqa: E402
from team_strength import base_rate_predictions, log_loss, paired_bootstrap, rps  # noqa: E402

N_BOOT = 5000
TIE_TOL = 0.005
L1_KW = dict(lam_tau=4.0, lam_u=300.0, lam_b=10000.0)
ML_GRID = dict(lam_u=[100.0, 300.0, 1000.0, np.inf], lam_b=[3000.0, 10000.0, 30000.0])
ML_REFINE = dict(lam_u=[300.0, np.inf], lam_b=[300.0, 1000.0])


# --------------------------------------------------------------------------- 1. round-level bootstrap
def cluster_bootstrap(d: np.ndarray, clusters: np.ndarray, n_boot: int = N_BOOT, seed: int = 0):
    """Mean of d with a 95% interval, resampling whole clusters (with replacement)."""
    rng = np.random.default_rng(seed)
    codes, inv = np.unique(clusters, return_inverse=True)
    k = len(codes)
    s = np.bincount(inv, weights=d, minlength=k)
    n = np.bincount(inv, minlength=k).astype(float)
    pick = rng.integers(0, k, (n_boot, k))
    boots = s[pick].sum(1) / n[pick].sum(1)
    return float(d.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)), k


def wdl_only(q: pd.DataFrame) -> dict:
    q = q.sort_values("match_id")
    P = q[["p_home", "p_draw", "p_away"]].to_numpy()
    o = q.outcome.to_numpy()
    return dict(rps=rps(P, o), log_loss=log_loss(P, o))


# --------------------------------------------------------------------------- 2. match-level L1
class MatchLevelData(StintData):
    """StintData collapsed to one directed row per team-match; players weighted by their share of the match."""

    def __init__(self):
        super().__init__()
        st = self.st
        key = st.match_id.astype(str) + "|" + st.team_id.astype(str)
        codes, inv = np.unique(key.to_numpy(), return_inverse=True)
        n_new = len(codes)
        e = st.e.to_numpy()
        E = np.bincount(inv, weights=e, minlength=n_new)
        W = sp.csr_matrix((e / E[inv], (inv, np.arange(len(st)))), shape=(n_new, len(st)))   # time-share weights
        first = pd.Series(np.arange(len(st))).groupby(inv).first().to_numpy()
        new = st.iloc[first][["match_id", "team_id", "opponent_id", "home", "date", "season", "season_idx"]].copy()
        new["stint"] = 0
        new["duration"] = np.bincount(inv, weights=st.duration.to_numpy(), minlength=n_new)
        new["npxg"] = np.bincount(inv, weights=st.npxg.to_numpy(), minlength=n_new)
        new["e"] = E
        new["state"], new["manadv"] = 0, 0
        order = np.lexsort((1 - new.home.to_numpy(), new.match_id.to_numpy(), new.date.to_numpy()))
        P = sp.csr_matrix((np.ones(n_new), (np.arange(n_new), order)), shape=(n_new, n_new))   # row k <- old row order[k]
        W = P @ W
        self.st = new.iloc[order].reset_index(drop=True)
        D = np.zeros((n_new, N_DENSE))
        D[:, 0] = 1
        D[:, 1] = self.st.home
        self.D = D                                              # no game state or man count without stints
        agg = lambda M: (W @ M).tocsr() if sp.issparse(M) else W @ M
        self.TA, self.TD = agg(self.TA), agg(self.TD)
        self.PA, self.PD = agg(self.PA), agg(self.PD)
        dense = lambda M: np.asarray(W @ M)
        self.SxA = [dense(M) for M in self.SxA]
        self.NfA = [dense(M) for M in self.NfA]
        self.SxD = [dense(M) for M in self.SxD]
        self.NfD = [dense(M) for M in self.NfD]
        self.SxA_t, self.NfA_t = sum(self.SxA), sum(self.NfA)
        self.SxD_t, self.NfD_t = sum(self.SxD), sum(self.NfD)


_ML = None


def ml_data():
    global _ML
    if _ML is None:
        _ML = MatchLevelData()
    return _ML


def _run_ml(args):
    name, cfg, seasons = args
    return name, cfg, rolling(ml_data(), cfg, seasons)


def match_level(l0: pd.DataFrame):
    tune_l0, test_l0 = l0[l0.season.isin(TUNE)], l0[l0.season.isin(TEST)]
    d = ml_data()
    # sanity: the collapsed data keep every team-match's npxG and minutes
    chk = dict(rows=int(len(d.st)), npxg_total=float(d.st.npxg.sum()), minutes_total=float(d.st.duration.sum()))
    gpath = REPORTS / "robustness_match_level_grid.csv"
    old = pd.read_csv(gpath) if gpath.exists() else pd.DataFrame(columns=["key"])

    def run_grid(cfgs):
        todo = [c for c in cfgs if c.key() not in set(old.key)]
        if todo:
            with Pool(2) as pool:
                tune = pool.map(_run_ml, [(c.key(), c, TUNE) for c in todo])
            new = pd.DataFrame([dict(key=k, lam_u=c.lam_u, lam_b=c.lam_b,
                                     **{f"tune_{m}": v for m, v in scores(to_match(p, tune_l0)).items()})
                                for k, c, p in tune])
            return pd.concat([old, new], ignore_index=True)
        return old

    old = run_grid([Config(lam_tau=4.0, lam_u=u, lam_b=b) for u, b in itertools.product(*ML_GRID.values())])
    # the best setting sat on the grid's weakest box-score penalty: extend it (as run_l1.py's REFINE does)
    if old.loc[old.tune_npxg_deviance.idxmin(), "lam_b"] <= min(ML_GRID["lam_b"]):
        old = run_grid([Config(lam_tau=4.0, lam_u=u, lam_b=b) for u, b in itertools.product(ML_REFINE["lam_u"], ML_REFINE["lam_b"])])
    g = old.sort_values("tune_npxg_deviance")
    g.to_csv(gpath, index=False)
    best = g.tune_npxg_deviance.min()
    r = g[g.tune_npxg_deviance <= best * (1 + TIE_TOL)].sort_values("tune_rps").iloc[0]
    tuned = Config(lam_tau=4.0, lam_u=float(r.lam_u), lam_b=float(r.lam_b))
    jobs = [("Match-level L1 (L1 settings)", Config(**L1_KW), TEST)]
    if tuned.key() != Config(**L1_KW).key():
        jobs.append(("Match-level L1 (tuned)", tuned, TEST))
    with Pool(2) as pool:
        res = pool.map(_run_ml, jobs)
    preds = {n: to_match(p, test_l0) for n, _, p in res}
    if "Match-level L1 (tuned)" not in preds:
        preds["Match-level L1 (tuned)"] = preds["Match-level L1 (L1 settings)"]
    pd.concat([q.assign(model=n) for n, q in preds.items()], ignore_index=True).to_parquet(
        PROC / "l1_match_level_predictions.parquet", index=False)
    return preds, dict(check=chk, tuned=dict(lam_u=tuned.lam_u, lam_b=tuned.lam_b), grid_best=r.to_dict())


# --------------------------------------------------------------------------- main
def main(run_match=True):
    l0 = l0_predictions()
    test_l0 = l0[l0.season.isin(TEST)].copy()
    outcome = test_l0[["match_id", "outcome"]]
    m = pd.read_parquet(PROC / "matches.parquet")
    o = pd.read_parquet(PROC / "odds.parquet")
    preds = {"L0": test_l0,
             "Base rates": base_rate_predictions(m, m[m.match_id.isin(test_l0.match_id)]).merge(outcome, on="match_id"),
             "Market": o[["match_id", "p_home", "p_draw", "p_away"]].merge(outcome, on="match_id"),
             "L1": pd.read_parquet(PROC / "l1_predictions.parquet")}
    f23 = PROC / "l23_all_predictions.parquet"
    if f23.exists():
        a = pd.read_parquet(f23)
        for n, q in a.groupby("model"):
            if n != "L1":
                preds[n] = q.drop(columns="model")
    info = {}
    if run_match:
        mp, info = match_level(l0)
        preds.update(mp)
    else:
        f = PROC / "l1_match_level_predictions.parquet"
        if f.exists():
            for n, q in pd.read_parquet(f).groupby("model"):
                preds[n] = q.drop(columns="model")

    rnd = test_l0.set_index("match_id")
    cluster = (rnd.season.astype(str) + " R" + rnd["round"].astype(str))
    full = {n for n, q in preds.items() if "npxg_home_pred" in q}
    pm_ = {n: (per_match(q) if n in full else wdl_only(q)) for n, q in preds.items()}
    ids = np.sort(test_l0.match_id.to_numpy())
    for n, q in preds.items():
        assert np.array_equal(np.sort(q.match_id.to_numpy()), ids), n
    cl = cluster.reindex(ids).to_numpy()

    pairs = [("L0", "Base rates"), ("L0", "Market"), ("L1", "L0"), ("L1", "Market"), ("L2", "L1"), ("L3", "L2"),
             ("L3", "L1"), ("L1 + matchups only", "L1"), ("L3", "L3 without network"),
             ("L3 network geometry only", "L3 without network"),
             ("Match-level L1 (L1 settings)", "L1"), ("Match-level L1 (L1 settings)", "L0"),
             ("Match-level L1 (tuned)", "L1"), ("Match-level L1 (tuned)", "L0")]
    rows = []
    for a, b in pairs:
        if a not in pm_ or b not in pm_:
            continue
        for metric in ["rps", "log_loss", "npxg_deviance"]:
            if metric not in pm_[a] or metric not in pm_[b]:
                continue
            dvec = pm_[a][metric] - pm_[b][metric]
            dm, lo_m, hi_m = paired_bootstrap(pm_[a][metric], pm_[b][metric], n_boot=N_BOOT)
            dr, lo_r, hi_r, k = cluster_bootstrap(dvec, cl)
            rows.append(dict(comparison=f"{a} minus {b}", metric=metric, diff=dm, ci_low_match=lo_m, ci_high_match=hi_m,
                             ci_low_round=lo_r, ci_high_round=hi_r, width_ratio=(hi_r - lo_r) / (hi_m - lo_m),
                             n_rounds=k, excludes_zero_match=bool(lo_m > 0 or hi_m < 0),
                             excludes_zero_round=bool(lo_r > 0 or hi_r < 0)))
    rb = pd.DataFrame(rows)
    rb.to_csv(REPORTS / "robustness_round_bootstrap.csv", index=False)

    tab = pd.DataFrame([dict(model=n, **(scores(q) if n in full else
                                          {k: float(np.mean(v)) for k, v in pm_[n].items()}))
                        for n, q in preds.items()])
    tab.to_csv(REPORTS / "robustness_match_level.csv", index=False)
    by_season = pd.DataFrame([dict(model=n, season=s, **scores(q[q.season == s])) for n, q in preds.items()
                              if n in full and n.startswith(("L0", "L1", "Match")) for s in TEST])
    summary = dict(n_matches=int(len(ids)), n_rounds=int(len(np.unique(cl))), match_level=info,
                   scores=tab.set_index("model").to_dict("index"),
                   by_season=by_season.to_dict("records"))
    (REPORTS / "robustness_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return rb, tab, by_season


if __name__ == "__main__":
    pd.set_option("display.width", 240)
    rb, tab, bs = main(run_match="--no-match" not in sys.argv)
    print(tab.round(4).to_string(index=False))
    print(rb.round(4).to_string(index=False))
    print(bs.round(4).to_string(index=False))
