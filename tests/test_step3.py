"""Checks for the Step 3 code (stints, goalkeeper prior, player-effects model). Run from the repo root:
python tests/test_step3.py

Needs the processed tables (ingest.py, network.py, style.py, profiles.py, roles.py, stints.py, gk.py)."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")
import gk as GK  # noqa: E402
import outcome as OC  # noqa: E402

PROC = OC.PROC


def test_stints_reconcile():
    """Stints cover every minute of every match, carry every non-penalty shot, and nearly always have
    eleven players a side unless a player was sent off."""
    st = pd.read_parquet(PROC / "stints.parquet")
    shots = pd.read_parquet(PROC / "shots.parquet")
    s = shots[(shots.is_own_goal == 0) & (shots.situation_group != "penalty")]
    a = st.groupby(["match_id", "team_id"]).npxg.sum()
    b = s.groupby(["match_id", "team_id"]).xg.sum().reindex(a.index).fillna(0)
    assert np.allclose(a, b)
    gaps = st.sort_values(["match_id", "stint"]).groupby(["match_id", "team_id"]).apply(
        lambda g: np.abs(g.start.to_numpy()[1:] - g.end.to_numpy()[:-1]).max() if len(g) > 1 else 0)
    assert gaps.max() < 1e-9
    assert (st.loc[st.manadv == 0, "n_on"] == st.loc[st.manadv == 0, "n_on_opp"]).mean() > 0.99
    assert st.duration.gt(0).all()


def test_goal_ends_its_stint():
    """The score changes only between stints: within a stint the goal difference is fixed, and each goal
    shot sits in the stint that ends at the goal."""
    st = pd.read_parquet(PROC / "stints.parquet")
    ss = pd.read_parquet(PROC / "shot_stints.parquet")
    shots = pd.read_parquet(PROC / "shots.parquet")
    g = shots[shots.is_goal == 1].merge(ss, on=["shot_id", "match_id"])
    ends = st.drop_duplicates(["match_id", "stint"]).set_index(["match_id", "stint"]).end
    e = ends.reindex(pd.MultiIndex.from_frame(g[["match_id", "stint"]])).to_numpy()
    assert np.allclose(e, g.t.to_numpy())


def test_fit_is_as_of():
    """Scrambling the npxG of every stint on or after the fit date leaves the fitted model unchanged."""
    d = OC.StintData()
    cfg = OC.Config(lam_u=1000.0, lam_b=30000.0)
    as_of = pd.Timestamp("2024-12-01")
    th0, _ = OC.fit(d, as_of, cfg, prior={})
    later = (d.st.date >= as_of).to_numpy()
    rng = np.random.default_rng(0)
    d.st.loc[later, "npxg"] = rng.permutation(d.st.loc[later, "npxg"].to_numpy()) * 3
    th1, _ = OC.fit(d, as_of, cfg, prior={})
    assert np.allclose(th0, th1)


def test_gk_prior_is_as_of_and_shrunk():
    """The keeper ratio for a match uses only earlier shots, and shrinkage pulls every keeper toward 1."""
    gk = pd.read_parquet(PROC / "gk_asof.parquet")
    s = GK.keeper_shots()
    r = gk[gk.season == "2025/2026"].iloc[100]
    past = s[(s.keeper_id == r.keeper_id) & (s.date < r.date)]
    assert np.isclose(past.is_goal.sum(), r.G) and np.isclose(past.xgot.sum(), r.X)
    raw = (gk.G / gk.X / gk.rbar).where(gk.X > 20)
    assert (np.abs(gk.mult - 1) <= np.abs(raw - 1) + 1e-9)[raw.notna()].all()


def test_credibility_recovers_variance():
    """Synthetic keepers: shrunk ratios beat raw ratios at recovering the true ratio."""
    rng = np.random.default_rng(3)
    K = 300
    true = rng.normal(1.0, 0.08, K)
    shots = rng.integers(10, 300, K)
    G, X = np.zeros(K), np.zeros(K)
    for k in range(K):
        xg = rng.uniform(0.05, 0.6, shots[k])
        G[k] = rng.binomial(1, np.clip(xg * true[k], 0, 1)).sum()
        X[k] = xg.sum()
    nu = 0.65
    rh, rbar, b = GK.credibility(G, X, nu)
    assert np.isfinite(b)
    assert ((rh - true) ** 2).mean() < ((G / X - true) ** 2).mean()


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
