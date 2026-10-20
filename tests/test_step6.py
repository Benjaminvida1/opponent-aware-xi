"""Checks for the Step 6 robustness code (match-level L1, round-level bootstrap). Run from the repo root:
python tests/test_step6.py

Needs the processed tables through Step 3 (ingest.py ... stints.py)."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")
import outcome as OC  # noqa: E402
import robustness as RB  # noqa: E402

_ML = None


def ml():
    global _ML
    if _ML is None:
        _ML = RB.MatchLevelData()
    return _ML


def test_match_level_reconciles():
    """One row per directed team-match; npxG and minutes add up to the stints; each row's player weights sum
    to the average number of players on the pitch (about 11) and the team columns are plain indicators."""
    d, st = ml(), pd.read_parquet(OC.PROC / "stints.parquet")
    assert len(d.st) == st.groupby(["match_id", "team_id"]).ngroups
    a = d.st.set_index(["match_id", "team_id"]).npxg.sort_index()
    b = st.groupby(["match_id", "team_id"]).npxg.sum().sort_index()
    assert np.allclose(a.to_numpy(), b.to_numpy())
    assert np.isclose(d.st.duration.sum(), st.duration.sum())
    w = np.asarray(d.PA.sum(1)).ravel()
    assert np.median(w) == 11 and w.min() > 9 and w.max() < 12.5
    assert np.allclose(np.asarray(d.TA.sum(1)).ravel(), 1) and np.allclose(np.asarray(d.TD.sum(1)).ravel(), 1)
    assert (d.D[:, 2:] == 0).all()                       # no game state or man count at match level
    assert d.st.date.is_monotonic_increasing


def test_match_level_fit_is_as_of():
    """Scrambling the npxG of every team-match on or after the fit date leaves the fit unchanged."""
    d = ml()
    cfg = OC.Config(lam_u=300.0, lam_b=10000.0)
    as_of = pd.Timestamp("2024-12-01")
    th0, _ = OC.fit(d, as_of, cfg, prior={})
    later = (d.st.date >= as_of).to_numpy()
    saved = d.st.npxg.copy()
    rng = np.random.default_rng(0)
    d.st.loc[later, "npxg"] = rng.permutation(d.st.loc[later, "npxg"].to_numpy()) * 3
    th1, _ = OC.fit(d, as_of, cfg, prior={})
    d.st["npxg"] = saved
    assert np.allclose(th0, th1)


def test_cluster_bootstrap():
    """Same mean as the data; with one observation per cluster it matches the match-level bootstrap; with
    perfectly correlated clusters the interval widens."""
    rng = np.random.default_rng(1)
    d = rng.normal(0.0, 1.0, 400)
    m, lo, hi, k = RB.cluster_bootstrap(d, np.arange(400))
    m2, lo2, hi2 = RB.paired_bootstrap(d, np.zeros(400))
    assert np.isclose(m, d.mean()) and k == 400
    assert abs((hi - lo) - (hi2 - lo2)) < 0.05
    shocks = np.repeat(rng.normal(0, 1, 40), 10)
    d2 = rng.normal(0, 0.3, 400) + shocks
    _, a, b, _ = RB.cluster_bootstrap(d2, np.repeat(np.arange(40), 10))
    _, c, e = RB.paired_bootstrap(d2, np.zeros(400))
    assert (b - a) > 1.5 * (e - c)


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
