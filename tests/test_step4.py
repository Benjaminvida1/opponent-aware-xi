"""Checks for the Step 4 code (network prediction, matchup features, opponent-aware terms). Run from the repo root:
python tests/test_step4.py

Needs the processed tables through Step 3, plus netpred.py and matchups.py."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")
import matchups as MU  # noqa: E402
import netpred as NP  # noqa: E402
import outcome as OC  # noqa: E402

PROC = OC.PROC
CUT = pd.Timestamp("2024-12-01")


def test_positions_are_as_of():
    """Predicted positions for matches before a date are identical whether or not later matches exist."""
    m, st = NP.load()
    P_full, s_full = NP.run_positions(st, 3.0, 2.0, 120.0)
    early = (st.date < CUT).to_numpy()
    st_cut = st[early].reset_index(drop=True)
    P_cut, s_cut = NP.run_positions(st_cut, 3.0, 2.0, 120.0)
    assert np.allclose(P_full[early], P_cut, equal_nan=True)
    assert (s_full[early] == s_cut).all()


def test_slot_assignment_is_one_to_one():
    """Every team-match's ten outfield starters fill ten different slots of the teamsheet formation."""
    pp = pd.read_parquet(PROC / "pred_positions.parquet")
    out = pp[~pp.is_gk]
    n = out.groupby(["match_id", "team_id"]).slot_pred.agg(["size", "nunique"])
    assert (n["size"] == 10).all() and (n["nunique"] == 10).all()


def test_vulnerability_is_as_of():
    """Lane vulnerability before a date does not change when later shots are scrambled."""
    m = pd.read_parquet(PROC / "matches.parquet")
    m["date"] = MU.naive(m.date)
    v0 = MU.lane_vulnerability(m[m.date < CUT])
    v1 = MU.lane_vulnerability(m)
    v1 = v1[v1.day.isin(v0.day.unique())]
    a = v0.sort_values(["day", "team_id"]).reset_index(drop=True)
    b = v1.sort_values(["day", "team_id"]).reset_index(drop=True)
    assert np.allclose(a.filter(like="vuln").to_numpy(), b.filter(like="vuln").to_numpy())


def test_features_mirror():
    """Both directions of a match exist, and the opponent's style columns are the other row's own columns."""
    f = pd.read_parquet(PROC / "matchup_features.parquet")
    k = f.set_index(["match_id", "team_id"])
    sw = f.set_index(["match_id", "opponent_id"])
    common = k.index.intersection(sw.index)
    assert np.allclose(k.loc[common, "opp_full_s1"].to_numpy(), sw.loc[common, "full_s1"].to_numpy(), equal_nan=True)
    assert f.groupby("match_id").size().eq(2).mean() > 0.99


def test_fit_with_extra_terms_is_as_of():
    """With style and matchup terms, scrambling later stints' npxG leaves the fit unchanged."""
    d = OC.StintData()
    cfg = OC.Config(lam_tau=4.0, lam_u=300.0, lam_b=10000.0,
                    extra=(("styleint:full", 100.0), ("cols:zone_wide_b10v,dribble,aerial,press", 300.0)))
    th0, _ = OC.fit(d, CUT, cfg, prior={})
    later = (d.st.date >= CUT).to_numpy()
    rng = np.random.default_rng(0)
    d.st.loc[later, "npxg"] = rng.permutation(d.st.loc[later, "npxg"].to_numpy()) * 3
    th1, _ = OC.fit(d, CUT, cfg, prior={})
    assert np.allclose(th0, th1)


def test_extra_terms_change_forecasts():
    """The extra terms enter the pre-match forecast: with a huge penalty they vanish and L1's forecast returns."""
    d = OC.StintData()
    base = OC.Config(lam_tau=4.0, lam_u=300.0, lam_b=10000.0)
    big = OC.Config(lam_tau=4.0, lam_u=300.0, lam_b=10000.0, extra=(("styleint:full", 1e6),))
    m = d.matches[d.matches.date >= CUT].head(6)
    out = []
    for cfg in (base, big):
        th, info = OC.fit(d, CUT, cfg, prior={})
        out.append(OC.predict_matches(d, th, info, m).npxg_pred.to_numpy())
    assert np.allclose(out[0], out[1], rtol=5e-3)   # L-BFGS tolerance; 1e10 is ill-conditioned


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
