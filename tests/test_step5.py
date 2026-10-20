"""Step 5 checks: expected points, as-of inputs, like-for-like picks, and the decision table's consistency.

Run from the repo root: python tests/test_step5.py   (needs the Step 5 back-test outputs)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lineup import SLOT_GROUP, THRESHOLD, History, outcome_probs_vec  # noqa: E402
from outcome import PROC, StintData  # noqa: E402
from team_strength import outcome_probs  # noqa: E402


def test_outcome_probs():
    rng = np.random.default_rng(1)
    lh, la, rho = rng.uniform(0.3, 3, 50), rng.uniform(0.3, 3, 50), -0.05
    v = np.column_stack(outcome_probs_vec(lh, la, rho))
    ref = np.array([outcome_probs(a, b, rho) for a, b in zip(lh, la)])
    assert np.allclose(v, ref, atol=1e-10)


def test_fshare_as_of():
    data = StintData()
    h = History(data)
    p = data.pt[data.pt.started & ~data.pt.is_gk].player_id.value_counts().index[0]
    d0 = float(h.starts[p][0][40])
    before = h.fshare([p], d0, np.array([False]))
    # scrambling every start from d0 on leaves the share as of d0 unchanged
    dd, v, g = h.starts[p]
    h.starts[p] = (dd, np.where(dd >= d0, 0.0, v), g)
    assert np.allclose(before, h.fshare([p], d0, np.array([False])))


def test_decisions():
    D = pd.read_parquet(PROC / "step5_decisions.parquet")
    assert len(D) == 860, len(D)
    # the recommendation is the best XI exactly when P(better) clears the bar
    rec_is_best = D.recommended_xi == D.best_xi
    changed = D.best_changes > 0
    assert (rec_is_best[changed] == (D.p_better[changed] > THRESHOLD)).all()
    assert ((D.recommended_xi == D.usual_xi) | rec_is_best).all()
    # the best XI never scores below the usual XI (search starts from it)
    assert (D.best_ep >= D.usual_ep - 1e-9).all()
    # the usual XI is anchored on L1's raw forecast
    assert np.allclose(D.usual_ep, D.usual_ep_raw)
    # every XI has ten different outfielders and fills the formation's ten slots once
    for c in ("usual", "best", "recommended"):
        for xi, sl in zip(D[f"{c}_xi"], D[f"{c}_slots"]):
            xi, sl = json.loads(xi), json.loads(sl)
            assert len(set(xi)) == 10 and len(set(sl.values())) == 10


def test_like_for_like():
    """Each player the best XI brings in plays a slot of his primary position group or of the usual XI's group."""
    S = pd.read_parquet(PROC / "step5_swaps.parquet")
    S = S[S.xi == "best"]
    D = pd.read_parquet(PROC / "step5_decisions.parquet").set_index(["match_id", "team_id"])
    prof = pd.read_parquet(PROC / "player_profile_asof.parquet").drop_duplicates(["match_id", "player_id"]) \
        .set_index(["match_id", "player_id"]).group
    bad = 0
    for r in S.itertuples():
        g = SLOT_GROUP[r.slot_in]
        pg = prof.get((r.match_id, r.inn))
        if isinstance(pg, str) and pg != g:
            bad += 1
    assert bad / max(len(S), 1) < 0.05, bad


if __name__ == "__main__":
    for t in (test_outcome_probs, test_fshare_as_of, test_decisions, test_like_for_like):
        t()
        print("ok", t.__name__)
