"""Checks for the Step 1 code (src/network.py, src/style.py).

Run from the repo root:  python tests/test_step1.py   (or: pytest tests/)
The last test needs data/processed (run src/ingest.py and src/network.py first).
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import network as N  # noqa: E402


def test_template_recovered_exactly():
    """A team standing exactly on a template (any scale, shifted, players shuffled) gets that template's slots."""
    t = N._template_arrays()["4-2-3-1"]
    rng = np.random.default_rng(0)
    order = rng.permutation(10)
    xy = t["xy"][order] * np.array([0.7, 0.9]) + np.array([12.0, -3.0])
    f = N.infer_formation(xy, only=["4-2-3-1"])
    assert f["cost"] < 1e-9
    assert [f["slots"][k] for k in range(10)] == [t["slots"][j] for j in order]


def test_hungarian_beats_or_equals_every_permutation_sample():
    rng = np.random.default_rng(1)
    xy = rng.uniform([20, 5], [80, 63], (10, 2))
    f = N.infer_formation(xy, only=["3-4-2-1"])
    t = N._template_arrays()["3-4-2-1"]
    z = N._standardize(xy)
    for _ in range(2000):
        p = rng.permutation(10)
        assert ((z - t["z"][p]) ** 2).sum() >= f["cost"] - 1e-9


def test_line_detection_is_the_exact_1d_kmeans_optimum():
    rng = np.random.default_rng(2)
    for _ in range(20):
        x = rng.uniform(25, 80, 10)
        label, sil, lab = N.detect_lines(x)
        k = lab.max() + 1
        xs = np.sort(x)
        best = min(sum(((g - g.mean()) ** 2).sum() for g in np.split(xs, c))
                   for c in itertools.combinations(range(1, 10), k - 1))
        got = sum(((x[lab == j] - x[lab == j].mean()) ** 2).sum() for j in range(k))
        assert abs(got - best) < 1e-9
        assert label == "-".join(str((lab == j).sum()) for j in range(k))
        # lines are ordered back to front
        assert all(x[lab == j].max() <= x[lab == j + 1].min() for j in range(k - 1))


def test_line_detection_clear_shape():
    x = np.array([30, 31, 32, 33, 50, 51, 52, 53, 70, 71.0])
    assert N.detect_lines(x)[0] == "4-4-2"


def test_gini():
    assert N.gini(np.ones(10)) == 0
    assert abs(N.gini(np.r_[np.zeros(9), 1.0]) - 0.9) < 1e-12


def test_game_state_shares():
    m = pd.DataFrame(dict(match_id=[1], home_team_id=[10], away_team_id=[20]))
    s = pd.DataFrame(dict(match_id=[1, 1], minute=[30, 60], minute_added=[0, 0], period=[1, 2],
                          scoring_team_id=[10, 20], is_goal=[1, 1]))
    g = N.game_state_shares(m, s).set_index("team_id")
    assert np.isclose(g.loc[10, "share_leading"], 30 / 90)
    assert np.isclose(g.loc[10, "share_level"], 60 / 90)
    assert np.isclose(g.loc[20, "share_trailing"], 30 / 90)
    assert np.isclose(g[["share_leading", "share_level", "share_trailing"]].sum(axis=1), 1).all()


def test_style_fit_ignores_future_matches():
    """The as-of rule: scrambling every match on or after the fit date leaves the fitted style unchanged."""
    if not (ROOT / "data" / "processed" / "team_match_network.parquet").exists():
        print("skipped: run src/ingest.py and src/network.py first")
        return
    import style as S
    tm = S.load_table()
    d = S.Design(tm)
    Z = tm[S.STYLE_METRICS].to_numpy(float)
    use_poss = np.array([m == "log_ppda" for m in S.STYLE_METRICS])
    as_of = tm[tm.season == "2025/2026"].date.iloc[200]

    def fit(Zx):
        w = d.weights(as_of, 2, 240.0, 0.75)
        Xw = d.X * w[:, None]
        return d.solve(Xw.T @ d.X, Xw.T @ np.nan_to_num(Zx), 3.0, 10.0, use_poss)

    Z2 = Z.copy()
    fut = (tm.date >= as_of).to_numpy()
    Z2[fut] = np.random.default_rng(3).normal(size=Z2[fut].shape) * 100
    assert np.abs(fit(Z) - fit(Z2)).max() == 0.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
