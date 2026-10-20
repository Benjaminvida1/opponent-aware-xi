"""Checks for the Step 2 code (player profiles and roles). Run from the repo root: python tests/test_step2.py

Needs the processed tables (ingest.py, network.py, style.py, profiles.py, roles.py)."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")
import profiles as PR  # noqa: E402
import roles as RL  # noqa: E402


def test_shrinkage_recovers_known_prior():
    """Synthetic Poisson players: the moment estimator finds the between-player variance, and the shrunk
    rate predicts the true rate better than the raw rate."""
    rng = np.random.default_rng(1)
    P, mu, tau = 400, 2.0, 0.6
    lam = rng.gamma((mu / tau) ** 2, tau ** 2 / mu, P)
    E = rng.uniform(0.5, 30, P)
    C = rng.poisson(lam * E).astype(float)
    nu = np.array([[mu]] * (len(PR.GROUPS) + 1))
    m, t2, b = PR.prior_moments(C[:, None], E[:, None], np.zeros(P, int), nu, None)
    assert abs(m[0, 0] - mu) < 0.15, m[0, 0]
    assert abs(np.sqrt(t2[0, 0]) - tau) < 0.15, np.sqrt(t2[0, 0])
    shr = (C + b[0, 0] * m[0, 0]) / (E + b[0, 0])
    assert ((shr - lam) ** 2).mean() < ((C / E - lam) ** 2).mean()


def test_profiles_are_as_of():
    """Scrambling every observation after a date leaves every profile on or before that date unchanged."""
    obs = PR.Obs()
    cut_day = obs.days[np.searchsorted(obs.days, pd.Timestamp("2024-12-01", tz="UTC"))]
    chosen = pd.read_csv(PR.REPORTS / "profile_settings.csv")
    target = obs.dayidx <= np.searchsorted(obs.days, cut_day)
    PR._GROUP_CACHE.clear(); PR._NU_CACHE.clear()
    base = PR.build(obs, PR.prepare(obs), chosen, target)["profile"]

    later = obs.dayidx > np.searchsorted(obs.days, cut_day)
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(later)
    perm = rng.permutation(idx)
    for arr in (obs.V, obs.X, obs.PF, obs.raw):
        arr[idx] = arr[perm] * rng.uniform(0.5, 2.0, (len(idx), 1))
    obs.start_group[idx] = rng.integers(-1, len(PR.GROUPS), len(idx))
    obs.poss[idx] = rng.uniform(0.3, 0.7, len(idx))
    obs._groups_asof()
    PR._GROUP_CACHE.clear(); PR._NU_CACHE.clear()
    scr = PR.build(obs, PR.prepare(obs), chosen, target)["profile"]
    assert np.allclose(base, scr, equal_nan=True), np.nanmax(np.abs(base - scr))


def test_role_fits_use_earlier_seasons_only():
    """Each season's role fit is trained on player-seasons from earlier seasons only, and memberships exist
    only from 2024/25 on."""
    roles = pd.read_parquet(RL.PROC / "player_roles_asof.parquet")
    assert roles.season.min() == "2024/2025"
    src = Path(RL.__file__).read_text()
    assert "train = ps[ps.season.isin(SEASONS[:s_idx])]" in src
    k = RL.K_CHOSEN
    p = roles[[f"role_{r}" for r in range(1, k + 1)]].to_numpy()
    assert np.allclose(p.sum(1), 1.0)


def test_profile_table_covers_every_squad_row():
    prof = pd.read_parquet(PR.PROC / "player_profile_asof.parquet")
    pm = pd.read_parquet(PR.PROC / "player_match.parquet")
    out = pm[~pm.is_gk]
    assert len(prof) == len(out)
    # only the first match day of the data (nothing earlier to learn from) has no profile
    first = prof.date == prof.date.min()
    assert prof.loc[~first, PR.FEAT].notna().all().all()
    assert prof.loc[first, PR.FEAT].isna().all().all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
