"""Player profiles (methodology §4.2-4.9): shrunk, recency-weighted, possession- and opponent-adjusted
rates for every outfield player in every matchday squad, as of the match date.

Run:  python src/profiles.py            (uses cached tuning results if present)
      python src/profiles.py --grid     (re-runs the tuning grid on 2024/25)

Needs data/processed/ from ingest.py and network.py.

Features (FEATURES below), each of one kind:
  count  per-90 rate of a count or of an expected-value stat (npxG, xA, metres of progression)
  prop   success share (successes / attempts)
  mean   a per-match value averaged over minutes (position relative to the team, betweenness, pass share)

One pass over the 360 match days. For every player and feature the code keeps decayed sums
    C = sum_m w_m * value_m * exposure_m,   E = sum_m w_m * exposure_m,
    w_m = 0.5 ** (age_days / half_life) * carry ** (seasons back) * (omega_sub if he came off the bench)
(exposure = minutes/90, or attempts for a share). Just before each day the sums give each squad player's
profile for that day's match, so a profile only ever uses earlier matches (§2.3).

Shrinkage (§4.4-4.6) uses one empirical-Bayes rule for all three kinds (Buhlmann credibility):
    profile = (C + b * mu_g) / (E + b),    b = nu_g / tau2_g
mu_g = position-group mean, tau2_g = between-player variance of true rates in the group (method of moments),
nu_g = within-player noise per unit of exposure (match-to-match variation of the per-match value, refit
monthly from the previous 365 days). For a Poisson count with no extra noise, nu = mu and this is exactly the
gamma-Poisson posterior mean of §4.4; for a share, nu = mu(1 - mu), the beta-binomial rule of §4.6; for a
continuous value it is the normal-normal rule of §4.5. If tau2 <= 0 the feature has no detectable
between-player signal in that group and every player gets the group mean (§4.5's diagnostic).

Position groups (prior groups until roles exist): CB, FB, CM, AM, W, FW from the Step 1 slots, taken as the
group of the player's decayed start count as of the date (365-day half-life). Players with no start yet
use the pooled outfield prior.

Possession adjustment (§4.2): defensive counts x clip(0.5/(1 - poss), 0.67, 1.5), on-ball volume counts
x clip(0.5/poss, 0.67, 1.5). Opponent adjustment (§4.9): each count is divided by the opponent's allowance
factor f_{B,g,s} for the player's group, itself shrunk and recency-weighted (365 days, carry 0.75), as of
that match.

Tuning (2024/25 only): per feature, half-life x carry x omega_sub x adjustment, by predicting each
player-match from the profile as of its date. Counts: Poisson deviance of the count given minutes,
possession and opponent; shares: binomial deviance given attempts; means: squared error weighted by minutes.

Outputs:
  data/processed/player_match_network.parquet   node features per player-match
  data/processed/player_profile_asof.parquet    profile per (match, player) for every outfield squad member
  reports/profile_grid.csv, reports/profile_settings.csv, reports/profile_test.csv,
  reports/profile_signal.csv, reports/profile_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
SEASONS = ["2023/2024", "2024/2025", "2025/2026", "2026/2027"]
TUNE, TEST = ["2024/2025"], ["2025/2026", "2026/2027"]

# name: (kind, numerator, denominator for shares, possession class, opponent-adjusted, label)
# possession class: "def" = defensive count, "on" = on-ball volume count, None = not adjusted (§4.2)
FEATURES = {
    # shooting
    "npxg": ("count", "npxg", None, None, True, "Non-penalty xG"),
    "np_shots": ("count", "np_shots", None, None, True, "Non-penalty shots"),
    "headed_shots": ("count", "headed_shots", None, None, True, "Headed shots"),
    "box_touches": ("count", "touches_opp_box", None, None, True, "Touches in opposition box"),
    # creation
    "xa": ("count", "xa", None, None, True, "Expected assists (xA)"),
    "chances_created": ("count", "chances_created", None, None, True, "Chances created"),
    "big_chances_created": ("count", "big_chances_created", None, None, True, "Big chances created"),
    "dribbles": ("count", "successful_dribbles", None, None, True, "Successful dribbles"),
    "dribble_attempts": ("count", "successful_dribbles_attempted", None, None, True, "Dribbles attempted"),
    # progression
    # line-breaking passes are left out: zero in every 2023/24 and 2024/25 row (not collected), and 2026/27's
    # league mean is 1.7x 2025/26's (definition change)
    "final_third_passes": ("count", "passes_into_final_third", None, "on", True, "Passes into final third"),
    # passing
    "passes": ("count", "passes_attempted", None, "on", True, "Passes attempted"),
    "touches": ("count", "touches", None, "on", True, "Touches"),
    "crosses": ("count", "crosses_attempted", None, "on", True, "Crosses attempted"),
    "long_balls": ("count", "accurate_long_balls_attempted", None, "on", True, "Long balls attempted"),
    # defending
    "tackles": ("count", "tackles", None, "def", True, "Tackles"),
    "interceptions": ("count", "interceptions", None, "def", True, "Interceptions"),
    "blocks": ("count", "blocks", None, "def", True, "Blocks"),
    "recoveries": ("count", "recoveries", None, "def", True, "Recoveries"),
    "clearances": ("count", "clearances", None, "def", True, "Clearances"),
    "dribbled_past": ("count", "dribbled_past", None, "def", True, "Dribbled past"),
    "fouls": ("count", "fouls_committed", None, "def", True, "Fouls committed"),
    "ground_duels": ("count", "ground_duels_attempted", None, None, True, "Ground duels"),
    "aerial_duels": ("count", "aerial_duels_attempted", None, None, True, "Aerial duels"),
    "was_fouled": ("count", "was_fouled", None, None, True, "Was fouled"),
    # shares (beta-binomial)
    "pass_acc": ("prop", "accurate_passes", "passes_attempted", None, False, "Pass accuracy"),
    "cross_acc": ("prop", "accurate_crosses", "crosses_attempted", None, False, "Cross accuracy"),
    "dribble_succ": ("prop", "successful_dribbles", "successful_dribbles_attempted", None, False, "Dribble success"),
    "ground_duel_win": ("prop", "ground_duels_won", "ground_duels_attempted", None, False, "Ground duel win rate"),
    "aerial_win": ("prop", "aerial_duels_won", "aerial_duels_attempted", None, False, "Aerial duel win rate"),
    "long_ball_acc": ("prop", "accurate_long_balls", "accurate_long_balls_attempted", None, False, "Long-ball accuracy"),
    # passing network (§4.10 node features)
    "passes_received": ("count", "passes_received", None, "on", True, "Passes received (network)"),
    "prog_out": ("count", "prog_out", None, "on", True, "Progression passed, m (network)"),
    "prog_in": ("count", "prog_in", None, "on", True, "Progression received, m (network)"),
    "x_rel": ("mean", "x_rel", None, None, False, "Depth vs team centroid, m"),
    "y_abs": ("mean", "y_abs", None, None, False, "Distance from centre line, m"),
    "y_side": ("mean", "y_side", None, None, False, "Side (+ left, - right), m"),
    "betweenness": ("mean", "betweenness", None, None, False, "Betweenness (network)"),
    "pass_share": ("mean", "pass_share", None, None, False, "Share of team passes involving him"),
}
FEAT = list(FEATURES)
KIND = np.array([FEATURES[f][0] for f in FEAT])

SLOT_GROUP = {"LCB": "CB", "RCB": "CB", "CB": "CB",
              "LB": "FB", "RB": "FB", "LWB": "FB", "RWB": "FB",
              "DM": "CM", "LDM": "CM", "RDM": "CM", "CM": "CM", "LCM": "CM", "RCM": "CM",
              "CAM": "AM", "SS": "AM",
              "LM": "W", "RM": "W", "LW": "W", "RW": "W", "LAM": "W", "RAM": "W",
              "ST": "FW", "LS": "FW", "RS": "FW"}
GROUPS = ["CB", "FB", "CM", "AM", "W", "FW"]
GROUP_HL = 365.0          # memory for the player's position group
OPP_HL, OPP_CARRY = 365.0, 0.75   # memory for opponent allowance factors
NU_WINDOW = 365           # days of history for the within-player noise estimate
MIN_E_PRIOR = 0.25        # players with less effective exposure don't enter prior moments

HALF_LIVES = [60.0, 120.0, 240.0, 365.0, 540.0, 730.0, np.inf]
CARRIES = [1.0, 0.75, 0.5]
OMEGAS = [0.5, 1.0]
THETAS = [1.0, 0.5, 0.0]   # possession elasticity: factor = clip(0.5/poss or 0.5/(1-poss), .67, 1.5) ** theta
ADJUST = [f"p{t:g}+opp" for t in THETAS] + [f"p{t:g}" for t in THETAS]   # order = preference within ties
TIE = 0.001               # deviance within 0.1% of the best is a tie -> longer memory, then simpler adjustment


# --------------------------------------------------------------------------- node features
def node_features(force: bool = False) -> pd.DataFrame:
    path = PROC / "player_match_network.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)
    nodes = pd.read_parquet(PROC / "network_nodes.parquet")
    edges = pd.read_parquet(PROC / "network_edges.parquet")
    tm = pd.read_parquet(PROC / "team_match_network.parquet")[["match_id", "team_id", "centroid_x"]]
    nodes = nodes[nodes.minutes_played.fillna(0) > 0].merge(tm, on=["match_id", "team_id"], how="left")
    rows = []
    eg = dict(tuple(edges[edges.pass_count > 0].groupby(["match_id", "team_id"])))
    for (mid, tid), g in nodes.groupby(["match_id", "team_id"]):
        pos = g.set_index("player_id")[["x_m", "y_m"]]
        ge = eg.get((mid, tid))
        recv = out_ = prog_o = prog_i = btw = pd.Series(dtype=float)
        tot = 0.0
        if ge is not None:
            ge = ge[ge.passer_id.isin(pos.index) & ge.receiver_id.isin(pos.index)]
            tot = float(ge.pass_count.sum())
            recv = ge.groupby("receiver_id").pass_count.sum()
            out_ = ge.groupby("passer_id").pass_count.sum()
            ok = pos.dropna().index
            gp = ge[ge.passer_id.isin(ok) & ge.receiver_id.isin(ok)]
            gain = np.maximum(0, pos.x_m.reindex(gp.receiver_id).to_numpy() - pos.x_m.reindex(gp.passer_id).to_numpy())
            prog = gp.assign(gain=gain * gp.pass_count.to_numpy())
            prog_o = prog.groupby("passer_id").gain.sum()
            prog_i = prog.groupby("receiver_id").gain.sum()
            G = nx.DiGraph()
            for a, b, c in zip(ge.passer_id, ge.receiver_id, ge.pass_count):
                if a != b:
                    G.add_edge(a, b, distance=1.0 / c)
            if G.number_of_nodes() > 2:
                btw = pd.Series(nx.betweenness_centrality(G, weight="distance", normalized=True))
        for r in g.itertuples():
            has_pos = not np.isnan(r.x_m)
            rows.append(dict(match_id=mid, team_id=tid, player_id=r.player_id,
                             passes_received=float(recv.get(r.player_id, 0.0)),
                             prog_out=float(prog_o.get(r.player_id, 0.0)) if has_pos else np.nan,
                             prog_in=float(prog_i.get(r.player_id, 0.0)) if has_pos else np.nan,
                             x_rel=r.x_m - r.centroid_x if has_pos else np.nan,
                             y_abs=abs(r.y_m - 34.0) if has_pos else np.nan,
                             y_side=r.y_m - 34.0 if has_pos else np.nan,
                             betweenness=float(btw.get(r.player_id, 0.0)) if has_pos else np.nan,
                             pass_share=(float(recv.get(r.player_id, 0.0) + out_.get(r.player_id, 0.0)) / tot)
                             if tot > 0 else np.nan))
    out = pd.DataFrame(rows)
    out.to_parquet(path, index=False)
    return out


# --------------------------------------------------------------------------- observations
class Obs:
    """Arrays for every outfield player-match where the player appeared, in date order."""

    def __init__(self):
        pm = pd.read_parquet(PROC / "player_match.parquet")
        pm = pm[~pm.is_gk].copy()
        pm["np_shots"] = pm.total_shots - pm.penalties_taken
        nf = node_features()
        pm = pm.merge(nf.drop(columns="team_id"), on=["match_id", "player_id"], how="left")
        slots = pd.read_parquet(PROC / "player_slots.parquet")[["match_id", "player_id", "slot"]]
        pm = pm.merge(slots, on=["match_id", "player_id"], how="left")
        pm["slot_group"] = pm.slot.map(SLOT_GROUP)
        m = pd.read_parquet(PROC / "matches.parquet")[["match_id", "home_team_id", "away_team_id"]]
        pm = pm.merge(m, on="match_id", how="left")
        pm["opp_id"] = np.where(pm.team_id == pm.home_team_id, pm.away_team_id, pm.home_team_id)
        pm["day"] = pm.date.dt.floor("D")
        pm["season_idx"] = pm.season.map({s: k for k, s in enumerate(SEASONS)})
        pm = pm.sort_values(["date", "match_id", "team_id", "player_id"]).reset_index(drop=True)
        self.squad = pm                                  # every outfield squad member (profile rows)
        self.played = pm.played.to_numpy()
        self.players = np.sort(pm.player_id.unique())
        self.pidx = np.searchsorted(self.players, pm.player_id.to_numpy())
        self.teams = np.sort(pd.unique(pd.concat([pm.team_id, pm.opp_id])))
        self.opp = np.searchsorted(self.teams, pm.opp_id.to_numpy())
        self.days = np.sort(pm.day.unique())
        self.dayidx = np.searchsorted(self.days, pm.day.to_numpy())
        self.season_of_day = pm.groupby("day").season_idx.first().reindex(self.days).to_numpy()
        self.sub = (self.played & ~pm.started.to_numpy())
        poss = pm.team_possession_pct.to_numpy(float) / 100.0
        poss = np.where(np.isnan(poss), 0.5, poss)
        self.poss = poss
        n, F = len(pm), len(FEAT)
        e90 = pm.minutes_played.fillna(0).to_numpy(float) / 90.0
        V = np.zeros((n, F))        # per-match "numerator" (count, successes, or value * e)
        X = np.zeros((n, F))        # exposure (90s, attempts, or 90s for means)
        PF = np.ones((n, F))        # possession factor applied to counts
        dfac = np.clip(0.5 / np.clip(1 - poss, 1e-3, None), 0.67, 1.5)
        ofac = np.clip(0.5 / np.clip(poss, 1e-3, None), 0.67, 1.5)
        for k, f in enumerate(FEAT):
            kind, num, den, pcls, _, _ = FEATURES[f]
            v = pm[num].to_numpy(float)
            if kind == "count":
                ok = self.played & ~np.isnan(v)
                V[:, k] = np.where(ok, v, 0.0)
                X[:, k] = np.where(ok, e90, 0.0)
                if pcls == "def":
                    PF[:, k] = dfac
                elif pcls == "on":
                    PF[:, k] = ofac
            elif kind == "prop":
                a = pm[den].to_numpy(float)
                ok = self.played & ~np.isnan(v) & ~np.isnan(a) & (a > 0)
                V[:, k] = np.where(ok, v, 0.0)
                X[:, k] = np.where(ok, a, 0.0)
            else:
                ok = self.played & ~np.isnan(v)
                V[:, k] = np.where(ok, v * e90, 0.0)
                X[:, k] = np.where(ok, e90, 0.0)
        self.V, self.X, self.PF, self.e90 = V, X, PF, e90
        self.raw = V.copy()
        g = pm.slot_group.map({gg: k for k, gg in enumerate(GROUPS)}).fillna(-1).to_numpy(int)
        self.start_group = g        # -1 for substitutes and unused players
        self.opp_adj = np.array([FEATURES[f][4] for f in FEAT])
        self.is_count = KIND == "count"
        self._groups_asof()

    def _groups_asof(self):
        """Position group of each row as of its date: argmax of decayed start counts per group."""
        P = len(self.players)
        S = np.zeros((P, len(GROUPS)))
        grp = np.full(len(self.squad), -1)
        last = None
        rows_by_day = np.split(np.arange(len(self.squad)), np.flatnonzero(np.diff(self.dayidx)) + 1)
        for rows in rows_by_day:
            d = self.days[self.dayidx[rows[0]]]
            if last is not None:
                S *= 0.5 ** ((d - last) / np.timedelta64(1, "D") / GROUP_HL)
            last = d
            p = self.pidx[rows]
            tot = S[p].sum(1)
            grp[rows] = np.where(tot > 0.5, S[p].argmax(1), -1)
            st = self.start_group[rows]
            ok = st >= 0
            np.add.at(S, (p[ok], st[ok]), 1.0)
        self.group = grp
        self._S_final = S

    def by_day(self):
        return np.split(np.arange(len(self.squad)), np.flatnonzero(np.diff(self.dayidx)) + 1)


# --------------------------------------------------------------------------- shrinkage machinery
def prior_moments(C, E, grp, nu, kinds):
    """Group mean mu, between-player variance tau2 and pseudo-exposure b per group (rows 0..G-1) and pooled (row G).

    C, E: (P, F) decayed sums; grp: (P,) group index or -1; nu: (G+1, F) within-player noise per unit exposure."""
    G = len(GROUPS)
    mu = np.full((G + 1, C.shape[1]), np.nan)
    tau2 = np.full_like(mu, np.nan)
    for gi in range(G + 1):
        sel = (grp == gi) if gi < G else np.ones(len(grp), bool)
        Eg, Cg = E[sel], C[sel]
        use = Eg > MIN_E_PRIOR
        Eu = np.where(use, Eg, 0.0)
        Cu = np.where(use, Cg, 0.0)
        n = use.sum(0)
        sE = Eu.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = Cu.sum(0) / sE
            r = np.where(use, Cu / np.where(Eu > 0, Eu, 1), 0.0)
            ss = (Eu * (r - m) ** 2).sum(0)
            denom = sE - (Eu ** 2).sum(0) / sE
            t2 = (ss - (n - 1) * nu[gi]) / denom
        mu[gi] = np.where(n >= 5, m, np.nan)
        tau2[gi] = np.where(n >= 5, t2, np.nan)
    # groups with too few players borrow the pooled moments
    for gi in range(G):
        bad = np.isnan(mu[gi])
        mu[gi, bad] = mu[G, bad]
        tau2[gi, bad] = tau2[G, bad]
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(tau2 > 0, nu / tau2, np.inf)
    return mu, tau2, b


def within_noise(obs: Obs, V, X, upto_day, mask_rows):
    """nu per group and pooled: sum_m x_m (v_m/x_m - r_i)^2 / (N - n_players), on rows before `upto_day`."""
    G, F = len(GROUPS), V.shape[1]
    nu = np.full((G + 1, F), np.nan)
    rows = np.flatnonzero(mask_rows)
    if len(rows) == 0:
        return nu
    p = obs.pidx[rows]
    g = obs.group[rows]
    Vr, Xr = V[rows], X[rows]
    P = len(obs.players)
    Cp = np.zeros((P, F)); Ep = np.zeros((P, F))
    np.add.at(Cp, p, Vr); np.add.at(Ep, p, Xr)
    with np.errstate(invalid="ignore", divide="ignore"):
        rp = Cp / Ep
        val = Vr / Xr
        dev = np.where(Xr > 0, Xr * (val - rp[p]) ** 2, 0.0)
    nobs = (Xr > 0)
    for gi in range(G + 1):
        sel = (g == gi) if gi < G else np.ones(len(rows), bool)
        if sel.sum() < 50:
            continue
        npl = np.array([len(np.unique(p[sel & nobs[:, k]])) for k in range(F)])
        N = nobs[sel].sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            nu[gi] = dev[sel].sum(0) / np.maximum(N - npl, 1)
    for gi in range(G):
        bad = np.isnan(nu[gi])
        nu[gi, bad] = nu[G, bad]
    return nu


def opponent_factors(obs: Obs, Vp: np.ndarray) -> np.ndarray:
    """Allowance factor f_{B,g,s} as of each row's date (before its match), for the row's own group (§4.9).

    Vp: possession-adjusted numerators. Returns (n, F) array of factors (1 where not applicable)."""
    T, G, F = len(obs.teams), len(GROUPS), len(FEAT)
    n = len(obs.squad)
    fac = np.ones((n, F))
    O = np.zeros((T, G, F)); Xp = np.zeros((T, G, F))     # observed vs expected, against each team
    Cl = np.zeros((G, F)); El = np.zeros((G, F))          # league group means
    last = None
    cnt = obs.is_count & obs.opp_adj
    for rows in obs.by_day():
        di = obs.dayidx[rows[0]]
        d = obs.days[di]
        if last is not None:
            dec = 0.5 ** ((d - last) / np.timedelta64(1, "D") / OPP_HL)
            dec *= OPP_CARRY ** (obs.season_of_day[di] - obs.season_of_day[last_i])
            O *= dec; Xp *= dec; Cl *= dec; El *= dec
        last, last_i = d, di
        g = obs.group[rows]
        with np.errstate(invalid="ignore", divide="ignore"):
            lam = Cl / El                                   # (G, F) league group rate before today
            # shrinkage of f per (g, s): between-team variance by moments; noise phi from Poisson-like dispersion
            r = O / Xp
            use = Xp > 1.0
            w = np.where(use, Xp, 0.0)
            sw = w.sum(0)
            fbar = np.where(use, O, 0).sum(0) / sw
            ss = (w * (np.where(use, r, 0) - fbar) ** 2).sum(0)
            nt = use.sum(0)
            t2 = (ss - (nt - 1) * PHI[None, :] * fbar) / (sw - (w ** 2).sum(0) / sw)
            bf = np.where((t2 > 0) & (nt >= 8), PHI[None, :] / t2, np.inf)
            fhat = np.where(np.isinf(bf), 1.0, (O + bf) / (Xp + bf))
        ok = g >= 0
        fr = np.ones((len(rows), F))
        fr[ok] = fhat[obs.opp[rows][ok], g[ok]]
        fr = np.where(cnt[None, :] & np.isfinite(fr) & (fr > 0), fr, 1.0)
        fac[rows] = fr
        # add today's observations
        pl = rows[obs.played[rows] & ok]
        gp = obs.group[pl]
        with np.errstate(invalid="ignore"):
            expv = obs.X[pl] * np.nan_to_num(lam[gp])
        np.add.at(O, (obs.opp[pl], gp), np.where(cnt, Vp[pl], 0.0))
        np.add.at(Xp, (obs.opp[pl], gp), np.where(cnt, expv, 0.0))
        np.add.at(Cl, gp, Vp[pl]); np.add.at(El, gp, obs.X[pl])
    return fac


PHI = None   # per-feature dispersion of counts relative to Poisson (set in prepare())


def prepare(obs: Obs) -> dict:
    """For each adjustment variant: adjusted numerators V and the context multiplier that turns a profile
    back into an expected raw count for a given match (raw ~ profile x exposure x ctx)."""
    global PHI
    warm = obs.played & (obs.squad.season_idx.to_numpy() == 0)
    # dispersion phi_s = nu/mu on 2023/24 (warm-up season); only scales the opponent-factor shrinkage
    nu = within_noise(obs, obs.V, obs.X, None, warm)[len(GROUPS)]
    mu = obs.V[warm].sum(0) / obs.X[warm].sum(0)
    PHI = np.where(obs.is_count & (mu > 0), nu / np.where(mu > 0, mu, 1), 1.0)
    out = {}
    cnt = obs.is_count[None, :]
    for t in THETAS:
        pf = np.where(cnt, obs.PF ** t, 1.0)
        Vp = obs.V * pf
        fac = opponent_factors(obs, Vp)
        out[f"p{t:g}"] = (Vp, 1.0 / pf)
        out[f"p{t:g}+opp"] = (np.where(cnt, Vp / fac, Vp), fac / pf)
    return out


def run(obs: Obs, prep: dict, settings: list[tuple], target_mask: np.ndarray, keep: bool = False):
    """Rolling profiles for each setting (hl, carry, omega, adjust).

    Returns, per setting, the predicted profile for every target row (n_target, F), the group-prior b and
    exposure E used, and the per-row predicted numerator for scoring."""
    P, F = len(obs.players), len(FEAT)
    target = np.flatnonzero(target_mask)
    tpos = np.full(len(obs.squad), -1)
    tpos[target] = np.arange(len(target))
    out = []
    month_key = np.array([str(d)[:7] for d in obs.days])
    for hl, carry, om, adj in settings:
        Vadj = prep[adj][0]
        C = np.zeros((P, F)); E = np.zeros((P, F))
        prof = np.full((len(target), F), np.nan)
        Eout = np.full((len(target), F), np.nan)
        bout = np.full((len(target), F), np.nan)
        mout = np.full((len(target), F), np.nan)
        rawout = np.full((len(target), F), np.nan)
        last = None
        nu = None
        nu_month = None
        wsub = np.where(obs.sub, om, 1.0)
        for rows in obs.by_day():
            di = obs.dayidx[rows[0]]
            d = obs.days[di]
            if last is not None:
                age = (d - last) / np.timedelta64(1, "D")
                dec = 1.0 if np.isinf(hl) else 0.5 ** (age / hl)
                dec *= carry ** (obs.season_of_day[di] - obs.season_of_day[last_i])
                C *= dec; E *= dec
            last, last_i = d, di
            tr = rows[tpos[rows] >= 0]
            if len(tr):
                if nu_month != month_key[di]:
                    ck = (adj, month_key[di])
                    if ck not in _NU_CACHE:
                        lo = d - np.timedelta64(NU_WINDOW, "D")
                        dd = obs.squad.day.to_numpy()
                        mask = obs.played & (dd < d) & (dd >= lo)
                        _NU_CACHE[ck] = within_noise(obs, Vadj, obs.X, d, mask)
                    nu = _NU_CACHE[ck]
                    nu_month = month_key[di]
                pgrp = player_groups_now(obs, tr, d)
                mu, tau2, b = prior_moments(C, E, pgrp, np.nan_to_num(nu, nan=np.inf), KIND)
                p = obs.pidx[tr]
                g = obs.group[tr]
                gi = np.where(g >= 0, g, len(GROUPS))
                m_, b_ = mu[gi], b[gi]
                with np.errstate(invalid="ignore", divide="ignore"):
                    shr = np.where(np.isinf(b_), m_, (C[p] + b_ * m_) / (E[p] + b_))
                shr = np.where(np.isnan(shr), m_, shr)
                prof[tpos[tr]] = shr
                Eout[tpos[tr]] = E[p]
                bout[tpos[tr]] = b_
                mout[tpos[tr]] = m_
                with np.errstate(invalid="ignore", divide="ignore"):
                    rawout[tpos[tr]] = np.where(E[p] > 0, C[p] / E[p], m_)
            pl = rows[obs.played[rows]]
            w = wsub[pl][:, None]
            np.add.at(C, obs.pidx[pl], w * Vadj[pl])
            np.add.at(E, obs.pidx[pl], w * obs.X[pl])
        out.append(dict(profile=prof, E=Eout, b=bout, prior=mout, raw=rawout))
    return target, out


_GROUP_CACHE = {}
_NU_CACHE = {}


def player_groups_now(obs: Obs, rows, d):
    """Group of every player as of day d (for prior moments): the group carried by his latest squad row
    before or on d."""
    key = d
    if key in _GROUP_CACHE:
        return _GROUP_CACHE[key]
    sq = obs.squad
    before = obs.dayidx <= obs.dayidx[rows[0]]
    last = pd.Series(obs.group[before], index=obs.pidx[before]).groupby(level=0).last()
    g = np.full(len(obs.players), -1)
    g[last.index.to_numpy()] = last.to_numpy()
    _GROUP_CACHE[key] = g
    return g


# --------------------------------------------------------------------------- scoring
def score_rows(obs: Obs, prep: dict, adj: str, prof: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Per-row loss (n_rows, F): counts -> Poisson deviance of the raw count; shares -> binomial deviance;
    means -> squared error x minutes. Rows where the player did not appear or the feature is absent -> nan."""
    y = obs.raw[rows]
    x = obs.X[rows]
    loss = np.full(y.shape, np.nan)
    ctx = prep[adj][1][rows]
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.clip(prof * x * ctx, 1e-9, None)
        pois = 2 * (np.where(y > 0, y * np.log(y / mu), 0.0) - (y - mu))
        p = np.clip(prof, 1e-6, 1 - 1e-6)
        k, n = y, x
        binl = -2 * (k * np.log(p) + (n - k) * np.log(1 - p))
        sat = -2 * (np.where(k > 0, k * np.log(np.clip(k / np.where(n > 0, n, 1), 1e-12, None)), 0)
                    + np.where(n - k > 0, (n - k) * np.log(np.clip((n - k) / np.where(n > 0, n, 1), 1e-12, None)), 0))
        sq = x * (y / np.where(x > 0, x, 1) - prof) ** 2
    for j in range(y.shape[1]):
        if KIND[j] == "count":
            loss[:, j] = pois[:, j]
        elif KIND[j] == "prop":
            loss[:, j] = binl[:, j] - sat[:, j]
        else:
            loss[:, j] = sq[:, j]
    loss[x <= 0] = np.nan
    return loss


def tuning_grid(obs: Obs, prep: dict) -> pd.DataFrame:
    tmask = obs.played & obs.squad.season.isin(TUNE).to_numpy()
    rows_all = np.flatnonzero(tmask)
    recs = []
    settings = [(h, c, o, a) for h in HALF_LIVES for c in CARRIES for o in OMEGAS for a in ADJUST]
    for s in settings:
        _, res = run(obs, prep, [s], tmask)
        loss = score_rows(obs, prep, s[3], res[0]["profile"], rows_all)
        tot = np.nansum(loss, 0)
        for j, f in enumerate(FEAT):
            recs.append(dict(feature=f, half_life=s[0], carry=s[1], omega_sub=s[2], adjust=s[3], loss=tot[j]))
    return pd.DataFrame(recs)


def choose(grid: pd.DataFrame) -> pd.DataFrame:
    """Per feature: first the adjustment, then the memory.

    Adjustment: the methodology's full adjustment (possession + opponent) is kept unless a simpler variant
    beats it by more than TIE on the tuning season. Memory: lowest loss; settings within TIE of the best are
    tied and the longest half-life, then the highest carry, then the full substitute weight wins."""
    out = []
    order = ADJUST
    for f, g in grid.groupby("feature", sort=False):
        kind, *_ , opp_ok, _ = FEATURES[f]
        if kind != "count":
            g = g[g.adjust == "p0"]
        best_by = g.groupby("adjust").loss.min()
        adj = next(a for a in order if a in best_by.index)
        for a in order:
            if a in best_by.index and best_by[a] < best_by[adj] * (1 - TIE):
                adj = a
        h = g[g.adjust == adj]
        best = h.loss.min()
        tied = h[h.loss <= best * (1 + TIE)].copy()
        tied["hl_key"] = tied.half_life.replace(np.inf, 1e9)
        r = tied.sort_values(["hl_key", "carry", "omega_sub"], ascending=False).iloc[0]
        out.append(dict(feature=f, kind=kind, half_life=r.half_life, carry=r.carry, omega_sub=r.omega_sub,
                        adjust=adj, tuning_loss=r.loss,
                        **{f"best_{a}": best_by.get(a, np.nan) for a in order}, possession_theta=float(adj.split("+")[0][1:]) if FEATURES[f][3] else np.nan,
                        opponent_adjusted=adj.endswith("+opp"), n_tied=len(tied)))
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- build and evaluate
def build(obs: Obs, prep: dict, chosen: pd.DataFrame, target_mask: np.ndarray) -> dict:
    """Profiles for target rows, each feature at its chosen setting. Also returns the group prior mean,
    the unshrunk rate, exposure E and pseudo-exposure b, for evaluation and reporting."""
    target = np.flatnonzero(target_mask)
    F = len(FEAT)
    res = {k: np.full((len(target), F), np.nan) for k in ("profile", "prior", "raw", "E", "b")}
    keys = chosen.set_index("feature")[["half_life", "carry", "omega_sub", "adjust"]]
    for s, feats in keys.groupby(list(keys.columns)).groups.items():
        _, out = run(obs, prep, [tuple(s)], target_mask)
        cols = [FEAT.index(f) for f in feats]
        for k in res:
            res[k][:, cols] = out[0][k][:, cols]
    res["target"] = target
    return res


def evaluate(obs, prep, chosen, res, mask_eval) -> pd.DataFrame:
    """Loss on the evaluation rows for: chosen profile, group mean only, unshrunk rate, and the chosen memory
    without any adjustment."""
    rows_pos = np.flatnonzero(mask_eval[res["target"]])
    rows = res["target"][rows_pos]
    adj_of = chosen.set_index("feature").adjust
    out = []
    no_adj = chosen.assign(adjust="p0")
    res0 = build(obs, prep, no_adj, mask_eval)
    for j, f in enumerate(FEAT):
        a = adj_of[f]
        L = {}
        for name, prof, aa in [("profile", res["profile"][rows_pos], a), ("group_mean", res["prior"][rows_pos], a),
                               ("unshrunk", res["raw"][rows_pos], a), ("no_adjustment", res0["profile"], "p0")]:
            L[name] = score_rows(obs, prep, aa, prof, rows)[:, j]
        base = L["profile"]
        rec = dict(feature=f, kind=FEATURES[f][0], n=int(np.isfinite(base).sum()), loss=np.nansum(base))
        for name in ("group_mean", "unshrunk", "no_adjustment"):
            d = L[name] - base
            ok = np.isfinite(d)
            rec[f"gain_vs_{name}_pct"] = 100 * np.nansum(d) / np.nansum(L[name])
            # bootstrap over matches
            mids = obs.squad.match_id.to_numpy()[rows][ok]
            agg = pd.Series(d[ok]).groupby(mids).sum().to_numpy()
            tot = pd.Series(L[name][ok]).groupby(mids).sum().sum()
            rng = np.random.default_rng(0)
            bs = np.array([agg[rng.integers(0, len(agg), len(agg))].sum() for _ in range(1000)]) / tot * 100
            rec[f"gain_vs_{name}_lo"], rec[f"gain_vs_{name}_hi"] = np.percentile(bs, [2.5, 97.5])
        out.append(rec)
    return pd.DataFrame(out)


def signal_table(obs: Obs, res: dict) -> pd.DataFrame:
    """Share of player-match profiles (test seasons) where a feature had no between-player signal in the
    player's group (tau2 <= 0 -> b infinite), and the median pseudo-exposure b in 90s (attempts for shares)."""
    sq = obs.squad.iloc[res["target"]]
    test = sq.season.isin(TEST).to_numpy()
    g = obs.group[res["target"]][test]
    recs = []
    for j, f in enumerate(FEAT):
        b = res["b"][test, j]
        for gi, gn in enumerate(GROUPS + ["none"]):
            sel = (g == gi) if gi < len(GROUPS) else (g < 0)
            if sel.sum() == 0:
                continue
            bb = b[sel]
            recs.append(dict(feature=f, group=gn, no_signal_share=float(np.isinf(bb).mean()),
                             median_b=float(np.median(bb[np.isfinite(bb)])) if np.isfinite(bb).any() else np.inf))
    return pd.DataFrame(recs)


def main(run_grid: bool):
    REPORTS.mkdir(exist_ok=True)
    obs = Obs()
    prep = prepare(obs)
    gpath = REPORTS / "profile_grid.csv"
    if run_grid or not gpath.exists():
        tuning_grid(obs, prep).to_csv(gpath, index=False)
    grid = pd.read_csv(gpath)
    chosen = choose(grid)
    chosen.to_csv(REPORTS / "profile_settings.csv", index=False)

    allrows = np.ones(len(obs.squad), bool)
    res = build(obs, prep, chosen, allrows)
    sq = obs.squad
    keep = ["match_id", "date", "season", "team_id", "opp_id", "player_id", "player_name", "played", "started",
            "minutes_played"]
    prof = sq[keep].copy().rename(columns={"opp_id": "opponent_id"})
    prof["group"] = [GROUPS[g] if g >= 0 else None for g in obs.group]
    for j, f in enumerate(FEAT):
        prof[f] = res["profile"][:, j]
    # effective 90s of evidence (decayed minutes) behind the profile, at the touches setting
    prof["eff_90s"] = res["E"][:, FEAT.index("touches")]
    prof.to_parquet(PROC / "player_profile_asof.parquet", index=False)

    test_mask = obs.played & sq.season.isin(TEST).to_numpy()
    tune_mask = obs.played & sq.season.isin(TUNE).to_numpy()
    ev = evaluate(obs, prep, chosen, res, test_mask)
    ev.to_csv(REPORTS / "profile_test.csv", index=False)
    ev_t = evaluate(obs, prep, chosen, res, tune_mask)
    ev_t.to_csv(REPORTS / "profile_tuning_eval.csv", index=False)
    sig = signal_table(obs, res)
    sig.to_csv(REPORTS / "profile_signal.csv", index=False)

    def frac(col, lo):
        return int(((ev[col] > 0) & (ev[lo] > 0)).sum())
    summary = dict(
        n_profile_rows=int(len(prof)), n_players=int(prof.player_id.nunique()),
        n_features=len(FEAT),
        test_rows=int(test_mask.sum()),
        features_better_than_group_mean_test=frac("gain_vs_group_mean_pct", "gain_vs_group_mean_lo"),
        features_better_than_unshrunk_test=frac("gain_vs_unshrunk_pct", "gain_vs_unshrunk_lo"),
        features_worse_than_unshrunk_test=int((ev.gain_vs_unshrunk_hi < 0).sum()),
        adjusted_features=int((chosen.adjust != "p0").sum()),
        adjustment_helps_test=frac("gain_vs_no_adjustment_pct", "gain_vs_no_adjustment_lo"),
        adjustment_hurts_test=int((ev.gain_vs_no_adjustment_hi < 0).sum()),
        median_gain_vs_group_mean_pct=float(ev.gain_vs_group_mean_pct.median()),
        median_gain_vs_unshrunk_pct=float(ev.gain_vs_unshrunk_pct.median()),
        omega_sub_chosen=sorted(chosen.omega_sub.unique().tolist()),
        rows_without_group=int(prof.group.isna().sum()),
    )
    (REPORTS / "profile_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(ev.round(2).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true")
    main(ap.parse_args().grid)
