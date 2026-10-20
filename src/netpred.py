"""Step 4: predict the passing network of a starting XI before the match (methodology §5.6).

Two sub-models, both fitted as of date (§2.3):

1. Positions. A starter i in slot r of formation F (the teamsheet's template, known from team news):
       (x_hat, y_hat) = S[F, r] + team_shift[A, F, r] + delta_i
   S = league average position of the slot over earlier matches; team_shift = team A's average offset
   from S in that slot, shrunk toward 0 with k_team pseudo-matches; delta_i = the player's average offset
   from (S + team_shift) over his earlier starts (any team, any slot), recency-weighted (half-life h) and
   shrunk toward 0 with k_player pseudo-starts (§4.5 normal-normal rule with a fixed k).
   Before the match the slot is not observed, so starters are assigned to the template's slots by the
   Hungarian algorithm on their expected positions (each player's own recency-weighted average position;
   a player with no start yet sits at his position group's average, or mid-pitch).
   Goalkeepers: their own average position, shrunk to the league keeper average.

2. Pass links. Poisson gravity model on ordered pairs of starters (§5.6):
       log E[w_ij] = log(T_ij / 90) + mu_A + a_i + b_j + beta' f_ij
   T_ij = minutes i and j were both on the pitch; f_ij = distance, log distance, forward gain, lateral gap,
   same slot line, keeper passer, keeper receiver (all from the predicted positions), home, rating gap.
   mu_A, a_i, b_j are ridge-penalised effects. Refit on the first match day of every month from earlier
   matches only; propensities of players with no earlier start are 0.

Applied to a starting XI, the two give a predicted network for any lineup, including one that never played
together; §5.2-5.4 metrics on the predicted network give its predicted shape and structure.

Run:  python src/netpred.py          (positions tuned on 2024/25, then the edge model and validation)
Outputs: data/processed/pred_positions.parquet   every starter of every match: predicted slot, x, y
         data/processed/pred_network_metrics.parquet   predicted shape/structure metrics per team-match
         reports/netpred_position_grid.csv, reports/netpred_validation.csv, reports/netpred_summary.json
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment, minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import network as N  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
TUNE, TEST = ["2024/2025"], ["2025/2026", "2026/2027"]
PITCH_Y = 68.0

K_TEAM = [3.0, 10.0, 30.0]
K_PLAYER = [0.25, 0.5, 1.0, 2.0, 4.0]       # pseudo-starts pulling a player's own average toward his slot's position
HL_PLAYER = [120.0, 365.0, np.inf]
GK_K = 5.0
EDGE_RIDGE = dict(team=10.0, player=10.0, beta=1.0)
EDGE_FEATS = ["dist", "log_dist", "dx", "abs_dy", "same_line", "gk_passer", "gk_receiver", "home", "rating_gap"]


def naive(s):
    return s.dt.tz_convert(None) if getattr(s.dt, "tz", None) is not None else s


def days(s):
    return ((s - pd.Timestamp("2000-01-01")).dt.total_seconds() / 86400).to_numpy()


# --------------------------------------------------------------------------- inputs
def load():
    m = pd.read_parquet(PROC / "matches.parquet")
    m["date"] = naive(m.date)
    slots = pd.read_parquet(PROC / "player_slots.parquet")
    nodes = pd.read_parquet(PROC / "network_nodes.parquet")
    pm = pd.read_parquet(PROC / "player_match.parquet")
    prof = pd.read_parquet(PROC / "player_profile_asof.parquet", columns=["match_id", "player_id", "group"])
    sheet = pd.concat([m[["match_id", "home_team_id", "home_formation"]].set_axis(["match_id", "team_id", "sheet"], axis=1),
                       m[["match_id", "away_team_id", "away_formation"]].set_axis(["match_id", "team_id", "sheet"], axis=1)])
    sheet["formation"] = sheet.sheet.map(N.sheet_template)
    gk = pm[pm.played & pm.started & pm.is_gk][["match_id", "team_id", "player_id"]]
    gk = gk.merge(nodes[["match_id", "player_id", "x_m", "y_m"]], on=["match_id", "player_id"], how="left")
    st = pm[pm.played & pm.started][["match_id", "team_id", "player_id", "is_gk"]].merge(
        m[["match_id", "date", "season", "home_team_id"]], on="match_id")
    st["home"] = (st.team_id == st.home_team_id).astype(int)
    st = st.merge(slots[["match_id", "player_id", "slot", "slot_line", "x_m", "y_m"]], on=["match_id", "player_id"], how="left")
    gpos = gk.set_index(["match_id", "player_id"])
    k = pd.MultiIndex.from_frame(st[["match_id", "player_id"]])
    st.loc[st.is_gk.to_numpy(), "x_m"] = gpos.x_m.reindex(k[st.is_gk.to_numpy()]).to_numpy()
    st.loc[st.is_gk.to_numpy(), "y_m"] = gpos.y_m.reindex(k[st.is_gk.to_numpy()]).to_numpy()
    st = st.merge(sheet[["match_id", "team_id", "formation"]], on=["match_id", "team_id"])
    st = st.merge(prof.drop_duplicates(["match_id", "player_id"]), on=["match_id", "player_id"], how="left")
    st = st.sort_values(["date", "match_id", "team_id", "is_gk", "player_id"]).reset_index(drop=True)
    return m, st


def template_xy():
    out = {}
    for name, lines in N.TEMPLATES.items():
        out[name] = {s: (x, y) for k, line in enumerate(lines) for s, x, y in line}
    return out


# --------------------------------------------------------------------------- positions
class PositionModel:
    """Accumulates slot averages, team shifts and player offsets in date order; queried before each match day."""

    def __init__(self, k_team, k_player, hl_player):
        self.k_team, self.k_player, self.hl = k_team, k_player, hl_player
        self.slot = {}                 # (F, r) -> [sx, sy, n]
        self.team = {}                 # (A, F, r) -> [sx, sy, n]
        self.player = {}               # i -> list of (day, rx, ry)
        self.own = {}                  # i -> list of (day, x, y) actual positions (for slot assignment)
        self.gk = {}                   # i -> [sx, sy, n]
        self.gk_league = [0.0, 0.0, 0]
        self.tmpl = template_xy()

    def S(self, F, r):
        v = self.slot.get((F, r))
        if v is None or v[2] < 5:
            return np.array(self.tmpl[F][r], float)
        return np.array([v[0] / v[2], v[1] / v[2]])

    def shift(self, A, F, r):
        v = self.team.get((A, F, r))
        if v is None:
            return np.zeros(2)
        return np.array([v[0], v[1]]) / (v[2] + self.k_team)

    def delta(self, i, d):
        h = self.player.get(i)
        if not h:
            return np.zeros(2)
        a = np.array(h)
        w = np.ones(len(a)) if np.isinf(self.hl) else 0.5 ** ((d - a[:, 0]) / self.hl)
        return (w[:, None] * a[:, 1:]).sum(0) / (w.sum() + self.k_player)

    def usual(self, i, d, target):
        """Position of a player in a slot he usually plays: his own recency-weighted average position (half-life h),
        pulled toward the slot's expected position for this team with k_player pseudo-starts. (For a slot he has not
        played, Step 5 uses target + delta(i) instead.)"""
        h = self.own.get(i)
        if not h:
            return target
        a = np.array(h)
        w = np.ones(len(a)) if np.isinf(self.hl) else 0.5 ** ((d - a[:, 0]) / self.hl)
        return ((w[:, None] * a[:, 1:]).sum(0) + self.k_player * target) / (w.sum() + self.k_player)

    def expected(self, i, d, fallback):
        """Player's own recency-weighted average position (365 days), for slot assignment."""
        h = self.own.get(i)
        if not h:
            return fallback
        a = np.array(h)
        w = 0.5 ** ((d - a[:, 0]) / 365.0)
        return (w[:, None] * a[:, 1:]).sum(0) / w.sum()

    def gk_pos(self, i):
        L = self.gk_league
        mu = np.array([L[0] / L[2], L[1] / L[2]]) if L[2] else np.array([10.0, 34.0])
        v = self.gk.get(i)
        if v is None:
            return mu
        return (np.array([v[0], v[1]]) + GK_K * mu) / (v[2] + GK_K)

    def assign(self, F, A, exp_xy):
        """Hungarian assignment of 10 outfield starters to the formation's slots, in metres, against the slots'
        expected positions for this team (league slot average + team shift). Standardizing both sides first (as
        network.py does for observed positions) agreed less often with the observed slot on 2024/25 (60% vs 62%)."""
        slots = list(self.tmpl[F])
        C = np.array([self.S(F, s) + self.shift(A, F, s) for s in slots])
        cost = ((np.asarray(exp_xy, float)[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        r, c = linear_sum_assignment(cost)
        out = [None] * len(exp_xy)
        for a, b in zip(r, c):
            out[a] = slots[b]
        return out

    def add(self, g, d):
        """Add one team-match's observed starters (after it is played)."""
        for row in g.itertuples():
            if row.is_gk:
                if not np.isnan(row.x_m):
                    v = self.gk.setdefault(row.player_id, [0.0, 0.0, 0])
                    v[0] += row.x_m; v[1] += row.y_m; v[2] += 1
                    self.gk_league[0] += row.x_m; self.gk_league[1] += row.y_m; self.gk_league[2] += 1
                continue
            F, r = row.formation, row.slot
            S = self.S(F, r)
            sh = self.shift(row.team_id, F, r)
            v = self.slot.setdefault((F, r), [0.0, 0.0, 0])
            v[0] += row.x_m; v[1] += row.y_m; v[2] += 1
            t = self.team.setdefault((row.team_id, F, r), [0.0, 0.0, 0])
            t[0] += row.x_m - S[0]; t[1] += row.y_m - S[1]; t[2] += 1
            self.player.setdefault(row.player_id, []).append((d, row.x_m - S[0] - sh[0], row.y_m - S[1] - sh[1]))
            self.own.setdefault(row.player_id, []).append((d, row.x_m, row.y_m))


def group_positions(st, before):
    """Average outfield position of each position group over earlier matches (fallback for new players)."""
    s = st[(st.date < before) & ~st.is_gk & st.group.notna()]
    return s.groupby("group")[["x_m", "y_m"]].mean()


def run_positions(st, k_team, k_player, hl, own_out=None):
    """Predicted position of every starter as of his match date (own_out, if given, receives each outfield
    starter's own recency-weighted average position: the baseline for players in their usual slot)."""
    pmod = PositionModel(k_team, k_player, hl)
    out = np.full((len(st), 2), np.nan)
    pslot = np.full(len(st), None, dtype=object)
    d_all = days(st.date)
    gp_cache = {}
    for day, idx in st.groupby(st.date.dt.floor("D")).groups.items():
        idx = np.asarray(idx)
        d = d_all[idx[0]]
        season = st.season.iloc[idx[0]]
        if season not in gp_cache:
            gp_cache[season] = group_positions(st, st.loc[st.season == season, "date"].min())
        gp = gp_cache[season]
        day_rows = st.loc[idx]
        for (mid, tid), g in day_rows.groupby(["match_id", "team_id"]):
            out_rows = g.index[~g.is_gk.to_numpy()]
            gk_rows = g.index[g.is_gk.to_numpy()]
            for r_ in gk_rows:
                out[r_] = pmod.gk_pos(st.at[r_, "player_id"])
            F = g.formation.iloc[0]
            exp = []
            for r_ in out_rows:
                grp = st.at[r_, "group"]
                fb = gp.loc[grp].to_numpy(float) if isinstance(grp, str) and grp in gp.index else np.array([50.0, 34.0])
                exp.append(pmod.expected(st.at[r_, "player_id"], d, fb))
            sl = pmod.assign(F, tid, exp)
            for k_, (r_, s) in enumerate(zip(out_rows, sl)):
                if own_out is not None:
                    own_out[r_] = exp[k_]
                pslot[r_] = s
                out[r_] = pmod.usual(st.at[r_, "player_id"], d, pmod.S(F, s) + pmod.shift(tid, F, s))
        for (mid, tid), g in day_rows.groupby(["match_id", "team_id"]):
            pmod.add(g, d)
    return out, pslot


def position_grid(st):
    rows = []
    mask = st.season.isin(TUNE).to_numpy() & st.x_m.notna().to_numpy() & ~st.is_gk.to_numpy()
    for kt, kp, hl in itertools.product(K_TEAM, K_PLAYER, HL_PLAYER):
        P, _ = run_positions(st, kt, kp, hl)
        err = ((P[mask] - st.loc[mask, ["x_m", "y_m"]].to_numpy()) ** 2).sum(1)
        rows.append(dict(k_team=kt, k_player=kp, hl_player=hl, tune_mse=float(err.mean())))
        print(rows[-1], flush=True)
    g = pd.DataFrame(rows).sort_values("tune_mse")
    g.to_csv(REPORTS / "netpred_position_grid.csv", index=False)
    return g


# --------------------------------------------------------------------------- edges
def pair_table(st, P, m):
    """Ordered pairs of starters per team-match with shared minutes, observed passes and features."""
    pt = pd.read_parquet(PROC / "player_times.parquet")
    edges = pd.read_parquet(PROC / "network_edges.parquet")
    ratings = pd.read_parquet(PROC / "team_ratings_asof.parquet")
    ratings["day"] = naive(ratings.day) if hasattr(ratings.day, "dt") else ratings.day
    s = st.assign(px=P[:, 0], py=P[:, 1])
    s = s.merge(pt[["match_id", "player_id", "t_on", "t_off"]], on=["match_id", "player_id"], how="left")
    s["line"] = s.slot_line_pred
    a = s[["match_id", "team_id", "player_id", "is_gk", "px", "py", "t_on", "t_off", "line", "date", "season", "home"]]
    pr = a.merge(a, on=["match_id", "team_id"], suffixes=("_i", "_j"))
    pr = pr[pr.player_id_i != pr.player_id_j]
    pr["T"] = (np.minimum(pr.t_off_i, pr.t_off_j) - np.maximum(pr.t_on_i, pr.t_on_j)).clip(lower=0)
    pr = pr[pr["T"] > 0.5].copy()
    w = edges.groupby(["match_id", "passer_id", "receiver_id"]).pass_count.sum()
    pr["w"] = w.reindex(pd.MultiIndex.from_arrays([pr.match_id, pr.player_id_i, pr.player_id_j])).fillna(0).to_numpy()
    dx = pr.px_j - pr.px_i
    dy = pr.py_j - pr.py_i
    pr["dist"] = np.hypot(dx, dy) / 10
    pr["log_dist"] = np.log1p(np.hypot(dx, dy))
    pr["dx"] = dx / 10
    pr["abs_dy"] = np.abs(dy) / 10
    pr["same_line"] = ((pr.line_i == pr.line_j) & pr.line_i.notna()).astype(float)
    pr["gk_passer"] = pr.is_gk_i.astype(float)
    pr["gk_receiver"] = pr.is_gk_j.astype(float)
    pr["home"] = pr.home_i.astype(float)
    rr = ratings.set_index(["day", "team_id"]).rating
    m2 = m[["match_id", "home_team_id", "away_team_id", "date"]].copy()
    m2["day"] = m2.date.dt.floor("D")
    m2["rh"] = rr.reindex(pd.MultiIndex.from_arrays([m2.day, m2.home_team_id])).to_numpy()
    m2["ra"] = rr.reindex(pd.MultiIndex.from_arrays([m2.day, m2.away_team_id])).to_numpy()
    pr = pr.merge(m2[["match_id", "home_team_id", "rh", "ra"]], on="match_id")
    pr["rating_gap"] = np.where(pr.team_id == pr.home_team_id, pr.rh - pr.ra, pr.ra - pr.rh)
    pr["rating_gap"] = pr.rating_gap.fillna(0)
    pr = pr.rename(columns={"date_i": "date", "season_i": "season"})
    return pr.sort_values(["date", "match_id", "team_id"]).reset_index(drop=True)


class EdgeModel:
    def __init__(self, pr):
        self.pr = pr
        self.teams = np.sort(pr.team_id.unique())
        self.players = np.sort(np.unique(np.r_[pr.player_id_i, pr.player_id_j]))
        tix = pd.Series(np.arange(len(self.teams)), index=self.teams)
        pix = pd.Series(np.arange(len(self.players)), index=self.players)
        n, T, P = len(pr), len(self.teams), len(self.players)
        r = np.arange(n)
        Xd = np.column_stack([np.ones(n)] + [pr[f].to_numpy(float) for f in EDGE_FEATS])
        self.X = sp.hstack([sp.csr_matrix(Xd),
                            sp.csr_matrix((np.ones(n), (r, tix[pr.team_id].to_numpy())), shape=(n, T)),
                            sp.csr_matrix((np.ones(n), (r, pix[pr.player_id_i].to_numpy())), shape=(n, P)),
                            sp.csr_matrix((np.ones(n), (r, pix[pr.player_id_j].to_numpy())), shape=(n, P))]).tocsr()
        self.off = np.log(pr["T"].to_numpy() / 90)
        self.y = pr.w.to_numpy(float)
        self.pen = np.r_[1e-8, np.full(len(EDGE_FEATS), EDGE_RIDGE["beta"]), np.full(T, EDGE_RIDGE["team"]),
                         np.full(2 * P, EDGE_RIDGE["player"])]
        self.dates = pr.date.to_numpy()

    def fit(self, as_of, theta0=None, half_life=365.0):
        rows = np.flatnonzero(self.dates < np.datetime64(as_of))
        X, y, off = self.X[rows], self.y[rows], self.off[rows]
        age = (np.datetime64(as_of) - self.dates[rows]) / np.timedelta64(1, "D")
        w = 0.5 ** (age / half_life)
        XT = X.T.tocsr()
        pen = self.pen * w.sum() / max(len(np.unique(self.pr.match_id.to_numpy()[rows])), 1) / 50

        def f(th):
            eta = X @ th + off
            lam = np.exp(eta)
            return np.sum(w * (lam - y * eta)) + np.sum(pen * th ** 2), XT @ (w * (lam - y)) + 2 * pen * th

        if theta0 is None:
            theta0 = np.zeros(X.shape[1])
            theta0[0] = np.log(y.sum() / np.exp(off).sum())
        return minimize(f, theta0, jac=True, method="L-BFGS-B", options=dict(maxiter=2000)).x

    def predict(self, th, rows):
        return np.exp(self.X[rows] @ th + self.off[rows])


def rolling_edges(em: EdgeModel):
    pr = em.pr
    lam = np.full(len(pr), np.nan)
    month = pr.date.dt.to_period("M")
    th = None
    for mo, idx in pr.groupby(month).groups.items():
        idx = np.asarray(idx)
        as_of = pr.date.iloc[idx].min().floor("D")
        if (pr.date < as_of).sum() < 20000:
            continue
        th = em.fit(as_of, th)
        lam[idx] = em.predict(th, idx)
        print("edges", mo, flush=True)
    return lam, th


# --------------------------------------------------------------------------- metrics on a (predicted) network
def network_metrics(ids, xy, is_gk, W):
    """§5.2 shape (outfield) and §5.4 structure metrics for one XI. W[i, j] = passes i -> j (expected or actual)."""
    out = {}
    of = ~is_gk
    x, y = xy[of, 0], xy[of, 1]
    out["centroid_x"], out["def_line_height"] = float(x.mean()), float(np.sort(x)[:4].mean())
    out["att_height"] = float(np.sort(x)[-3:].mean())
    out["width"] = float(2 * np.sqrt(((y - y.mean()) ** 2).mean()))
    tot = W.sum()
    if tot <= 0:
        return out
    pij = W / tot
    m = pij > 0
    out["passes"] = float(tot)
    out["entropy"] = float(-(pij[m] * np.log(pij[m])).sum() / np.log(m.sum()))
    strength = W.sum(0) + W.sum(1)
    out["gini_strength"] = N.gini(strength)
    out["reciprocity"] = float(np.minimum(W, W.T).sum() / tot)
    dx = xy[None, :, 0] - xy[:, None, 0]
    dy = xy[None, :, 1] - xy[:, None, 1]
    out["verticality"] = float((pij * dx).sum())
    out["forward_share"] = float(pij[dx > 5].sum())
    out["pass_span"] = float((pij * np.hypot(dx, dy)).sum())
    out["flank_share"] = float(pij[:, np.abs(xy[:, 1] - PITCH_Y / 2) > 20].sum())
    out["gk_share"] = float(pij[is_gk, :].sum() + pij[:, is_gk].sum() - pij[np.ix_(is_gk, is_gk)].sum())
    G = nx.DiGraph()
    n = len(ids)
    for i in range(n):
        for j in range(n):
            if i != j and W[i, j] > 1e-9:
                G.add_edge(i, j, distance=1.0 / W[i, j])
    out["max_betweenness"] = float(max(nx.betweenness_centrality(G, weight="distance", normalized=True).values()))
    return out


METRICS = ["centroid_x", "def_line_height", "att_height", "width", "passes", "entropy", "gini_strength", "reciprocity",
           "verticality", "forward_share", "pass_span", "flank_share", "gk_share", "max_betweenness"]


def team_metrics(pr, lam, st, P):
    """Predicted metrics per team-match from the predicted XI network."""
    s = st.assign(px=P[:, 0], py=P[:, 1]).set_index(["match_id", "player_id"])
    rows = []
    pr = pr.assign(lam=lam)
    for (mid, tid), g in pr[pr.lam.notna()].groupby(["match_id", "team_id"]):
        ids = np.unique(np.r_[g.player_id_i, g.player_id_j])
        ix = {p: k for k, p in enumerate(ids)}
        W = np.zeros((len(ids), len(ids)))
        W[g.player_id_i.map(ix), g.player_id_j.map(ix)] = g.lam.to_numpy()
        sub = s.loc[[(mid, p) for p in ids]]
        rows.append(dict(match_id=mid, team_id=tid, **network_metrics(ids, sub[["px", "py"]].to_numpy(float),
                                                                       sub.is_gk.to_numpy(bool), W)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- main
def main():
    m, st = load()
    gpath = REPORTS / "netpred_position_grid.csv"
    g = pd.read_csv(gpath) if gpath.exists() else position_grid(st)
    best = g.sort_values("tune_mse").iloc[0]
    kt, kp, hl = float(best.k_team), float(best.k_player), float(best.hl_player)
    P, pslot = run_positions(st, kt, kp, hl)
    st["slot_pred"] = pslot
    tmpl_line = {(F, s): k for F, lines in N.TEMPLATES.items() for k, line in enumerate(lines) for s, _, _ in line}
    st["slot_line_pred"] = [tmpl_line.get((F, s), np.nan) if s is not None else np.nan for F, s in zip(st.formation, pslot)]
    st.loc[st.is_gk, "slot_line_pred"] = -1
    st["x_pred"], st["y_pred"] = P[:, 0], P[:, 1]
    st[["match_id", "team_id", "player_id", "is_gk", "formation", "slot_pred", "slot_line_pred", "x_pred", "y_pred",
        "slot", "x_m", "y_m"]].to_parquet(PROC / "pred_positions.parquet", index=False)

    # position validation (outfield starters), against two simple baselines
    out = ~st.is_gk.to_numpy() & st.x_m.notna().to_numpy()
    P_slot, _ = run_positions(st, 1e9, 1e9, np.inf)           # league slot average only
    P_own = np.full_like(P, np.nan)
    run_positions(st, kt, kp, hl, own_out=P_own)               # player's own recent average position
    val = []
    for split, seas in (("tuning", TUNE), ("test", TEST)):
        sel = out & st.season.isin(seas).to_numpy()
        act = st.loc[sel, ["x_m", "y_m"]].to_numpy()
        for name, Q in (("model", P), ("slot average only", P_slot), ("own recent average", P_own)):
            e = Q[sel] - act
            val.append(dict(part="positions", split=split, variant=name, rmse_x=float(np.sqrt((e[:, 0] ** 2).mean())),
                            rmse_y=float(np.sqrt((e[:, 1] ** 2).mean())), n=int(sel.sum())))
        val.append(dict(part="positions", split=split, variant="slot agreement (pre-match vs observed)",
                        share=float((st.loc[sel, "slot_pred"] == st.loc[sel, "slot"]).mean()), n=int(sel.sum())))

    # edges
    pr = pair_table(st, P, m)
    em = EdgeModel(pr)
    lam, th_last = rolling_edges(em)
    pr["lam"] = lam
    for split, seas in (("tuning", TUNE), ("test", TEST)):
        sel = pr.season.isin(seas).to_numpy() & ~np.isnan(lam)
        y, mu = pr.w.to_numpy()[sel], lam[sel]
        dev = 2 * np.sum(np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0) - (y - mu))
        # null: team-match average rate per pair-minute (uses the observed total: an upper bound on a volume-only model)
        g_ = pr[sel].assign(rate=pr.w[sel] / pr["T"][sel])
        tot = g_.groupby(["match_id", "team_id"]).w.transform("sum") / g_.groupby(["match_id", "team_id"])["T"].transform("sum")
        mu0 = (tot * g_["T"]).to_numpy()
        dev0 = 2 * np.sum(np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / np.maximum(mu0, 1e-9)), 0) - (y - mu0))
        val.append(dict(part="edges", split=split, variant="gravity model", deviance_explained_vs_volume_only=float(1 - dev / dev0),
                        corr=float(np.corrcoef(y, mu)[0, 1]), n=int(sel.sum())))

    # predicted team metrics vs actual, and against the team's as-of own style
    tmp = team_metrics(pr, lam, st, P)
    tmp.to_parquet(PROC / "pred_network_metrics.parquet", index=False)
    act = pd.read_parquet(PROC / "team_match_network.parquet")
    sty = pd.read_parquet(PROC / "team_style_asof.parquet")
    j = tmp.merge(act[["match_id", "team_id", "season"] + METRICS], on=["match_id", "team_id"], suffixes=("_pred", ""))
    j = j.merge(sty[["match_id", "team_id"] + [f"own_{c}" for c in METRICS] + [f"imposed_{c}" for c in METRICS]],
                on=["match_id", "team_id"], how="left")
    for split, seas in (("tuning", TUNE), ("test", TEST)):
        q = j[j.season.isin(seas)]
        for c in METRICS:
            ok = q[[c, f"{c}_pred", f"own_{c}", f"imposed_{c}"]].dropna()
            if len(ok) < 50:
                continue
            yv = ok[c].to_numpy()
            r = float(np.corrcoef(yv, ok[f"{c}_pred"])[0, 1])
            # does the predicted XI network add to the team's own and imposed style? (in-sample OLS R2 on the split)
            def r2(cols):
                X = np.column_stack([np.ones(len(ok))] + [ok[k].to_numpy() for k in cols])
                b = np.linalg.lstsq(X, yv, rcond=None)[0]
                return 1 - np.var(yv - X @ b) / np.var(yv)
            val.append(dict(part="team metrics", split=split, variant=c, corr_pred_actual=r,
                            r2_style=float(r2([f"own_{c}", f"imposed_{c}"])),
                            r2_style_plus_pred=float(r2([f"own_{c}", f"imposed_{c}", f"{c}_pred"])), n=len(ok)))
    v = pd.DataFrame(val)
    v.to_csv(REPORTS / "netpred_validation.csv", index=False)
    summ = dict(position_settings=dict(k_team=kt, k_player=kp, hl_player=hl), edge_ridge=EDGE_RIDGE,
                edge_coefs=dict(zip(["intercept"] + EDGE_FEATS, [float(x) for x in th_last[:1 + len(EDGE_FEATS)]])))
    (REPORTS / "netpred_summary.json").write_text(json.dumps(summ, indent=2))
    pd.set_option("display.width", 200)
    print(v.round(3).to_string())
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
