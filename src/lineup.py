"""Step 5: expected points, the lineup picker and the 70% rule (methodology §10.1, §13, §14), scored with L1.

Everything here is as of the match date (§2.3). For a team A facing team B (B's actual XI, §12 deferred):

Expected points (§10.1). L1's pre-match npxG for each side (outcome.predict_matches, written out here so any
candidate XI can be scored):
    log lam_A = mu + h*home_A + G + log L90 + tau_A - tau'_B + O_A - D_B      (times the level calibration)
    O_A = sum_{i in XI} f_i o_i + sum_i (1 - f_i) * o_bench_A                    (likewise D_A with d)
f_i is the player's expected share of the match (outcome.F_* rule, from his earlier starts), o_bench_A the team's
average substitute. Goals ~ Poisson(c (lam + penalty xG)) with Dixon-Coles rho, c and penalty xG from L0's as-of
values for the date (the L1 conversion); EP_A = 3 P(A wins) + P(draw).

Lineup picker (§13). The formation is the teamsheet's (decided 2026-10-19: L1 cannot judge shape, §9.4, §15.2).
  eligible:  outfielder i may fill slot r if r's position group (CB, FB, CM, AM, W, FW) is his as-of profile
             group or a group he has started in during the previous 365 days; only the matchday squad (§2.1).
  keeper:    the incumbent (most recency-weighted starts for the team among the squad's keepers, §8).
  usual XI:  §13.4's baseline, ignoring the opponent: Hungarian assignment maximising each player's
             recency-weighted starts for this team in that exact slot (+ half weight for the slot's group).
  best XI:   maximises exact EP. First-order integer program (§13.2): v_i = g_A Δlog-lam_A + g_B Δlog-lam_B per
             player (gradients of EP at the usual XI; slot-independent because L1 effects are), solved as an
             assignment problem, then single bench<->starter swaps while exact EP improves (§13.3).
  neutral:   the same search against a league-average opponent (as-of averages), i.e. the best XI that ignores
             who the opponent is.
  70% rule:  P(better) = share of draws from L1's Laplace posterior in which EP(best) > EP(usual), opponent's XI
             drawn jointly; recommend the best XI only if P(better) > 0.7, otherwise the usual XI (§13.4).
Laplace posterior: covariance phi * H^-1, H = X'WX + 2*diag(penalties) at the MAP, phi = weighted Pearson ratio
sum w (y - mu)^2 / sum w mu (quasi-Poisson dispersion). Bench averages and the game-state term are held at the MAP.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg as sla
from scipy.optimize import linear_sum_assignment
from scipy.stats import poisson

sys.path.insert(0, str(Path(__file__).resolve().parent))
import network as N  # noqa: E402
from outcome import (F_HL, F_K, F_PRIOR_GK, F_PRIOR_OUT, FEATS, GROUPS, N_DENSE, PROC, Config,  # noqa: E402
                     StintData, design, penalties, player_effects, ts, unpack)

L1_CFG = Config(lam_tau=4.0, lam_u=300.0, lam_b=10000.0)
THRESHOLD = 0.7
N_DRAWS = 300
MAX_GOALS = 10
BIG = 1e6
SLOT_GROUP = {"LCB": "CB", "RCB": "CB", "CB": "CB", "LB": "FB", "RB": "FB", "LWB": "FB", "RWB": "FB",
              "DM": "CM", "LDM": "CM", "RDM": "CM", "CM": "CM", "LCM": "CM", "RCM": "CM", "CAM": "AM", "SS": "AM",
              "LM": "W", "RM": "W", "LW": "W", "RW": "W", "LAM": "W", "RAM": "W", "ST": "FW", "LS": "FW", "RS": "FW"}
ELIG_DAYS = 365.0
ELIG_SHARE, ELIG_MIN = 0.2, 2     # a secondary group needs 20% of his recent starts (and 2 starts)


def slots_of(F):
    return [s for line in N.TEMPLATES[F] for s, _, _ in line]


# --------------------------------------------------------------------------- expected points
def outcome_probs_vec(lh, la, rho):
    """Vectorised Dixon-Coles W/D/L probabilities (home win, draw, away win) for arrays of goal rates."""
    lh, la = np.atleast_1d(lh).astype(float), np.atleast_1d(la).astype(float)
    g = np.arange(MAX_GOALS + 1)
    M = poisson.pmf(g[None, :], lh[:, None])[:, :, None] * poisson.pmf(g[None, :], la[:, None])[:, None, :]
    if rho:
        M[:, 0, 0] *= 1 - lh * la * rho
        M[:, 0, 1] *= 1 + lh * rho
        M[:, 1, 0] *= 1 + la * rho
        M[:, 1, 1] *= 1 - rho
    M /= M.sum((1, 2), keepdims=True)
    tri = np.tril(np.ones((MAX_GOALS + 1, MAX_GOALS + 1)), -1)
    ph = (M * tri).sum((1, 2))
    pd_ = np.trace(M, axis1=1, axis2=2)
    return ph, pd_, 1 - ph - pd_


@dataclass
class Match:
    """Everything fixed for one side's decision: A's constant terms and the opponent's XI."""
    home: int                      # 1 if A is at home
    base_A: np.ndarray             # log-rate constant of A's attack (scalar or per draw)
    base_B: np.ndarray             # log-rate constant of B's attack
    OB: np.ndarray                 # opponent XI's attacking total
    DB: np.ndarray                 # opponent XI's defensive total
    scale: float
    c: float
    pen_A: float
    pen_B: float
    rho: float

    def rates(self, OA, DA):
        lamA = self.scale * np.exp(self.base_A + OA - self.DB)
        lamB = self.scale * np.exp(self.base_B + self.OB - DA)
        return lamA, lamB

    def ep(self, OA, DA, full=False):
        lamA, lamB = self.rates(OA, DA)
        gA, gB = self.c * (lamA + self.pen_A), self.c * (lamB + self.pen_B)
        if self.home:
            pw, pdr, pl = outcome_probs_vec(gA, gB, self.rho)
        else:
            pl, pdr, pw = outcome_probs_vec(gB, gA, self.rho)
        ep = 3 * pw + pdr
        if full:
            return ep, dict(npxg_for=lamA, npxg_against=lamB, p_win=pw, p_draw=pdr, p_loss=pl)
        return ep

    def gradients(self, OA, DA, h=1e-4):
        """g_A = dEP/dlog lam_A > 0 and g_B = -dEP/dlog lam_B > 0 (§13.2), numerically."""
        e0 = self.ep(OA, DA)[0]
        return float((self.ep(OA + h, DA)[0] - e0) / h), float((self.ep(OA, DA + h)[0] - e0) / h)


# --------------------------------------------------------------------------- history lookups (as of date)
class History:
    """Slot starts, groups and expected shares, queried as of a date (only earlier matches count)."""

    def __init__(self, data: StintData):
        m = data.matches[["match_id", "date"]]
        sl = pd.read_parquet(PROC / "player_slots.parquet")[["match_id", "team_id", "player_id", "slot"]]
        sl = sl.merge(m, on="match_id")
        sl["group"] = sl.slot.map(SLOT_GROUP)
        self.slots = sl.sort_values("date").reset_index(drop=True)
        self.slot_day = days(self.slots.date)
        pm = pd.read_parquet(PROC / "player_match.parquet")
        pm = pm.merge(m.rename(columns={"date": "mdate"}), on="match_id")
        self.pm = pm
        gks = pm[pm.started & pm.is_gk & pm.played][["match_id", "team_id", "player_id", "mdate"]]
        self.gk = gks.assign(day=days(gks.mdate))
        st = data.pt[data.pt.started].sort_values("date")
        self.starts = {p: (days(g.date), g.share.to_numpy(), bool(g.is_gk.iloc[0])) for p, g in st.groupby("player_id")}
        self.prof_group = data.prof.group
        pr = data.prof.reset_index()[["player_id", "date"] + FEATS + ["group"]].sort_values("date")
        self.prof_by_player = {p: g.reset_index(drop=True) for p, g in pr.groupby("player_id")}

    def latest_profile(self, players, as_of):
        """Each player's most recent profile row dated before as_of (blank if none)."""
        rows = []
        for p in players:
            g = self.prof_by_player.get(p)
            g = g[g.date < as_of] if g is not None else None
            rows.append(g.iloc[-1] if g is not None and len(g) else pd.Series(dtype=object))
        return pd.DataFrame(rows, columns=FEATS + ["group"]).reset_index(drop=True)

    def fshare(self, players, d, is_gk):
        out = np.empty(len(players))
        for k, (p, g) in enumerate(zip(players, is_gk)):
            prior = F_PRIOR_GK if g else F_PRIOR_OUT
            h = self.starts.get(p)
            if h is None:
                out[k] = prior
                continue
            dd, v, _ = h
            sel = dd < d
            w = 0.5 ** ((d - dd[sel]) / F_HL)
            out[k] = (np.sum(w * v[sel]) + F_K * prior) / (np.sum(w) + F_K)
        return np.where(is_gk, 1.0, out)

    def slot_weights(self, team, d, hl):
        """Recency-weighted starts for this team by (player, slot) and (player, group), before day d."""
        s = self.slots[(self.slot_day < d) & (self.slots.team_id == team).to_numpy()]
        w = 0.5 ** ((d - days(s.date)) / hl)
        s = s.assign(w=w)
        return s.groupby(["player_id", "slot"]).w.sum(), s.groupby(["player_id", "group"]).w.sum()

    def eligible_groups(self, match_id, players, d, with_primary=False):
        """Position groups a player may fill: his as-of profile group (his primary group), plus any group that
        holds at least ELIG_SHARE of his starts (and ELIG_MIN starts) in the previous 365 days."""
        s = self.slots[(self.slot_day < d) & (self.slot_day >= d - ELIG_DAYS)]
        s = s[s.player_id.isin(players)]
        cnt = s.groupby(["player_id", "group"]).size()
        tot = s.groupby("player_id").size()
        out, prim = {}, {}
        for p in players:
            g = set()
            if p in tot.index:
                c = cnt.loc[p]
                g = {k for k, v in c.items() if v >= ELIG_MIN and v / tot[p] >= ELIG_SHARE}
            pg = self.prof_group.get((match_id, p))
            if not isinstance(pg, str):
                pg = cnt.loc[p].idxmax() if p in tot.index else None
            if pg is not None:
                g.add(pg)
            out[p], prim[p] = g, pg
        return (out, prim) if with_primary else out

    def incumbent_gk(self, team, keepers, d, hl=365.0):
        if len(keepers) == 1:
            return keepers[0]
        g = self.gk[(self.gk.team_id == team) & (self.gk.day < d) & self.gk.player_id.isin(keepers)]
        if g.empty:
            return keepers[0]
        w = (0.5 ** ((d - g.day) / hl)).groupby(g.player_id).sum()
        return int(w.idxmax())


def days(s):
    return ((pd.to_datetime(s) - pd.Timestamp("2000-01-01")).dt.total_seconds() / 86400).to_numpy()


# --------------------------------------------------------------------------- the Laplace posterior of a fit
def laplace_draws(data: StintData, th, info, cfg: Config, n_draws=N_DRAWS, seed=0):
    """Draws of all L1 parameters from N(theta_hat, phi * H^-1) at the MAP of the fit."""
    rows, zs = info["rows"], info["zs"]
    X = design(data, rows, zs, None)
    s = data.st.iloc[rows]
    as_of = info["as_of"]
    age = (as_of - s.date).dt.total_seconds().to_numpy() / 86400
    w = 0.5 ** (age / cfg.half_life)
    cur = data.season_idx[data.st.loc[data.st.date >= as_of, "season"].iloc[0]] if (data.st.date >= as_of).any() \
        else s.season_idx.max()
    w = w * cfg.carry ** np.clip(cur - s.season_idx.to_numpy(), 0, None)
    mu = s.e.to_numpy() * np.exp(X @ th)
    y = s.npxg.to_numpy()
    phi = float(np.sum(w * (y - mu) ** 2) / np.sum(w * mu))
    pen, _ = penalties(data, cfg, {})
    H = (X.T @ X.multiply((w * mu)[:, None]).tocsr()).toarray() + 2 * np.diag(pen)
    L = np.linalg.cholesky(H)
    z = np.random.default_rng(seed).standard_normal((len(th), n_draws))
    D = sla.solve_triangular(L.T, z, lower=False) * np.sqrt(phi)
    return th[None, :] + D.T, phi


# --------------------------------------------------------------------------- per-day model state
class DayModel:
    """One L1 fit (as of a match day) with its posterior draws; scores any XI against any opponent XI."""

    def __init__(self, data: StintData, th, info, draws, sub_avg, neutral, hist: History):
        self.data, self.th, self.info, self.hist = data, th, info, hist
        self.q = unpack(data, th)
        self.qs = [unpack(data, t) for t in draws]
        self.sub_avg = sub_avg
        self.neutral = neutral           # (O, D) of an average XI, from the previous 120 days' actual XIs
        self.group_means = {"all": (0.0, 0.0)}   # as-of mean (o, d) of starters by position group; set by caller
        self.d = days(pd.Series([info["as_of"]]))[0]

    def effects(self, match_id, players):
        """MAP o, d for (match, player) pairs, and their draws (n_draws x n)."""
        data = self.data
        players = list(players)
        mu, sd = self.info["zs"]
        pr = data.prof.reindex(pd.MultiIndex.from_arrays([np.full(len(players), match_id), np.asarray(players)]))
        miss = pr[FEATS].isna().all(1).to_numpy()
        if miss.any():   # not in this squad (counterfactual only): his latest earlier profile, as of the date
            lp = self.hist.latest_profile([p for p, m_ in zip(players, miss) if m_], self.info["as_of"])
            pr = pr[FEATS + ["group"]].copy()
            pr["group"] = pr.group.astype(object)
            pr.loc[miss, FEATS] = lp[FEATS].to_numpy(float)
            pr.loc[miss, "group"] = lp["group"].astype(object).to_numpy()
        g = pr.group.map({g: j for j, g in enumerate(GROUPS)}).fillna(len(GROUPS)).astype(int).to_numpy()
        Z = np.nan_to_num((pr[FEATS].to_numpy(float) - mu[g]) / sd[g])
        pc = pd.Series(np.arange(data.P), index=data.players).reindex(players).to_numpy()
        has = ~np.isnan(pc)
        pci = np.where(has, pc, 0).astype(int)
        qm = self.q
        o = Z @ qm["bo"] + qm["u"][pci] * has
        d = Z @ qm["bd"] + qm["v"][pci] * has
        BO = np.array([q["bo"] for q in self.qs])
        BD = np.array([q["bd"] for q in self.qs])
        U = np.array([q["u"][pci] * has for q in self.qs])
        V = np.array([q["v"][pci] * has for q in self.qs])
        return o, d, BO @ Z.T + U, BD @ Z.T + V

    def team_const(self, att, dfn, home, q=None):
        q = q or self.q
        t = self.data.tix
        return q["mu"] + q["home"] * home + self.info["G"] + np.log(self.info["L90"]) + q["ta"][t[att]] - q["td"][t[dfn]]

    def team_const_draws(self, att, dfn, home):
        return np.array([self.team_const(att, dfn, home, q) for q in self.qs])


# --------------------------------------------------------------------------- one side's decision
@dataclass
class Decision:
    match_id: int
    team_id: int
    formation: str
    keeper: int
    xis: dict = field(default_factory=dict)          # name -> list of 10 outfield player ids (+ keeper separately)
    slots: dict = field(default_factory=dict)        # name -> {player: slot}
    scores: dict = field(default_factory=dict)       # name -> dict(ep, npxg_for, npxg_against, ...)
    p_better: float = np.nan
    gA: float = np.nan
    gB: float = np.nan
    notes: list = field(default_factory=list)


def assign(players, slots, score, elig):
    """Max-score one-to-one assignment of players to slots; ineligible pairs cost BIG (used only if unavoidable)."""
    C = np.array([[-(score[i]) + (0 if SLOT_GROUP[s] in elig[p] else BIG) for s in slots]
                  for i, p in enumerate(players)])
    r, c = linear_sum_assignment(C)
    return {players[a]: slots[b] for a, b in zip(r, c)}, float(C[r, c].sum())


def usual_xi(hist: History, match_id, team, out, F, d, hl):
    """§13.4 baseline: the manager's usual personnel for formation F among the available outfielders, ignoring
    the opponent. Score of (player, slot) = recency-weighted starts for this team in that slot + half weight for
    starts in the slot's position group; ineligible pairs only if unavoidable."""
    slots = slots_of(F)
    elig = hist.eligible_groups(match_id, out, d)
    w_slot, w_grp = hist.slot_weights(team, d, hl)
    fam = np.array([[w_slot.get((p, s), 0.0) + 0.5 * w_grp.get((p, SLOT_GROUP[s]), 0.0) for s in slots] for p in out])
    Cu = -fam + np.array([[0 if SLOT_GROUP[s] in elig[p] else BIG for s in slots] for p in out])
    r, cc = linear_sum_assignment(Cu)
    return [out[a] for a in r], {out[a]: slots[b] for a, b in zip(r, cc)}, elig, fam, Cu, Cu[r, cc].sum() < BIG / 2


def decide(dm: DayModel, mt, team, squad, opp_xi, conv, usual_hl, full_pool=None):
    """squad: matchday squad rows of `team` (player_id, is_gk, started); opp_xi: opponent starters (player_id, is_gk);
    full_pool: (outfielders, keepers) the team could pick at full strength (for the forced-change check, §15.4)."""
    hist, d = dm.hist, dm.d
    home = int(team == mt.home_team_id)
    opp = mt.away_team_id if home else mt.home_team_id
    F = N.sheet_template(mt.home_formation if home else mt.away_formation) or "4-2-3-1"
    slots = slots_of(F)
    keepers = squad.loc[squad.is_gk, "player_id"].tolist()
    gk = hist.incumbent_gk(team, keepers, d) if keepers else None
    out = squad.loc[~squad.is_gk, "player_id"].tolist()
    dec = Decision(mt.match_id, team, F, gk)

    # player effects, MAP and draws, for our squad and the opponent XI
    pool = ([gk] if gk is not None else []) + out
    o, dd, os_, ds_ = dm.effects(mt.match_id, pool)
    isgk = np.array([p == gk for p in pool])
    f = hist.fshare(pool, d, isgk)
    (sub_o, sub_d), (lo, ld) = dm.sub_avg
    so, sd = sub_o.get(team, lo), sub_d.get(team, ld)
    # per-player contribution to the team totals (bench share filled by the team's average substitute)
    cO, cD = f * o + (1 - f) * so, f * dd + (1 - f) * sd
    # Within-group comparison (§9.4): for choosing, each outfielder counts by his effect relative to the as-of
    # average starter of his own (primary) position group; the usual XI's totals are kept at L1's raw forecast, so
    # other XIs differ from it only by within-group differences. Position-group levels are not identified.
    _, prim = hist.eligible_groups(mt.match_id, out, d, with_primary=True)
    gm = dm.group_means
    lvl = np.array([gm.get(prim.get(p), gm["all"]) if p != gk else (0.0, 0.0) for p in pool])
    cOc, cDc = cO - f * lvl[:, 0], cD - f * lvl[:, 1]
    cOs, cDs = f * (os_ - lvl[:, 0]) + (1 - f) * so, f * (ds_ - lvl[:, 1]) + (1 - f) * sd
    idx = {p: k for k, p in enumerate(pool)}
    K = dict(O=0.0, D=0.0)

    opl = opp_xi.player_id.tolist()
    oo, od, oos, ods = dm.effects(mt.match_id, opl)
    of = hist.fshare(opl, d, opp_xi.is_gk.to_numpy())
    (osub_o, osub_d) = (sub_o.get(opp, lo), sub_d.get(opp, ld))
    OB = np.sum(of * oo) + np.sum(1 - of) * osub_o
    DB = np.sum(of * od) + np.sum(1 - of) * osub_d
    OBs = (of * oos).sum(1) + np.sum(1 - of) * osub_o
    DBs = (of * ods).sum(1) + np.sum(1 - of) * osub_d

    c, rho = conv["c"], conv["rho"]
    penA, penB = (conv["pen_home"], conv["pen_away"]) if home else (conv["pen_away"], conv["pen_home"])
    scale = dm.info["scale"]
    M = Match(home, dm.team_const(team, opp, home), dm.team_const(opp, team, 1 - home), OB, DB, scale, c, penA, penB, rho)
    Ms = Match(home, dm.team_const_draws(team, opp, home), dm.team_const_draws(opp, team, 1 - home), OBs, DBs,
               scale, c, penA, penB, rho)
    nO, nD = dm.neutral
    q = dm.q
    Mn = Match(home, q["mu"] + q["home"] * home + dm.info["G"] + np.log(dm.info["L90"]) + q["ta"][dm.data.tix[team]],
               q["mu"] + q["home"] * (1 - home) + dm.info["G"] + np.log(dm.info["L90"]) - q["td"][dm.data.tix[team]],
               nO, nD, scale, c, penA, penB, rho)

    def totals(xi, draws=False, raw=False):
        k = [idx[gk]] + [idx[p] for p in xi] if gk is not None else [idx[p] for p in xi]
        if raw:
            return cO[k].sum(), cD[k].sum()
        if draws:
            return cOs[:, k].sum(1) + K["O"], cDs[:, k].sum(1) + K["D"]
        return cOc[k].sum() + K["O"], cDc[k].sum() + K["D"]

    if len(out) < 10:
        dec.notes.append("fewer than 10 outfielders in squad")
        return dec, None
    usual, dec.slots["usual"], elig, fam, Cu, ok = usual_xi(hist, mt.match_id, team, out, F, d, usual_hl)
    ku = [idx[gk]] + [idx[p] for p in usual] if gk is not None else [idx[p] for p in usual]
    K["O"], K["D"] = cO[ku].sum() - cOc[ku].sum(), cD[ku].sum() - cDc[ku].sum()

    # Search is like-for-like: a player fills only slots of his primary position group, or the group of the slot
    # he fills in the usual XI (the manager's own out-of-position choices); anything else only if unavoidable.
    ug = {p: SLOT_GROUP[s_] for p, s_ in dec.slots["usual"].items()}
    Vp = np.array([[0.0 if SLOT_GROUP[s] in (prim.get(p), ug.get(p)) else BIG for s in slots] for p in out])

    def violations(xi):
        sub = [out.index(p) for p in xi]
        C = Vp[sub]
        rr, ccol = linear_sum_assignment(C)
        return int(round(C[rr, ccol].sum() / BIG))

    allowed = violations(usual)
    if allowed:
        dec.notes.append(f"{allowed} out-of-group placement(s) in the usual XI")
    feas_cache = {}

    def feasible(xi):
        k = frozenset(xi)
        if k not in feas_cache:
            feas_cache[k] = violations(xi) <= allowed
        return feas_cache[k]

    def search(Mx, start):
        OA0, DA0 = totals(start)
        gA, gB = Mx.gradients(np.atleast_1d(OA0), np.atleast_1d(DA0))
        v = {p: gA * cOc[idx[p]] + gB * cDc[idx[p]] for p in out}
        C = np.array([[-v[p] for s in slots] for p in out]) + Vp
        rr, ccol = linear_sum_assignment(C)
        cur = [out[a] for a in rr]
        if not feasible(cur):
            cur = list(start)
        cur_ep = float(Mx.ep(*map(np.atleast_1d, totals(cur)))[0])
        while True:
            best = None
            bench = [p for p in out if p not in cur]
            for po in cur:
                for pi in bench:
                    cand = [pi if x == po else x for x in cur]
                    if not feasible(cand):
                        continue
                    e = float(Mx.ep(*map(np.atleast_1d, totals(cand)))[0])
                    if e > cur_ep + 1e-9 and (best is None or e > best[1]):
                        best = (cand, e)
            if best is None:
                return cur, (gA, gB)
            cur, cur_ep = best

    best, (gA, gB) = search(M, usual)
    neutral, _ = search(Mn, usual)
    dec.gA, dec.gB = gA, gB
    actual = squad.loc[squad.started & ~squad.is_gk, "player_id"].tolist()
    act_gk = squad.loc[squad.started & squad.is_gk, "player_id"].tolist()

    def disp(xi):
        sub = [out.index(p) for p in xi]
        C = -fam[sub] + Vp[sub] + np.array([[0 if SLOT_GROUP[s] in elig[out[i]] else BIG for s in slots] for i in sub])
        rr, ccol = linear_sum_assignment(C)
        return {xi[a]: slots[b] for a, b in zip(rr, ccol)}

    for name, xi in (("best", best), ("neutral", neutral)):
        dec.slots[name] = disp(xi)
    # P(better): best vs usual over the posterior draws, opponent drawn jointly
    if set(best) == set(usual):
        dec.p_better = np.nan
    else:
        eb, eu = Ms.ep(*totals(best, True)), Ms.ep(*totals(usual, True))
        dec.p_better = float(np.mean(eb > eu))
    rec = best if (dec.p_better > THRESHOLD) else usual
    dec.xis = dict(usual=usual, best=best, neutral=neutral, recommended=rec, actual=actual)
    dec.slots["recommended"] = dec.slots["best"] if rec is best else dec.slots["usual"]
    def score_any(k, xi):
        """Raw L1 totals (L1's forecast for this XI) and within-group totals (the picker's comparison scale)."""
        pl = ([k] if k is not None else []) + list(xi)
        ao, ad, _, _ = dm.effects(mt.match_id, pl)
        af = hist.fshare(pl, d, np.array([p == k for p in pl]))
        _, pr_ = hist.eligible_groups(mt.match_id, list(xi), d, with_primary=True)
        lv = np.array([(0.0, 0.0) if p == k else gm.get(pr_.get(p), gm["all"]) for p in pl])
        raw = (np.sum(af * ao) + np.sum(1 - af) * so, np.sum(af * ad) + np.sum(1 - af) * sd)
        return raw, (raw[0] - np.sum(af * lv[:, 0]) + K["O"], raw[1] - np.sum(af * lv[:, 1]) + K["D"])

    if full_pool is not None:
        fo, fk = full_pool
        if len(fo) >= 10:
            fxi = usual_xi(hist, mt.match_id, team, list(fo), F, d, usual_hl)[0]
            fgk = hist.incumbent_gk(team, list(fk), d) if len(fk) else gk
            dec.xis["full_usual"] = fxi
            dec.keeper_full = fgk
    for name, xi in dec.xis.items():
        k = (act_gk[0] if act_gk else gk) if name == "actual" else (dec.keeper_full if name == "full_usual" else gk)
        (OA, DA), (OW, DW) = score_any(k, xi)
        e, det = M.ep(np.atleast_1d(OA), np.atleast_1d(DA), full=True)
        ew, detw = M.ep(np.atleast_1d(OW), np.atleast_1d(DW), full=True)
        en = Mn.ep(np.atleast_1d(OW), np.atleast_1d(DW))
        dec.scores[name] = dict(ep=float(ew[0]), ep_neutral=float(en[0]), ep_raw=float(e[0]), OA=float(OW),
                                DA=float(DW), npxg_for=float(detw["npxg_for"][0]),
                                npxg_against=float(detw["npxg_against"][0]), p_win=float(detw["p_win"][0]),
                                p_draw=float(detw["p_draw"][0]), raw_npxg_for=float(det["npxg_for"][0]),
                                raw_npxg_against=float(det["npxg_against"][0]))
    dec.keeper_actual = act_gk[0] if act_gk else None
    ctx = dict(M=M, Ms=Ms, Mn=Mn, cO=cOc, cD=cDc, idx=idx, gk=gk, totals=totals, elig=elig, out=out, f=f, o=o, d=dd,
               pool=pool)
    return dec, ctx


def swap_breakdown(dec: Decision, ctx, a="usual", b="recommended"):
    """§14 player terms: each swap from XI a to XI b, matched by slot, with its EP change on its own
    (one swap applied to XI a) and the attack / defence split through the gradients."""
    sa, sb = dec.slots[a], dec.slots[b]
    outs = [p for p in dec.xis[a] if p not in dec.xis[b]]
    ins = [p for p in dec.xis[b] if p not in dec.xis[a]]
    if not outs:
        return []
    by_slot_b = {s: p for p, s in sb.items()}
    rows = []
    M, totals, cO, cD, idx = ctx["M"], ctx["totals"], ctx["cO"], ctx["cD"], ctx["idx"]
    base = float(M.ep(*map(np.atleast_1d, totals(dec.xis[a])))[0])
    used = set()
    for po in outs:
        s = sa[po]
        pi = by_slot_b.get(s)
        if pi is None or pi not in ins or pi in used:
            pi = next(x for x in ins if x not in used)
        used.add(pi)
        cand = [pi if x == po else x for x in dec.xis[a]]
        e = float(M.ep(*map(np.atleast_1d, totals(cand)))[0])
        rows.append(dict(slot=s, slot_in=sb.get(pi), out=po, inn=pi, dEP=e - base,
                         dEP_attack=dec.gA * (cO[idx[pi]] - cO[idx[po]]),
                         dEP_defence=dec.gB * (cD[idx[pi]] - cD[idx[po]])))
    return rows
