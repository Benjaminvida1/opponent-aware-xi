"""Step 3: player-effects outcome model, L1 (methodology §8, §9 without the style and matchup terms).

For every directed stint s (team A attacking team B, duration t_s minutes):

    npxG_s ~ quasi-Poisson( (t_s / 90) * lambda_s )
    log lambda_s = mu + h*home + g(state_s) + rho(manadv_s) + tau_A - tau'_B
                   + sum_{i in A on pitch} o_i - sum_{j in B on pitch} d_j
    o_i = x_i' beta_o + u_i,   d_j = x_j' beta_d + v_j

x_i is the player's Step 2 profile as of the match (37 measurements, z-scored with constants from
earlier squad rows only); goalkeepers have no profile and enter through u, v only. Fitted as a MAP
estimate: weighted Poisson likelihood plus ridge penalties, which is the posterior mode of the
hierarchical model with fixed prior variances (§9.2, "MAP with ridge penalties as a faster first
version"):

    lambda_tau * sum (tau - prior)^2   team effects (one per team, decayed over time like L0)
    lambda_u   * sum (u^2 + v^2)       player deviations from the box-score prior
    lambda_b   * sum (beta_o^2 + beta_d^2)
    5 * sum g^2, 2 * sum rho^2         game state ~ N(0, 0.3^2), man advantage ~ N(0, 0.5^2)

Match weights as L0: 0.5 ** (age_days / half_life) * carry ** (seasons back). Every fit uses only
stints played before the prediction date (§2.3); promoted teams' team effects start at the mean of
the relegated teams', fitted as of the first day of the season (as L0).

Pre-match prediction (§9.3) for a team's starting XI:
    each starter counts for his expected share f_i of the match (shrunk decayed average of his past
    starts; goalkeepers 1), and the rest of his share goes to the team's average substitute
    (minutes-weighted o or d of its substitute appearances in the previous 365 days);
    game state and man count enter as their weighted average effect over the training stints;
    the per-90 rate is scaled to a whole match by the average match length (stoppage time included,
    about 100 minutes) over the previous 365 days, and then by an as-of level calibration (actual over
    predicted npxG for the previous 120 days of matches, re-predicted with the current fit), which keeps
    the forecast on the league's current npxG level as L0's intercept does.
Goals and W/D/L as L0: goals ~ Poisson(c * (npxG + penalty xG)) with the Dixon–Coles correction; c,
the penalty rate and rho are taken from L0's as-of estimates for the same date, so L1 and L0 differ
only in the npxG forecast (and, if kept, the goalkeeper's shot-stopping, §8, see gk.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
SEASONS = ["2023/2024", "2024/2025", "2025/2026", "2026/2027"]
TUNE, TEST = ["2024/2025"], ["2025/2026", "2026/2027"]

FEATS = ["npxg", "np_shots", "headed_shots", "box_touches", "xa", "chances_created", "big_chances_created",
         "dribbles", "dribble_attempts", "final_third_passes", "passes", "touches", "crosses", "long_balls",
         "tackles", "interceptions", "blocks", "recoveries", "clearances", "dribbled_past", "fouls",
         "ground_duels", "aerial_duels", "was_fouled", "pass_acc", "cross_acc", "dribble_succ",
         "ground_duel_win", "aerial_win", "long_ball_acc", "passes_received", "prog_out", "prog_in",
         "x_rel", "y_abs", "betweenness", "pass_share"]          # the 38 profile measurements less side (y_side)
# Compact prior: chance creation and progression, and the main defensive actions and duel quality.
COMPACT = ["npxg", "xa", "box_touches", "chances_created", "dribbles", "prog_in", "prog_out", "pass_acc",
           "tackles", "interceptions", "blocks", "clearances", "aerial_win", "dribbled_past"]
STATES = [-2, -1, 1, 2]
GROUPS = ["CB", "FB", "CM", "AM", "W", "FW"]
NG = len(GROUPS) + 1                                            # + players with no prior group yet
N_DENSE = 2 + len(STATES) + 2                                   # mu, home, g(4), rho(-1, +1)
F_PRIOR_OUT, F_PRIOR_GK, F_K, F_HL = 0.85, 1.0, 3.0, 365.0      # expected-minutes shrinkage


@dataclass
class Config:
    half_life: float = 540.0
    carry: float = 0.75
    lam_tau: float = 4.0
    lam_u: float = 30.0          # np.inf = no player deviations (profile only)
    lam_b: float = 30.0          # np.inf = no box-score prior (beta = 0)
    use_players: bool = True     # False = team effects only (L0 fitted on stints)
    zmode: str = "all"           # profiles z-scored over all outfielders ("all") or within position group ("group")
    feats: str = "full"          # box-score prior on all 37 measurements ("full") or the compact set ("compact")
    # Step 4: extra opponent-aware terms, each (spec, ridge penalty); see extra_columns() for the specs.
    extra: tuple = ()

    def key(self):
        k = f"hl{self.half_life:g}_c{self.carry:g}_t{self.lam_tau:g}_u{self.lam_u:g}_b{self.lam_b:g}_p{int(self.use_players)}_{self.zmode}_{self.feats}"
        return k + "".join(f"__{sp}@{lam:g}" for sp, lam in self.extra)


# --------------------------------------------------------------------------- data
class StintData:
    def __init__(self):
        m = pd.read_parquet(PROC / "matches.parquet").sort_values("date").reset_index(drop=True)
        st = pd.read_parquet(PROC / "stints.parquet")
        spl = pd.read_parquet(PROC / "stint_players.parquet")
        prof = pd.read_parquet(PROC / "player_profile_asof.parquet")
        pm = pd.read_parquet(PROC / "player_match.parquet")
        pt = pd.read_parquet(PROC / "player_times.parquet")
        # all dates as naive UTC
        for df in (m, st, prof, pm):
            df["date"] = naive(df["date"])
        self.matches = m
        self.season_idx = {s: k for k, s in enumerate(SEASONS)}
        st = st.sort_values(["date", "match_id", "stint", "home"], ascending=[True, True, True, False]).reset_index(drop=True)
        st["season_idx"] = st.season.map(self.season_idx)
        st["e"] = st.duration / 90
        self.st = st
        self.teams = np.sort(pd.unique(np.r_[m.home_team_id, m.away_team_id]))
        self.tix = {t: k for k, t in enumerate(self.teams)}
        self.players = np.sort(pm.loc[pm.played, "player_id"].unique())
        self.pix = {p: k for k, p in enumerate(self.players)}
        T, P, F = len(self.teams), len(self.players), len(FEATS)
        self.T, self.P, self.F = T, P, F
        n = len(st)
        # dense covariates
        D = np.zeros((n, N_DENSE))
        D[:, 0] = 1
        D[:, 1] = st.home
        for k, s in enumerate(STATES):
            D[:, 2 + k] = st.state == s
        D[:, 6] = st.manadv == -1
        D[:, 7] = st.manadv == 1
        self.D = D
        r = np.arange(n)
        self.TA = sp.csr_matrix((np.ones(n), (r, st.team_id.map(self.tix))), shape=(n, T))
        self.TD = sp.csr_matrix((np.ones(n), (r, st.opponent_id.map(self.tix))), shape=(n, T))
        # on-pitch players: rows keyed by (match, stint, team)
        key = pd.MultiIndex.from_frame(st[["match_id", "stint", "team_id"]])
        okey = pd.MultiIndex.from_frame(st[["match_id", "stint", "opponent_id"]].set_axis(["match_id", "stint", "team_id"], axis=1))
        rowpos = pd.Series(r, index=key)
        orowpos = pd.Series(r, index=okey)
        spk = pd.MultiIndex.from_frame(spl[["match_id", "stint", "team_id"]])
        ra = rowpos.reindex(spk).to_numpy()      # row where this player's team attacks
        rd = orowpos.reindex(spk).to_numpy()     # row where this player's team defends
        pcol = spl.player_id.map(self.pix).to_numpy()
        ok = ~np.isnan(ra) & ~np.isnan(rd) & ~pd.isna(pcol)
        ra, rd, pcol = ra[ok].astype(int), rd[ok].astype(int), pcol[ok].astype(int)
        self.PA = sp.csr_matrix((np.ones(len(ra)), (ra, pcol)), shape=(n, P))
        self.PD = sp.csr_matrix((np.ones(len(rd)), (rd, pcol)), shape=(n, P))
        # profile sums of outfield on-pitch players (raw values and non-missing counts)
        prof = prof.set_index(["match_id", "player_id"])
        prof = prof[~prof.index.duplicated()]
        self.prof = prof
        out = spl[ok & ~spl.is_gk.to_numpy()]
        pv = prof[FEATS].reindex(pd.MultiIndex.from_frame(out[["match_id", "player_id"]]))
        X = pv.to_numpy(float)
        Nf = (~np.isnan(X)).astype(float)
        X = np.nan_to_num(X)
        gidx = prof.group.reindex(pd.MultiIndex.from_frame(out[["match_id", "player_id"]])).map(
            {g: k for k, g in enumerate(GROUPS)}).fillna(len(GROUPS)).astype(int).to_numpy()
        ra_o = rowpos.reindex(pd.MultiIndex.from_frame(out[["match_id", "stint", "team_id"]])).to_numpy().astype(int)
        rd_o = orowpos.reindex(pd.MultiIndex.from_frame(out[["match_id", "stint", "team_id"]])).to_numpy().astype(int)
        agg = lambda rows, M: sp.csr_matrix((np.ones(len(rows)), (rows, np.arange(len(rows)))), shape=(n, len(rows))) @ M
        # per prior group (six position groups + no group), so profiles can be z-scored within group
        self.SxA = [agg(ra_o[gidx == g], X[gidx == g]) for g in range(NG)]
        self.NfA = [agg(ra_o[gidx == g], Nf[gidx == g]) for g in range(NG)]
        self.SxD = [agg(rd_o[gidx == g], X[gidx == g]) for g in range(NG)]
        self.NfD = [agg(rd_o[gidx == g], Nf[gidx == g]) for g in range(NG)]
        self.SxA_t, self.NfA_t = sum(self.SxA), sum(self.NfA)
        self.SxD_t, self.NfD_t = sum(self.SxD), sum(self.NfD)
        self.prof_dates = prof.date.to_numpy()
        self.prof_X = prof[FEATS].to_numpy(float)
        self.prof_g = prof.group.map({g: k for k, g in enumerate(GROUPS)}).fillna(len(GROUPS)).astype(int).to_numpy()
        # starters and expected minutes
        pm = pm[pm.played].merge(m[["match_id", "date", "season"]], on="match_id", suffixes=("", "_m"))
        pt = pt.merge(pm[["match_id", "player_id", "date"]], on=["match_id", "player_id"])
        T2 = st.groupby("match_id").end.max()
        pt["share"] = ((pt.t_off - pt.t_on) / pt.match_id.map(T2)).clip(0, 1)
        self.pt = pt
        self.starters = {k: g for k, g in pt[pt.started].groupby("match_id")}
        self.fexp = expected_shares(pt)
        self.season_starts = m.groupby("season").date.min()
        self._promoted_prior = {}
        self._extra = {}
        self._mf = None

    # ---------------------------------------------------------------- Step 4 extra terms
    def matchup_table(self):
        if self._mf is None:
            self._mf = pd.read_parquet(PROC / "matchup_features.parquet").set_index(["match_id", "team_id"])
        return self._mf

    def extra(self, cfg: "Config"):
        """Dense matrix of the extra terms for every directed stint, and a (match, attacking team) lookup."""
        if not cfg.extra:
            return None, None
        key = tuple(sp for sp, _ in cfg.extra)
        if key not in self._extra:
            mf = self.matchup_table()
            E = pd.concat([extra_columns(mf, sp) for sp in key], axis=1).fillna(0.0)
            idx = pd.MultiIndex.from_frame(self.st[["match_id", "team_id"]])
            M = E.reindex(idx).fillna(0.0).to_numpy(float)
            self._extra[key] = (M, dict(zip(E.index, E.to_numpy(float))))
        return self._extra[key]

    # ---------------------------------------------------------------- standardization (as of)
    def zstats(self, as_of, zmode: str = "group"):
        """Means and SDs of each measurement, as of the date: pooled ('all') or by prior group ('group')."""
        as_of = ts(as_of)
        keep = self.prof_dates < np.datetime64(as_of)
        X, g = self.prof_X[keep], self.prof_g[keep]
        mu0, sd0 = np.nanmean(X, 0), np.nanstd(X, 0)
        mu, sd = np.tile(mu0, (NG, 1)), np.tile(sd0, (NG, 1))
        if zmode == "group":
            for k in range(len(GROUPS)):
                if (g == k).sum() >= 50:
                    mu[k], sd[k] = np.nanmean(X[g == k], 0), np.nanstd(X[g == k], 0)
        sd[~(sd > 0)] = 1
        return mu, sd

    def promoted_prior(self, season, cfg: "Config"):
        """Promoted teams start at the mean team effect of the relegated teams (fit as of season start)."""
        key = (season, cfg.key())
        if key in self._promoted_prior:
            return self._promoted_prior[key]
        k = SEASONS.index(season)
        out = {}
        if k > 0:
            m = self.matches
            prev = set(m.loc[m.season == SEASONS[k - 1], "home_team_id"])
            cur = set(m.loc[m.season == season, "home_team_id"])
            start = self.season_starts[season]
            th = fit(self, start, cfg, prior={})[0]
            ta, td = th[N_DENSE:N_DENSE + self.T], th[N_DENSE + self.T:N_DENSE + 2 * self.T]
            rel = [self.tix[t] for t in prev - cur]
            out = {t: (float(ta[rel].mean()), float(td[rel].mean())) for t in cur - prev}
        self._promoted_prior[key] = out
        return out


def naive(s: pd.Series) -> pd.Series:
    return s.dt.tz_convert(None) if getattr(s.dt, "tz", None) is not None else s


def extra_columns(mf: pd.DataFrame, spec: str) -> pd.DataFrame:
    """Columns for one extra term. Specs:
      styleint:<v>   style clash, every product s_A,i * s_B,j of the attacking and defending teams' scores (v = full,
                     geometry or none: the embedding variant), the full-rank form of s_A' U V' s_B (§9.1)
      stylemain:<v>  the attacking team's scores (own attack) and the defending team's (its defence)
      cols:a,b,...   matchup feature columns as they are in matchup_features.parquet"""
    kind, arg = spec.split(":", 1)
    if kind in ("styleint", "stylemain"):
        sa = mf[[c for c in mf.columns if c.startswith(f"{arg}_s") and not c.startswith("opp_")]]
        sb = mf[[f"opp_{c}" for c in sa.columns]]
        if kind == "stylemain":
            return pd.concat([sa.add_prefix("A_"), sb.add_prefix("B_")], axis=1).fillna(0.0)
        out = {f"{arg}_x{i+1}{j+1}": sa.iloc[:, i].fillna(0) * sb.iloc[:, j].fillna(0)
               for i in range(sa.shape[1]) for j in range(sb.shape[1])}
        return pd.DataFrame(out, index=mf.index)
    if kind == "cols":
        return mf[arg.split(",")].astype(float)
    raise ValueError(spec)


def ts(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_convert(None) if t.tzinfo is not None else t


def expected_shares(pt: pd.DataFrame) -> pd.Series:
    """f_i: a starter's expected share of the match, from his earlier starts only (decayed, shrunk)."""
    s = pt[pt.started].sort_values("date")
    out = np.empty(len(s))
    for _, g in s.groupby("player_id", sort=False):
        d = (g.date - pd.Timestamp("2000-01-01")).dt.total_seconds().to_numpy() / 86400
        v = g.share.to_numpy()
        prior = F_PRIOR_GK if g.is_gk.iloc[0] else F_PRIOR_OUT
        res = np.empty(len(g))
        for j in range(len(g)):
            w = 0.5 ** ((d[j] - d[:j][d[:j] < d[j]]) / F_HL)
            vv = v[:j][d[:j] < d[j]]
            res[j] = (np.sum(w * vv) + F_K * prior) / (np.sum(w) + F_K)
        out[s.index.get_indexer(g.index)] = res
    return pd.Series(out, index=pd.MultiIndex.from_frame(s[["match_id", "player_id"]]))


# --------------------------------------------------------------------------- fitting
def design(data: StintData, rows: np.ndarray, zs, EX=None):
    mu, sd = zs
    if np.allclose(mu, mu[0]) and np.allclose(sd, sd[0]):          # pooled standardization
        ZA = (data.SxA_t[rows] - data.NfA_t[rows] * mu[0]) / sd[0]
        ZD = (data.SxD_t[rows] - data.NfD_t[rows] * mu[0]) / sd[0]
    else:
        ZA = sum((data.SxA[g][rows] - data.NfA[g][rows] * mu[g]) / sd[g] for g in range(NG))
        ZD = sum((data.SxD[g][rows] - data.NfD[g][rows] * mu[g]) / sd[g] for g in range(NG))
    X = sp.hstack([sp.csr_matrix(data.D[rows]), data.TA[rows], -data.TD[rows], data.PA[rows], -data.PD[rows],
                   sp.csr_matrix(ZA), sp.csr_matrix(-ZD)]
                  + ([sp.csr_matrix(EX[rows])] if EX is not None else [])).tocsr()
    return X


def penalties(data: StintData, cfg: Config, prior: dict):
    T, P, F = data.T, data.P, data.F
    pen = np.r_[1e-8, 1e-8, np.full(len(STATES), 5.0), 2.0, 2.0,
                np.full(2 * T, cfg.lam_tau),
                np.full(2 * P, 1e8 if (np.isinf(cfg.lam_u) or not cfg.use_players) else cfg.lam_u),
                np.full(2 * F, 1e8 if (np.isinf(cfg.lam_b) or not cfg.use_players) else cfg.lam_b)]
    if cfg.feats == "compact":
        off = np.array([f not in COMPACT for f in FEATS] * 2)
        pen[-2 * F:][off] = 1e8
    for spec, lam in cfg.extra:
        M, _ = data.extra(Config(extra=((spec, lam),)))
        pen = np.r_[pen, np.full(M.shape[1], lam)]
    c0 = np.zeros_like(pen)
    for t, (a0, d0) in prior.items():
        c0[N_DENSE + data.tix[t]] = a0
        c0[N_DENSE + T + data.tix[t]] = d0
    return pen, c0


def fit(data: StintData, as_of, cfg: Config, prior: dict | None = None, theta0=None):
    st = data.st
    as_of = ts(as_of)
    rows = np.flatnonzero((st.date < as_of).to_numpy())
    zs = data.zstats(as_of, cfg.zmode)
    EX, _ = data.extra(cfg)
    X = design(data, rows, zs, EX)
    s = st.iloc[rows]
    age = (as_of - s.date).dt.total_seconds().to_numpy() / 86400
    w = np.ones_like(age) if np.isinf(cfg.half_life) else 0.5 ** (age / cfg.half_life)
    cur = data.season_idx[st.loc[st.date >= as_of, "season"].iloc[0]] if (st.date >= as_of).any() else s.season_idx.max()
    w = w * cfg.carry ** np.clip(cur - s.season_idx.to_numpy(), 0, None)
    e, y = s.e.to_numpy(), s.npxg.to_numpy()
    we, wy = w * e, w * y
    if prior is None:
        season = st.loc[st.date >= as_of, "season"].iloc[0] if (st.date >= as_of).any() else SEASONS[-1]
        prior = data.promoted_prior(season, cfg)
    pen, c0 = penalties(data, cfg, prior)
    XT = X.T.tocsr()

    def f(th):
        eta = X @ th
        lam = np.exp(eta)
        loss = np.sum(we * lam - wy * eta) + np.sum(pen * (th - c0) ** 2)
        g = XT @ (we * lam - wy) + 2 * pen * (th - c0)
        return loss, g

    if theta0 is None or len(theta0) != len(c0):
        theta0 = c0.copy()
        theta0[0] = np.log(wy.sum() / we.sum())
    res = minimize(f, theta0, jac=True, method="L-BFGS-B", options=dict(maxiter=3000, maxfun=6000))
    th = res.x
    # weighted average state / man-count effect over the training stints
    gs = data.D[rows][:, 2:N_DENSE] @ th[2:N_DENSE]
    G = float(np.log(np.sum(we * np.exp(gs)) / we.sum()))
    # expected match length in 90s (stoppage time included), from the previous 365 days
    recent = s[s.date >= as_of - pd.Timedelta(days=365)]
    L90 = float(recent.groupby("match_id").duration.sum().mean() / 2 / 90) if len(recent) else 1.0
    return th, dict(zs=zs, G=G, L90=L90, nit=res.nit, success=res.success, as_of=as_of, rows=rows, cfg=cfg)


def unpack(data: StintData, th):
    T, P, F = data.T, data.P, data.F
    o = N_DENSE
    return dict(mu=th[0], home=th[1], g=th[2:6], rho=th[6:8], ta=th[o:o + T], td=th[o + T:o + 2 * T],
                u=th[o + 2 * T:o + 2 * T + P], v=th[o + 2 * T + P:o + 2 * T + 2 * P],
                bo=th[o + 2 * T + 2 * P:o + 2 * T + 2 * P + F], bd=th[o + 2 * T + 2 * P + F:o + 2 * T + 2 * P + 2 * F],
                extra=th[o + 2 * T + 2 * P + 2 * F:])


# --------------------------------------------------------------------------- player effects
def player_effects(data: StintData, th, zs, match_ids, player_ids):
    """o_i and d_i of each (match, player) pair with the profile as of that match."""
    q = unpack(data, th)
    mu, sd = zs
    k = pd.MultiIndex.from_arrays([match_ids, player_ids])
    pr = data.prof.reindex(k)
    X = pr[FEATS].to_numpy(float)
    g = pr.group.map({g: j for j, g in enumerate(GROUPS)}).fillna(len(GROUPS)).astype(int).to_numpy()
    Z = np.nan_to_num((X - mu[g]) / sd[g])
    pc = pd.Series(np.arange(data.P), index=data.players).reindex(player_ids).to_numpy()
    has = ~np.isnan(pc)
    u = np.zeros(len(pc))
    v = np.zeros(len(pc))
    u[has] = q["u"][pc[has].astype(int)]
    v[has] = q["v"][pc[has].astype(int)]
    return Z @ q["bo"] + u, Z @ q["bd"] + v, Z @ q["bo"], Z @ q["bd"]


def sub_averages(data: StintData, th, zs, as_of, window_days=365):
    """Minutes-weighted average o and d of each team's substitute appearances in the window before as_of."""
    pt = data.pt
    s = pt[(~pt.started) & (pt.date < as_of) & (pt.date >= as_of - pd.Timedelta(days=window_days)) & (pt.share > 0)]
    o, d, _, _ = player_effects(data, th, zs, s.match_id.to_numpy(), s.player_id.to_numpy())
    s = s.assign(o=o * s.share, d=d * s.share)
    g = s.groupby("team_id")[["o", "d", "share"]].sum()
    avg = (g.o / g.share, g.d / g.share)
    league = (s.o.sum() / s.share.sum(), s.d.sum() / s.share.sum()) if len(s) else (0.0, 0.0)
    return avg, league


def predict_matches(data: StintData, th, info, target: pd.DataFrame) -> pd.DataFrame:
    """Pre-match npxG rate of each side from its starting XI (§9.3)."""
    q = unpack(data, th)
    zs = info["zs"]
    cfg = info.get("cfg")
    E = data.extra(cfg)[1] if (cfg is not None and cfg.extra) else None
    (sub_o, sub_d), (lo, ld) = sub_averages(data, th, zs, info["as_of"])
    pt = data.pt
    rows = []
    for mt in target.itertuples():
        side = {}
        for team in (mt.home_team_id, mt.away_team_id):
            xi = data.starters[mt.match_id]
            xi = xi[xi.team_id == team]
            o, d, xo, xd = player_effects(data, th, zs, xi.match_id.to_numpy(), xi.player_id.to_numpy())
            f = data.fexp.reindex(pd.MultiIndex.from_frame(xi[["match_id", "player_id"]])).fillna(F_PRIOR_OUT).to_numpy()
            f = np.where(xi.is_gk.to_numpy(), 1.0, f)
            so, sd_ = sub_o.get(team, lo), sub_d.get(team, ld)
            side[team] = dict(O=np.sum(f * o) + np.sum(1 - f) * so, D=np.sum(f * d) + np.sum(1 - f) * sd_,
                              n=len(xi))
        for att, dfn, home in ((mt.home_team_id, mt.away_team_id, 1), (mt.away_team_id, mt.home_team_id, 0)):
            eta = (q["mu"] + q["home"] * home + info["G"] + np.log(info["L90"])
                   + q["ta"][data.tix[att]] - q["td"][data.tix[dfn]]
                   + side[att]["O"] - side[dfn]["D"])
            xterm = 0.0
            if E is not None and (mt.match_id, att) in E:
                xterm = float(E[(mt.match_id, att)] @ q["extra"])
            eta = eta + xterm
            rows.append(dict(match_id=mt.match_id, team_id=att, home=home,
                             npxg_pred=float(np.exp(eta)) * info.get("scale", 1.0),
                             O=side[att]["O"], D_opp=side[dfn]["D"], ta=q["ta"][data.tix[att]],
                             td_opp=q["td"][data.tix[dfn]], n_xi=side[att]["n"], xterm=xterm))
    return pd.DataFrame(rows)


def level_scale(data: StintData, th, info, window_days: int = 120) -> float:
    """Level calibration, as of the fit date: actual npxG over predicted npxG for the matches of the
    previous window_days, each re-predicted pre-match with the current fit (training data only)."""
    as_of = info["as_of"]
    m = data.matches[(data.matches.date < as_of) & (data.matches.date >= as_of - pd.Timedelta(days=window_days))]
    if len(m) < 20:
        return 1.0
    p = predict_matches(data, th, dict(info, scale=1.0), m)
    act = data.st.groupby(["match_id", "team_id"]).npxg.sum()
    a = act.reindex(pd.MultiIndex.from_frame(p[["match_id", "team_id"]])).to_numpy()
    return float(a.sum() / p.npxg_pred.sum())


def refit_dates(dates: pd.Series, min_gap_days: int = 0) -> dict:
    """Map each match date to the as-of date of the fit used for it (refit at least every min_gap days)."""
    days = np.sort(dates.dt.floor("D").unique())
    out, last = {}, None
    for d in days:
        if last is None or (d - last) / np.timedelta64(1, "D") >= max(min_gap_days, 1e-9):
            last = d
        out[d] = last
    return out


def rolling(data: StintData, cfg: Config, seasons, min_gap_days: int = 0, verbose=False) -> pd.DataFrame:
    """As-of predictions for every match of the given seasons."""
    m = data.matches[data.matches.season.isin(seasons)]
    fitmap = refit_dates(m.date, min_gap_days)
    out, theta, cur_fit = [], None, None
    for day, g in m.groupby(m.date.dt.floor("D")):
        fd = fitmap[np.datetime64(day)]
        if cur_fit is None or cur_fit[0] != fd:
            as_of = g.date.min() if pd.Timestamp(fd) == day else pd.Timestamp(fd)
            th, info = fit(data, as_of, cfg, theta0=theta)
            info["scale"] = level_scale(data, th, info)
            theta = th
            cur_fit = (fd, th, info)
            if verbose:
                print(cfg.key(), str(as_of)[:10], info["nit"], flush=True)
        out.append(predict_matches(data, cur_fit[1], cur_fit[2], g).assign(fit_as_of=cur_fit[2]["as_of"]))
    return pd.concat(out, ignore_index=True)
