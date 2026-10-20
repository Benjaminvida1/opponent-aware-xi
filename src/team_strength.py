"""Team strength (methodology §7), expected points (§10.1) and the rolling-origin harness (§15.1).

Model, per directed team-match side A -> B in match m:

    log E[npxG_{A->B,m}] = mu + h * home_{A,m} + att_A - def_B

fitted by weighted quasi-Poisson likelihood with match weights

    w_m = 0.5 ** (age_days / half_life) * carry ** (seasons back)

and a ridge penalty pulling att/def toward a prior centre (0 = league average for established
teams; the mean rating of the relegated sides for promoted teams). The season carry-over factor
(carry <= 1) down-weights earlier seasons beyond the smooth time decay, which shrinks carried-over
ratings toward the prior at each summer break (methodology §7.1, omega).

Everything is refit "as of" each prediction date: a match on date d is predicted only
from matches played strictly before d.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MAX_GOALS = 10


# --------------------------------------------------------------------------- data
def team_match_sides(matches: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    """Two rows per match (one per attacking side) with npxG, open-play/set-piece xG, goals."""
    s = shots[shots.is_own_goal == 0]
    agg = s.groupby(["match_id", "team_id"]).agg(
        npxg=("xg", lambda v: v[s.loc[v.index, "situation_group"] != "penalty"].sum()),
        xg_open_play=("xg", lambda v: v[s.loc[v.index, "situation_group"] == "open_play"].sum()),
        xg_set_piece=("xg", lambda v: v[s.loc[v.index, "situation_group"] == "set_piece"].sum()),
        pen_xg=("xg", lambda v: v[s.loc[v.index, "situation_group"] == "penalty"].sum()),
        shots=("shot_id", "size"),
    )
    rows = []
    for r in matches.itertuples():
        for side, team, tid, opp, oid, gf, ga in [
            (1, r.home_team, r.home_team_id, r.away_team, r.away_team_id, r.home_score, r.away_score),
            (0, r.away_team, r.away_team_id, r.home_team, r.home_team_id, r.away_score, r.home_score),
        ]:
            rows.append(dict(match_id=r.match_id, season=r.season, round=r.round, date=r.date, home=side,
                             team=team, team_id=tid, opponent=opp, opponent_id=oid, goals=gf, goals_against=ga))
    df = pd.DataFrame(rows)
    order = {s: k for k, s in enumerate(sorted(matches.season.unique()))}
    df["season_idx"] = df.season.map(order)
    df = df.join(agg, on=["match_id", "team_id"])
    for c in ["npxg", "xg_open_play", "xg_set_piece", "pen_xg", "shots"]:
        df[c] = df[c].fillna(0.0)
    return df


# --------------------------------------------------------------------------- model
@dataclass
class StrengthModel:
    half_life_days: float = 120.0      # xi = ln 2 / half-life; np.inf = no decay
    ridge: float = 2.0                 # penalty weight on (att - prior)^2 + (def - prior)^2
    response: str = "npxg"             # "npxg" or "goals"
    carry: float = 1.0                 # extra weight per season back (1 = time decay only)
    teams: list = field(default_factory=list)
    params: dict = field(default_factory=dict)

    def fit(self, sides: pd.DataFrame, as_of: pd.Timestamp, prior: dict[int, tuple[float, float]] | None = None,
            season_idx: int | None = None):
        d = sides[sides.date < as_of]
        self.teams = sorted(set(d.team_id) | set(d.opponent_id) | set((prior or {}).keys()))
        idx = {t: k for k, t in enumerate(self.teams)}
        n = len(self.teams)
        ti = d.team_id.map(idx).to_numpy()
        oi = d.opponent_id.map(idx).to_numpy()
        home = d.home.to_numpy(float)
        y = d[self.response].to_numpy(float)
        age = (as_of - d.date).dt.total_seconds().to_numpy() / 86400
        w = np.ones_like(age) if np.isinf(self.half_life_days) else 0.5 ** (age / self.half_life_days)
        if self.carry < 1 and season_idx is not None and "season_idx" in d:
            w = w * self.carry ** np.clip(season_idx - d.season_idx.to_numpy(), 0, None)
        pa = np.zeros(n)
        pd_ = np.zeros(n)
        for t, (a0, d0) in (prior or {}).items():
            pa[idx[t]], pd_[idx[t]] = a0, d0
        lam = self.ridge

        def nll(theta):
            mu, h = theta[0], theta[1]
            att, dfn = theta[2:2 + n], theta[2 + n:]
            eta = mu + h * home + att[ti] - dfn[oi]
            rate = np.exp(eta)
            loss = np.sum(w * (rate - y * eta)) + lam * (np.sum((att - pa) ** 2) + np.sum((dfn - pd_) ** 2))
            r = w * (rate - y)
            g = np.empty_like(theta)
            g[0] = r.sum()
            g[1] = (r * home).sum()
            g[2:2 + n] = np.bincount(ti, r, n) + 2 * lam * (att - pa)
            g[2 + n:] = -np.bincount(oi, r, n) + 2 * lam * (dfn - pd_)
            return loss, g

        theta0 = np.zeros(2 + 2 * n)
        theta0[0] = np.log(max(np.average(y, weights=w), 1e-3)) if len(y) else np.log(1.2)
        res = minimize(nll, theta0, jac=True, method="L-BFGS-B")
        th = res.x
        self.params = dict(mu=th[0], home=th[1], att=dict(zip(self.teams, th[2:2 + n])),
                           defence=dict(zip(self.teams, th[2 + n:])), n_obs=len(y), weight_sum=w.sum())
        return self

    def rates(self, team_ids, opp_ids, home: int) -> np.ndarray:
        p = self.params
        att = pd.Series(p["att"]).reindex(np.asarray(team_ids)).fillna(0).to_numpy()
        dfn = pd.Series(p["defence"]).reindex(np.asarray(opp_ids)).fillna(0).to_numpy()
        return np.exp(p["mu"] + p["home"] * home + att - dfn)

    def rate(self, team_id: int, opp_id: int, home: int) -> float:
        p = self.params
        return float(np.exp(p["mu"] + p["home"] * home + p["att"].get(team_id, 0.0) - p["defence"].get(opp_id, 0.0)))


# --------------------------------------------------------------------------- outcomes
def score_matrix(lam_home: float, lam_away: float, rho: float = 0.0) -> np.ndarray:
    """Joint scoreline probabilities; rho is the Dixon–Coles low-score correction (rho < 0 adds draws)."""
    g = np.arange(MAX_GOALS + 1)
    m = np.outer(poisson.pmf(g, lam_home), poisson.pmf(g, lam_away))
    if rho:
        m[0, 0] *= 1 - lam_home * lam_away * rho
        m[0, 1] *= 1 + lam_home * rho
        m[1, 0] *= 1 + lam_away * rho
        m[1, 1] *= 1 - rho
    return m / m.sum()


def outcome_probs(lam_home: float, lam_away: float, rho: float = 0.0) -> tuple[float, float, float]:
    m = score_matrix(lam_home, lam_away, rho)
    return float(np.tril(m, -1).sum()), float(np.trace(m)), float(np.triu(m, 1).sum())


def fit_rho(lh: np.ndarray, la: np.ndarray, hg: np.ndarray, ag: np.ndarray, w: np.ndarray) -> float:
    """Weighted ML estimate of the Dixon–Coles rho from past scorelines and fitted goal rates."""
    from scipy.optimize import minimize_scalar
    low = (hg <= 1) & (ag <= 1)
    lh, la, hg, ag, w = lh[low], la[low], hg[low], ag[low], w[low]

    def nll(rho):
        tau = np.where((hg == 0) & (ag == 0), 1 - lh * la * rho,
              np.where((hg == 0) & (ag == 1), 1 + lh * rho,
              np.where((hg == 1) & (ag == 0), 1 + la * rho, 1 - rho)))
        return -np.sum(w * np.log(np.clip(tau, 1e-9, None)))
    # Only the tau terms depend on rho (the normalising constant is ~1 for these ranges).
    return float(minimize_scalar(nll, bounds=(-0.3, 0.1), method="bounded").x)


def expected_points(p_win: float, p_draw: float) -> float:
    return 3 * p_win + p_draw


def rps(probs: np.ndarray, outcome_idx: np.ndarray) -> np.ndarray:
    """Ranked probability score for ordered outcomes (home win, draw, away win). Lower is better."""
    obs = np.zeros_like(probs)
    obs[np.arange(len(probs)), outcome_idx] = 1
    cp, co = np.cumsum(probs, 1)[:, :-1], np.cumsum(obs, 1)[:, :-1]
    return ((cp - co) ** 2).sum(1) / (probs.shape[1] - 1)


def log_loss(probs: np.ndarray, outcome_idx: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(probs[np.arange(len(probs)), outcome_idx], 1e-12, 1))


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    y = np.asarray(y, float)
    mu = np.asarray(mu, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return 2 * (term - (y - mu))


# --------------------------------------------------------------------------- harness
def season_start_prior(sides: pd.DataFrame, matches: pd.DataFrame, model_kw: dict, season: str) -> dict:
    """Promoted teams start at the mean rating of the teams relegated at the end of the previous season."""
    seasons = sorted(matches.season.unique())
    k = seasons.index(season)
    if k == 0:
        return {}
    prev, cur = seasons[k - 1], season
    prev_teams = set(matches.loc[matches.season == prev, "home_team_id"])
    cur_teams = set(matches.loc[matches.season == cur, "home_team_id"])
    promoted, relegated = cur_teams - prev_teams, prev_teams - cur_teams
    start = matches.loc[matches.season == cur, "date"].min()
    m = StrengthModel(**model_kw).fit(sides, start, season_idx=k - 1)
    a0 = float(np.mean([m.params["att"][t] for t in relegated]))
    d0 = float(np.mean([m.params["defence"][t] for t in relegated]))
    return {t: (a0, d0) for t in promoted}


def rolling_predictions(matches: pd.DataFrame, sides: pd.DataFrame, model_kw: dict,
                        eval_mask: pd.Series | None = None, dixon_coles: bool = True) -> pd.DataFrame:
    """Refit as of each match date and predict that date's matches. Returns one row per match.

    Goals ~ Poisson(c * (predicted npxG + league penalty xG per team-match)), with c and the
    Dixon–Coles rho both estimated from matches before the prediction date.
    """
    seasons = sorted(matches.season.unique())
    target = matches if eval_mask is None else matches[eval_mask]
    promoted_prior = {s: season_start_prior(sides, matches, model_kw, s) for s in target.season.unique()}
    out = []
    for date, day in target.groupby(target.date.dt.floor("D")):
        as_of = day.date.min()
        season = day.season.iloc[0]
        m = StrengthModel(**model_kw).fit(sides, as_of, promoted_prior.get(season), seasons.index(season))
        past = sides[sides.date < as_of]
        pen_rate = past.pen_xg.mean() if len(past) else 0.09
        c = (past.goals.sum() / (past.npxg + past.pen_xg).sum()) if len(past) else 1.0
        rho = 0.0
        if dixon_coles and len(past):
            pm_ = matches[matches.date < as_of]
            fh = c * (m.rates(pm_.home_team_id, pm_.away_team_id, 1) + pen_rate)
            fa = c * (m.rates(pm_.away_team_id, pm_.home_team_id, 0) + pen_rate)
            age = (as_of - pm_.date).dt.total_seconds().to_numpy() / 86400
            hl = model_kw.get("half_life_days", np.inf)
            w = np.ones_like(age) if np.isinf(hl) else 0.5 ** (age / hl)
            rho = fit_rho(fh, fa, pm_.home_score.to_numpy(), pm_.away_score.to_numpy(), w)
        for r in day.itertuples():
            lh = m.rate(r.home_team_id, r.away_team_id, 1)
            la = m.rate(r.away_team_id, r.home_team_id, 0)
            gh, ga = c * (lh + pen_rate), c * (la + pen_rate)
            ph, pdraw, pa = outcome_probs(gh, ga, rho)
            out.append(dict(match_id=r.match_id, date=r.date, season=r.season, round=r.round,
                            home_team=r.home_team, away_team=r.away_team,
                            npxg_home_pred=lh, npxg_away_pred=la, goals_home_pred=gh, goals_away_pred=ga,
                            p_home=ph, p_draw=pdraw, p_away=pa,
                            ep_home=expected_points(ph, pdraw), ep_away=expected_points(pa, pdraw),
                            calib_c=c, dc_rho=rho, home_adv=m.params["home"]))
    pred = pd.DataFrame(out)
    obs = sides[sides.home == 1][["match_id", "npxg", "goals", "goals_against"]].rename(
        columns={"npxg": "npxg_home", "goals": "home_score", "goals_against": "away_score"})
    obs_a = sides[sides.home == 0][["match_id", "npxg"]].rename(columns={"npxg": "npxg_away"})
    pred = pred.merge(obs, on="match_id").merge(obs_a, on="match_id")
    pred["outcome"] = np.select([pred.home_score > pred.away_score, pred.home_score == pred.away_score], [0, 1], 2)
    return pred


def base_rate_predictions(matches: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    """Climatology benchmark: past home/draw/away frequencies as of each date."""
    res = np.select([matches.home_score > matches.away_score, matches.home_score == matches.away_score], [0, 1], 2)
    mm = matches.assign(outcome=res)
    out = []
    for r in target.itertuples():
        past = mm[mm.date < r.date]
        f = np.bincount(past.outcome, minlength=3) + 1  # add-one smoothing
        f = f / f.sum()
        out.append(dict(match_id=r.match_id, p_home=f[0], p_draw=f[1], p_away=f[2]))
    return pd.DataFrame(out)


def score(pred: pd.DataFrame) -> dict:
    P = pred[["p_home", "p_draw", "p_away"]].to_numpy()
    o = pred.outcome.to_numpy()
    out = dict(n=len(pred), rps=rps(P, o).mean(), log_loss=log_loss(P, o).mean())
    if "npxg_home_pred" in pred:
        dev = np.r_[poisson_deviance(pred.npxg_home, pred.npxg_home_pred),
                    poisson_deviance(pred.npxg_away, pred.npxg_away_pred)]
        out["npxg_deviance"] = dev.mean()
    return out


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of (a - b) with a 95% interval, resampling matches."""
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    boots = d[idx].mean(1)
    return float(d.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    matches = pd.read_parquet(PROC / "matches.parquet")
    shots = pd.read_parquet(PROC / "shots.parquet")
    return matches, team_match_sides(matches, shots)
