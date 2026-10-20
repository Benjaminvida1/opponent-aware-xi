"""Team style: own style vs style imposed on opponents (methodology §5.5) and the style embedding (§6).

Run:  python src/style.py            (uses cached tuning results if present)
      python src/style.py --grid     (re-runs the tuning grid on 2024/25)

Needs data/processed/team_match_network.parquet from src/network.py.

1. As-of team ratings. The L0 model (chosen settings) is refit before every match date; each team's
   rating is att + def (log npxG scale). Cached in data/processed/team_ratings_asof.parquet.

2. Decomposition, per metric z and team-match (A vs B):
       z = mu + alpha_A + beta_B + gamma*home + eta_lead*share_leading + eta_trail*share_trailing
           + kappa*(rating_A - rating_B) [+ pi*poss for the PPDA proxy only]  + e
   alpha_A = A's own style, beta_B = what B imposes on its opponents. Fitted as weighted ridge
   regression (ridge = random effects with a fixed variance ratio), match weights
   0.5**(age/half_life) * carry**(seasons back), refit before every match date from earlier matches
   only. Matches with a red card before minute 60 are left out of fitting (§2.4).
   Half-life, carry and the two ridge strengths are chosen per metric on 2024/25 by one-step-ahead
   prediction error of the metric itself; the test seasons are only reported.
   Prediction for an upcoming match uses alpha_A + beta_B + home + rating difference, with game state
   set to its average (unknown before kick-off) and, for PPDA, possession set to its own prediction.

3. Embedding. At the start of each season, PCA on the standardized own-style effects (alpha) of every
   team-season before it (end-of-season fits), a non-redundant subset of metrics. Components are
   sign/order-aligned with the previous refit and projected onto each team's as-of alphas.

Outputs:
  data/processed/team_ratings_asof.parquet, data/processed/team_style_asof.parquet
  reports/style_grid.csv, reports/style_settings.csv, reports/style_prediction.csv,
  reports/style_embedding_loadings.csv, reports/style_embedding_variance.csv, reports/style_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import network as N
import team_strength as ts

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

STYLE_METRICS = [m for m in N.METRICS] + ["back_line_in_possession"]
LABELS = dict(N.METRICS, back_line_in_possession="Players in back line, in possession")
# Embedding inputs: drop exact or near duplicates (length = att - def; width_range ~ width; tempo = passes/poss)
# and the sided measures (left/right asymmetries), which are kept for matchups, not as style dimensions.
EMBED_METRICS = ["centroid_x", "def_line_height", "att_height", "width", "compactness", "gap_back_mid",
                 "gap_mid_front", "back_line_in_possession", "passes", "entropy", "gini_strength", "reciprocity",
                 "clustering", "verticality", "forward_share", "pass_span", "flank_share", "gk_share",
                 "max_betweenness", "alg_connectivity", "poss", "log_ppda", "box_entries", "cross_tendency",
                 "long_ball_share"]
HALF_LIVES = [30.0, 60.0, 120.0, 240.0, 365.0, 540.0, 730.0, np.inf]
CARRIES = [1.0, 0.75, 0.5]
RIDGES = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0]   # in units of full-weight team-matches
COVS = ["home", "share_leading", "share_trailing", "rating_diff"]
SEASONS = ["2023/2024", "2024/2025", "2025/2026", "2026/2027"]
TUNE, TEST = ["2024/2025"], ["2025/2026", "2026/2027"]
# Names agreed with Benjamin at the Step 1 checkpoint (2026-10-18); high end first. Sign convention is fixed
# by the alignment in fit_pca, so a higher score always means more of the first-named pole.
STYLE_NAMES = {1: "Control vs direct", 2: "Central vs wide", 3: "Forward reach", 4: "Build-up through the keeper"}
EMBED_MIN_R2 = 0.03   # tuning-season own-style R^2 below this = no detectable team signal (cf. §4.5)


# --------------------------------------------------------------------------- 1. as-of ratings
def asof_ratings(force: bool = False) -> pd.DataFrame:
    path = PROC / "team_ratings_asof.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)
    matches, sides = ts.load()
    kw = {k: v for k, v in json.loads((REPORTS / "l0_chosen.json").read_text())["npxg"].items()}
    seasons = sorted(matches.season.unique())
    priors = {s: ts.season_start_prior(sides, matches, kw, s) for s in seasons}
    rows = []
    for day, g in matches.groupby(matches.date.dt.floor("D")):
        as_of = g.date.min()
        season = g.season.iloc[0]
        if (sides.date < as_of).sum() == 0:
            for t in set(g.home_team_id) | set(g.away_team_id):
                rows.append(dict(day=day, team_id=t, rating=0.0))
            continue
        m = ts.StrengthModel(**kw).fit(sides, as_of, priors[season], seasons.index(season))
        for t in set(g.home_team_id) | set(g.away_team_id):
            rows.append(dict(day=day, team_id=t,
                             rating=m.params["att"].get(t, 0.0) + m.params["defence"].get(t, 0.0)))
    out = pd.DataFrame(rows)
    out.to_parquet(path, index=False)
    return out


# --------------------------------------------------------------------------- 2. decomposition
def load_table() -> pd.DataFrame:
    tm = pd.read_parquet(PROC / "team_match_network.parquet")
    tm["day"] = tm.date.dt.floor("D")
    r = asof_ratings()
    tm = tm.merge(r, on=["day", "team_id"], how="left")
    tm = tm.merge(r.rename(columns={"team_id": "opponent_id", "rating": "opp_rating"}), on=["day", "opponent_id"], how="left")
    tm["rating_diff"] = tm.rating - tm.opp_rating
    tm["season_idx"] = tm.season.map({s: k for k, s in enumerate(SEASONS)})
    return tm.sort_values(["date", "match_id", "team_id"]).reset_index(drop=True)


class Design:
    """Fixed design matrix: covariates, team (alpha) and opponent (beta) indicators."""

    def __init__(self, tm: pd.DataFrame):
        self.tm = tm
        self.teams = sorted(set(tm.team_id) | set(tm.opponent_id))
        self.tidx = {t: k for k, t in enumerate(self.teams)}
        n, T = len(tm), len(self.teams)
        self.n_cov = 1 + len(COVS) + 1   # intercept, covariates, poss (used for PPDA only)
        X = np.zeros((n, self.n_cov + 2 * T))
        X[:, 0] = 1
        for k, c in enumerate(COVS):
            X[:, 1 + k] = tm[c].to_numpy(float)
        X[:, self.n_cov - 1] = tm.poss.to_numpy(float)
        X[np.arange(n), self.n_cov + tm.team_id.map(self.tidx).to_numpy()] = 1
        X[np.arange(n), self.n_cov + T + tm.opponent_id.map(self.tidx).to_numpy()] = 1
        self.X = X
        self.T = T
        self.fit_ok = ~tm.red_before_60.to_numpy()
        self.age_days = None

    def weights(self, as_of: pd.Timestamp, season_idx: int, hl: float, carry: float) -> np.ndarray:
        tm = self.tm
        mask = (tm.date < as_of).to_numpy() & self.fit_ok
        age = (as_of - tm.date).dt.total_seconds().to_numpy() / 86400
        w = np.ones(len(tm)) if np.isinf(hl) else 0.5 ** (age / hl)
        w = w * carry ** np.clip(season_idx - tm.season_idx.to_numpy(), 0, None)
        return np.where(mask, w, 0.0)

    def solve(self, XtWX, XtWz, lam_a, lam_b, use_poss: np.ndarray):
        """Ridge solve for every metric column of XtWz; poss column dropped (huge penalty) unless used."""
        p = XtWX.shape[0]
        out = np.empty((p, XtWz.shape[1]))
        base = np.zeros(p)
        base[1:self.n_cov] = 1e-6
        base[self.n_cov:self.n_cov + self.T] = lam_a
        base[self.n_cov + self.T:] = lam_b
        for flag in (False, True):
            cols = np.where(use_poss == flag)[0]
            if len(cols) == 0:
                continue
            P = base.copy()
            if not flag:
                P[self.n_cov - 1] = 1e12
            out[:, cols] = np.linalg.solve(XtWX + np.diag(P), XtWz[:, cols])
        return out

    def predict_rows(self, theta, rows, gbar, poss_hat):
        """Pre-match prediction: game state at its average, possession from its own prediction (PPDA only)."""
        X = self.X[rows].copy()
        X[:, 2] = gbar[0]
        X[:, 3] = gbar[1]
        X[:, self.n_cov - 1] = poss_hat
        return X @ theta


def rolling(design: Design, Z: np.ndarray, use_poss: np.ndarray, settings: list[tuple], seasons: list[str],
            keep_params: bool = False, poss_col: int | None = None):
    """For each prediction day in `seasons`, fit as of that day and predict that day's team-matches.

    settings: list of (hl, carry, lam_a, lam_b). Returns predictions array (n_settings, n_rows, n_metrics)
    for the target rows, the target row index, and optionally the fitted parameters per day (first setting)."""
    tm = design.tm
    target = np.where(tm.season.isin(seasons).to_numpy())[0]
    preds = np.full((len(settings), len(target), Z.shape[1]), np.nan)
    pos = {r: k for k, r in enumerate(target)}
    params = {}
    by_w = {}
    for k, (hl, carry, la, lb) in enumerate(settings):
        by_w.setdefault((hl, carry), []).append((k, la, lb))
    Zf = np.nan_to_num(Z)
    days = tm.loc[target].groupby("day").groups
    for day, idx in days.items():
        rows = np.asarray(idx)
        as_of = tm.loc[rows, "date"].min()
        sidx = int(tm.loc[rows[0], "season_idx"])
        for (hl, carry), lst in by_w.items():
            w = design.weights(as_of, sidx, hl, carry)
            if w.sum() == 0:
                continue
            Xw = design.X * w[:, None]
            XtWX = Xw.T @ design.X
            XtWz = Xw.T @ Zf
            g = design.tm[["share_leading", "share_trailing"]].to_numpy()
            gbar = (w[:, None] * g).sum(0) / w.sum()
            for k, la, lb in lst:
                theta = design.solve(XtWX, XtWz, la, lb, use_poss)
                poss_hat = design.predict_rows(theta[:, [poss_col]], rows, gbar, 0.5)[:, 0] if poss_col is not None else 0.5
                pr = design.predict_rows(theta, rows, gbar, poss_hat)
                preds[k, [pos[r] for r in rows]] = pr
                if keep_params and k == 0:
                    params[day] = (theta, gbar)
    return preds, target, params


def tuning_grid(design: Design, Z: np.ndarray, use_poss: np.ndarray, metrics: list[str], poss_col: int) -> pd.DataFrame:
    settings = [(hl, c, la, lb) for hl in HALF_LIVES for c in CARRIES for la in RIDGES for lb in RIDGES]
    preds, target, _ = rolling(design, Z, use_poss, settings, TUNE, poss_col=poss_col)
    y = Z[target]
    rows = []
    var = np.nanvar(y, 0)
    for k, (hl, c, la, lb) in enumerate(settings):
        mse = np.nanmean((preds[k] - y) ** 2, 0)
        for j, m in enumerate(metrics):
            rows.append(dict(metric=m, half_life_days=hl, carry=c, ridge_own=la, ridge_imposed=lb,
                             mse=mse[j], r2_oos=1 - mse[j] / var[j]))
    return pd.DataFrame(rows)


def choose_settings(grid: pd.DataFrame) -> pd.DataFrame:
    """Per metric: lowest tuning-season MSE. Ties within 0.5% go to the longer half-life and stronger
    ridge (the smoother, more conservative fit)."""
    out = []
    for m, g in grid.groupby("metric", sort=False):
        best = g.mse.min()
        tied = g[g.mse <= best * 1.005].copy()
        tied["hl_sort"] = tied.half_life_days.replace(np.inf, 1e9)
        pick = tied.sort_values(["hl_sort", "ridge_own", "ridge_imposed", "carry"], ascending=[False, False, False, False]).iloc[0]
        out.append(pick.drop("hl_sort").to_dict() | dict(n_tied=len(tied)))
    return pd.DataFrame(out)


def run_metric_settings(design, Z, use_poss, metrics, chosen: pd.DataFrame, seasons, poss_col, variant="full"):
    """Rolling predictions with each metric's own chosen settings. variant: 'full'; 'no_imposed' (no opponent
    style term, rating difference kept); 'own_only' (no opponent term, no rating difference); or 'league'
    (intercept, home, game state only)."""
    tm = design.tm
    target = np.where(tm.season.isin(seasons).to_numpy())[0]
    out = np.full((len(target), len(metrics)), np.nan)
    params = {}
    groups = chosen.groupby(["half_life_days", "carry", "ridge_own", "ridge_imposed"]).metric.apply(list)
    poss_m = metrics[poss_col]
    for (hl, c, la, lb), ms in groups.items():
        if variant in ("own_only", "no_imposed"):
            lb = 1e12
        if variant == "league":
            la = lb = 1e12
        cols = [metrics.index(m) for m in ms]
        need_poss = any(use_poss[cols])
        if need_poss and poss_m not in ms:
            cols = cols + [poss_col]
        sub_use = use_poss[cols]
        pcol = cols.index(poss_col) if need_poss else None
        Zs = Z[:, cols].copy()
        pr, tgt, prm = rolling(_variant(design, variant), Zs, sub_use, [(hl, c, la, lb)], seasons,
                               keep_params=True, poss_col=pcol)
        for j, col in enumerate(cols):
            if metrics[col] in ms:
                out[:, col] = pr[0][:, j]
        for day, (theta, gbar) in prm.items():
            params.setdefault(day, {})
            for j, col in enumerate(cols):
                if metrics[col] in ms:
                    params[day][metrics[col]] = theta[:, j]
    return out, target, params


def _variant(design: Design, variant: str) -> Design:
    if variant == "full":
        return design
    d = Design.__new__(Design)
    d.__dict__.update(design.__dict__)
    X = design.X.copy()
    if variant in ("own_only", "league"):
        X[:, 1 + COVS.index("rating_diff")] = 0.0
    d.X = X
    return d


# --------------------------------------------------------------------------- 3. embedding
def end_of_season_alphas(design, Z, use_poss, metrics, chosen, season: str, poss_col: int) -> pd.DataFrame:
    """Alphas fitted as of the day after the season's last match, for that season's teams."""
    tm = design.tm
    last = tm.loc[tm.season == season, "date"].max() + pd.Timedelta(days=1)
    sidx = SEASONS.index(season)
    rows = {}
    for (hl, c, la, lb), ms in chosen.groupby(["half_life_days", "carry", "ridge_own", "ridge_imposed"]).metric.apply(list).items():
        cols = [metrics.index(m) for m in ms]
        w = design.weights(last, sidx, hl, c)
        Xw = design.X * w[:, None]
        theta = design.solve(Xw.T @ design.X, Xw.T @ np.nan_to_num(Z[:, cols]), la, lb, use_poss[cols])
        for j, m in enumerate(ms):
            rows[m] = theta[design.n_cov:design.n_cov + design.T, j]
    teams = sorted(set(tm.loc[tm.season == season, "team_id"]))
    df = pd.DataFrame(rows, index=design.teams).loc[teams]
    return df.assign(season=season)


def fit_pca(A: pd.DataFrame, prev: dict | None, n_comp: int, cols: list[str]):
    X = A[cols].to_numpy(float)
    mu, sd = X.mean(0), X.std(0, ddof=1)
    Xs = (X - mu) / sd
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    var = S ** 2 / (S ** 2).sum()
    L = Vt[:n_comp].T                       # metrics x components
    if prev is not None:                    # align order and sign with the previous season's loadings
        from scipy.optimize import linear_sum_assignment
        C = prev["L"].T @ Vt[: n_comp + 2].T  # prev comps x candidate comps
        r, c = linear_sum_assignment(-np.abs(C))
        L = np.column_stack([Vt[j] * np.sign(C[i, j]) for i, j in zip(r, c)])
        var = var[: n_comp + 2][c]
    else:                                   # convention: largest loading positive
        L = L * np.sign(L[np.abs(L).argmax(0), np.arange(n_comp)])
        var = var[:n_comp]
    return dict(mu=mu, sd=sd, L=L, var=var, all_var=(S ** 2 / (S ** 2).sum()))


# --------------------------------------------------------------------------- main
def main(run_grid: bool):
    tm = load_table()
    metrics = STYLE_METRICS
    Z = tm[metrics].to_numpy(float)
    use_poss = np.array([m == "log_ppda" for m in metrics])
    poss_col = metrics.index("poss")
    design = Design(tm)

    grid_path = REPORTS / "style_grid.csv"
    if run_grid or not grid_path.exists():
        grid = tuning_grid(design, Z, use_poss, metrics, poss_col)
        grid.to_csv(grid_path, index=False)
    grid = pd.read_csv(grid_path)
    chosen = choose_settings(grid)
    chosen.to_csv(REPORTS / "style_settings.csv", index=False)

    # Predictive check on tuning and test seasons: league average < own style < + strength gap < + imposed style.
    res = {}
    allseasons = TUNE + TEST
    for variant in ["full", "no_imposed", "own_only", "league"]:
        pr, target, params = run_metric_settings(design, Z, use_poss, metrics, chosen, SEASONS[0:1] + allseasons,
                                                 poss_col, variant)
        res[variant] = (pr, target, params)
    pr_full, target, params_full = res["full"]
    y = Z[target]
    seas = tm.season.to_numpy()[target]
    rows = []
    rng = np.random.default_rng(0)
    mids = tm.match_id.to_numpy()[target]

    def boot_ci(d, sel):
        """Paired bootstrap over matches (both sides of a match resampled together), 2,000 resamples."""
        codes, inv = np.unique(mids[sel], return_inverse=True)
        per_match = np.bincount(inv, d, len(codes))
        cnt = np.bincount(inv, None, len(codes))
        idx = rng.integers(0, len(codes), (2000, len(codes)))
        boots = per_match[idx].sum(1) / cnt[idx].sum(1)
        return np.quantile(boots, 0.025), np.quantile(boots, 0.975)

    for split, sel_seasons in [("tuning", TUNE), ("test", TEST)]:
        sel = np.isin(seas, sel_seasons)
        for j, m in enumerate(metrics):
            yy = y[sel, j]
            e = {v: (res[v][0][sel, j] - yy) ** 2 for v in res}
            var = np.var(yy)
            d_opp = (e["own_only"] - e["full"]) / var        # >0: knowing the opponent (strength + style) helps
            d_imp = (e["no_imposed"] - e["full"]) / var      # >0: the opponent's imposed style helps beyond strength
            lo, hi = boot_ci(d_opp, sel)
            ilo, ihi = boot_ci(d_imp, sel)
            rows.append(dict(metric=m, label=LABELS[m], split=split, n=int(sel.sum()),
                             r2_league=1 - e["league"].mean() / var, r2_own_only=1 - e["own_only"].mean() / var,
                             r2_with_strength=1 - e["no_imposed"].mean() / var, r2_full=1 - e["full"].mean() / var,
                             opp_gain=d_opp.mean(), opp_gain_lo=lo, opp_gain_hi=hi,
                             imposed_gain=d_imp.mean(), imposed_gain_lo=ilo, imposed_gain_hi=ihi))
    pred_tab = pd.DataFrame(rows)
    pred_tab.to_csv(REPORTS / "style_prediction.csv", index=False)

    # As-of style table: alpha (own) and beta (imposed) for both teams of every match from 2023/24 R2 on.
    T = design.T
    style_rows = []
    for day, prm in params_full.items():
        day_rows = tm[tm.day == day]
        for r in day_rows.itertuples():
            rec = dict(match_id=r.match_id, team_id=r.team_id, opponent_id=r.opponent_id, date=r.date, season=r.season)
            ti, oi = design.tidx[r.team_id], design.tidx[r.opponent_id]
            for m, th in prm.items():
                rec[f"own_{m}"] = th[design.n_cov + ti]
                rec[f"imposed_{m}"] = th[design.n_cov + T + oi]       # what this team's opponent imposes on it
                rec[f"imposes_{m}"] = th[design.n_cov + T + ti]       # what this team imposes on opponents
            style_rows.append(rec)
    style = pd.DataFrame(style_rows)

    # Embedding inputs: candidate metrics with detectable team signal on the tuning season
    # (own-style-only out-of-sample R^2 >= EMBED_MIN_R2); the rest are kept as metrics but not embedded.
    tune_r2 = pred_tab[pred_tab.split == "tuning"].set_index("metric").r2_own_only
    embed = [m for m in EMBED_METRICS if tune_r2[m] >= EMBED_MIN_R2]
    dropped = [m for m in EMBED_METRICS if m not in embed]

    # Embedding: refit at each season start from all earlier seasons' end-of-season alphas.
    alphas = {s: end_of_season_alphas(design, Z, use_poss, metrics, chosen, s, poss_col) for s in SEASONS[:-1]}
    n_comp = None
    prev = None
    load_rows, var_rows, emb = [], [], []
    for k, s in enumerate(SEASONS[1:], start=1):
        A = pd.concat([alphas[p] for p in SEASONS[:k]])
        if n_comp is None:  # chosen on the first fit (2023/24 only) is too thin; fix on the 2025/26 fit below
            pass
        fit_all = fit_pca(A, None, min(len(embed), len(A) - 1), embed)
        cum = np.cumsum(fit_all["all_var"])
        if s == "2025/2026":
            n_comp = int(np.clip(np.searchsorted(cum, 0.70) + 1, 4, 6))
        var_rows.append(dict(season_fit=s, n_team_seasons=len(A), **{f"cum_var_{i+1}": cum[i] for i in range(8)}))
    for k, s in enumerate(SEASONS[1:], start=1):
        A = pd.concat([alphas[p] for p in SEASONS[:k]])
        fit = fit_pca(A, prev, n_comp, embed)
        prev = fit
        for i in range(n_comp):
            for j, m in enumerate(embed):
                load_rows.append(dict(season_fit=s, component=i + 1, metric=m, label=LABELS[m], loading=fit["L"][j, i]))
        st = style[style.season == s]
        Xo = (st[[f"own_{m}" for m in embed]].to_numpy() - fit["mu"]) / fit["sd"]
        sc = Xo @ fit["L"]
        e = st[["match_id", "team_id"]].copy()
        for i in range(n_comp):
            e[f"style_{i+1}"] = sc[:, i]
        emb.append(e)
    style = style.merge(pd.concat(emb), on=["match_id", "team_id"], how="left")
    style.to_parquet(PROC / "team_style_asof.parquet", index=False)
    pd.DataFrame(load_rows).to_csv(REPORTS / "style_embedding_loadings.csv", index=False)
    pd.DataFrame(var_rows).to_csv(REPORTS / "style_embedding_variance.csv", index=False)

    # Season-to-season stability of end-of-season embedding scores (last fit's loadings).
    fit = prev
    stab = {}
    sc = {s: pd.DataFrame(((alphas[s][embed].to_numpy() - fit["mu"]) / fit["sd"]) @ fit["L"], index=alphas[s].index)
          for s in SEASONS[:-1]}
    for a, b in zip(SEASONS[:2], SEASONS[1:3]):
        common = sc[a].index.intersection(sc[b].index)
        stab[f"{a}->{b}"] = [float(np.corrcoef(sc[a].loc[common, i], sc[b].loc[common, i])[0, 1]) for i in range(n_comp)]

    summary = dict(
        n_team_matches=int(len(tm)), excluded_red_before_60=int(tm.red_before_60.sum()),
        n_components=n_comp, style_names=STYLE_NAMES, embedding_metrics=embed, embedding_metrics_dropped_no_signal=dropped,
        variance_explained_last_fit=[float(v) for v in fit["var"]],
        embedding_stability=stab,
        settings_summary=chosen[["metric", "half_life_days", "carry", "ridge_own", "ridge_imposed", "r2_oos"]].to_dict("records"),
        prediction_test_mean=pred_tab[pred_tab.split == "test"][["r2_league", "r2_own_only", "r2_with_strength", "r2_full", "opp_gain", "imposed_gain"]].mean().to_dict(),
        metrics_opp_gain_ci_above_zero_test=pred_tab[(pred_tab.split == "test") & (pred_tab.opp_gain_lo > 0)].metric.tolist(),
        metrics_imposed_gain_ci_above_zero_test=pred_tab[(pred_tab.split == "test") & (pred_tab.imposed_gain_lo > 0)].metric.tolist(),
        metrics_imposed_gain_ci_below_zero_test=pred_tab[(pred_tab.split == "test") & (pred_tab.imposed_gain_hi < 0)].metric.tolist(),
    )
    (REPORTS / "style_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true")
    main(ap.parse_args().grid)
