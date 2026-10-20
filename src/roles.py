"""Role discovery (methodology §4.10): soft roles for outfield players from what they do.

Run:  python src/roles.py        (needs data/processed/player_profile_asof.parquet from src/profiles.py)

1. Inputs. One row per player-season with at least MIN_MINUTES league minutes: his profile as of his last
   matchday squad of the season (shrunk, recency-weighted, possession- and opponent-adjusted, §4.2-4.9).
   ROLE_FEATURES: 23 rates and shares plus 6 passing-network features (depth relative to the team,
   distance from the centre line, betweenness, share of team passes, progression passed and received).
   Left and right are merged (distance from the centre line); side is kept separately (y_side).
   Count rates enter as square roots (variance-stabilising for counts), then every feature is z-scored.
2. Model. PCA to N_PC components, then a Gaussian mixture with one covariance shared by all roles ("tied"):
   roles differ in their average profile, and the spread around it has the same shape for every role.
   Full covariances per role over-fit with 300-900 player-seasons (BIC picks 4-5 roles); tied ones put the
   BIC minimum in the expected 8-12 range.
3. Number of roles K: BIC on the fit from 2023/24-2024/25 (the fit used for 2025/26), plus stability on the
   two halves of 2024/25 (tuning season), plus interpretability. The halves of 2025/26 are reported only.
4. As of date. Refit at the start of 2024/25, 2025/26 and 2026/27 from earlier player-seasons only, with
   standardization and PCA from the same rows (§2.3). Each refit's roles are matched to the previous
   fit's by Hungarian assignment on role means, so a role keeps its number and name. 2023/24 has no roles.
5. Soft memberships pi_{i,r} for every outfield squad row (2024/25 on) = posterior role probabilities of the
   player's as-of profile under that season's fit.

Outputs:
  data/processed/player_roles_asof.parquet     match_id, player_id, role_1..role_K, top_role, top_prob
  reports/role_bic.csv, reports/role_stability.csv, reports/role_means.csv, reports/role_examples.csv,
  reports/role_vs_slot.csv, reports/role_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

import profiles as PR

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
SEASONS = PR.SEASONS

ROLE_FEATURES = ["npxg", "np_shots", "headed_shots", "box_touches", "xa", "chances_created", "dribble_attempts",
                 "final_third_passes", "passes", "touches", "crosses", "long_balls", "tackles", "interceptions",
                 "blocks", "recoveries", "clearances", "aerial_duels", "ground_duels", "was_fouled", "pass_acc",
                 "aerial_win", "passes_received", "x_rel", "y_abs", "betweenness", "pass_share", "prog_out",
                 "prog_in"]
SQRT = [f for f in ROLE_FEATURES if PR.FEATURES[f][0] == "count"]
MIN_MINUTES = 900
MIN_MINUTES_HALF = 450
N_PC = 8
K_GRID = list(range(6, 15))
K_CHOSEN = 10   # see choose_k notes in the memo: BIC flat from 10 to 12, best tuning-half stability among 9-12
N_INIT = 10

# Proposed names (pending Benjamin's approval at the Step 2 checkpoint), keyed by aligned role number.
ROLE_NAMES: dict[int, str] = {
    1: "Target striker", 2: "Full-back", 3: "Creative attacking midfielder", 4: "Ball-winning midfielder",
    5: "Ball-playing centre-back", 6: "Mobile striker", 7: "Winger / inside forward", 8: "Stopper centre-back",
    9: "Deep playmaker", 10: "Box-to-box midfielder"}


# --------------------------------------------------------------------------- inputs
def transform(df: pd.DataFrame) -> pd.DataFrame:
    X = df[ROLE_FEATURES].astype(float).copy()
    for c in SQRT:
        X[c] = np.sqrt(X[c].clip(lower=0))
    return X


def player_seasons(prof: pd.DataFrame, pm: pd.DataFrame) -> pd.DataFrame:
    mins = pm[pm.played & ~pm.is_gk].groupby(["season", "player_id"]).minutes_played.sum().rename("season_minutes")
    last = prof.sort_values("date").groupby(["season", "player_id"]).tail(1).set_index(["season", "player_id"])
    last = last.join(mins, how="inner")
    return last[last.season_minutes >= MIN_MINUTES].reset_index()


def half_season_rows(season: str, pm: pd.DataFrame, nf: pd.DataFrame) -> pd.DataFrame:
    """Unshrunk per-90 profiles from each half of a season alone (for the stability check)."""
    p = pm[pm.played & ~pm.is_gk & (pm.season == season)].copy()
    p["np_shots"] = p.total_shots - p.penalties_taken
    p = p.merge(nf.drop(columns="team_id"), on=["match_id", "player_id"], how="left")
    p["half"] = np.where(p["round"] <= 19, 1, 2)
    p["e"] = p.minutes_played / 90
    out = []
    for (h, pid), g in p.groupby(["half", "player_id"]):
        if g.minutes_played.sum() < MIN_MINUTES_HALF:
            continue
        r = dict(half=h, player_id=pid, player_name=g.player_name.iloc[-1])
        E = g.e.sum()
        for f in ROLE_FEATURES:
            kind, num, den, *_ = PR.FEATURES[f]
            if kind == "count":
                r[f] = g[num].sum() / E
            elif kind == "prop":
                r[f] = g[num].sum() / max(g[den].sum(), 1)
            else:
                ok = g[num].notna()
                r[f] = (g.loc[ok, num] * g.loc[ok, "e"]).sum() / max(g.loc[ok, "e"].sum(), 1e-9)
        out.append(r)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- model
class RoleModel:
    def __init__(self, K: int, seed: int = 0):
        self.K, self.seed = K, seed

    def fit(self, X: pd.DataFrame, init_means_raw: pd.DataFrame | None = None):
        """init_means_raw: role means of the previous fit (natural units). When given, EM starts from them
        (warm start), so each role keeps its meaning from one refit to the next; role r stays role r."""
        self.mu, self.sd = X.mean(), X.std(ddof=0)
        Z = ((X - self.mu) / self.sd).to_numpy()
        self.pca = PCA(N_PC, random_state=self.seed).fit(Z)
        Zp = self.pca.transform(Z)
        if init_means_raw is None:
            self.gmm = GaussianMixture(self.K, covariance_type="tied", n_init=N_INIT, random_state=self.seed).fit(Zp)
        else:
            m0 = self.pca.transform(((transform(init_means_raw) - self.mu) / self.sd).to_numpy())
            self.gmm = GaussianMixture(self.K, covariance_type="tied", means_init=m0,
                                       random_state=self.seed).fit(Zp)
        self.bic = self.gmm.bic(Zp)
        self.perm = np.arange(self.K)        # aligned role r = component perm[r]
        return self

    def _zp(self, X):
        return self.pca.transform(((X - self.mu) / self.sd).to_numpy())

    def proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.gmm.predict_proba(self._zp(X))[:, self.perm]

    def means_z(self) -> np.ndarray:
        """Role means in standardized feature units (K x features), aligned order."""
        return self.pca.inverse_transform(self.gmm.means_)[self.perm]

    def means_raw(self) -> pd.DataFrame:
        m = self.means_z() * self.sd.to_numpy() + self.mu.to_numpy()
        df = pd.DataFrame(m, columns=self.mu.index)
        for c in SQRT:
            df[c] = df[c].clip(lower=0) ** 2
        return df


def align(new: RoleModel, ref_means_raw: pd.DataFrame, scale: pd.Series):
    """Order new's roles to match a reference set of role means (Hungarian on standardized distance)."""
    a = transform(new.means_raw()).to_numpy() / scale.to_numpy()
    b = transform(ref_means_raw).to_numpy() / scale.to_numpy()
    cost = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    rows, cols = linear_sum_assignment(cost)
    perm = np.empty(new.K, int)
    perm[cols] = rows
    new.perm = new.perm[perm]
    return float(np.sqrt(cost[rows, cols]).mean())


def stability(Xa: pd.DataFrame, Xb: pd.DataFrame, ids_a, ids_b, K: int) -> dict:
    """Fit on each half separately; match roles by Hungarian on means; report how similar the matched role
    means are and how often players seen in both halves keep their (matched) role."""
    ma, mb = RoleModel(K).fit(Xa), RoleModel(K).fit(Xb)
    scale = pd.concat([Xa, Xb]).std(ddof=0)
    dist = align(mb, ma.means_raw(), scale)
    center = pd.concat([Xa, Xb]).mean()
    za = (transform(ma.means_raw()) - center) / scale
    zb = (transform(mb.means_raw()) - center) / scale
    corr = np.mean([np.corrcoef(za.iloc[k], zb.iloc[k])[0, 1] for k in range(K)])
    common = np.intersect1d(ids_a, ids_b)
    la = pd.Series(ma.proba(Xa).argmax(1), index=ids_a).loc[common]
    lb = pd.Series(mb.proba(Xb).argmax(1), index=ids_b).loc[common]
    return dict(K=K, mean_matched_role_corr=corr, mean_matched_dist=dist,
                same_role_share=float((la.to_numpy() == lb.to_numpy()).mean()),
                adjusted_rand=adjusted_rand_score(la, lb), n_players_both_halves=len(common))


# --------------------------------------------------------------------------- main
def main(K: int | None = None):
    prof = pd.read_parquet(PROC / "player_profile_asof.parquet")
    pm = pd.read_parquet(PROC / "player_match.parquet")
    nf = pd.read_parquet(PROC / "player_match_network.parquet")
    ps = player_seasons(prof, pm)

    # ---- choose K: BIC on 2023/24-2024/25 player-seasons, stability on the halves of 2024/25
    base = ps[ps.season.isin(SEASONS[:2])]
    Xb = transform(base)
    bic = []
    for k in K_GRID:
        bic.append(dict(K=k, bic=RoleModel(k).fit(Xb).bic))
    bic = pd.DataFrame(bic)
    stab = []
    for season in ["2024/2025", "2025/2026"]:
        hs = half_season_rows(season, pm, nf)
        a, b = hs[hs.half == 1], hs[hs.half == 2]
        for k in K_GRID:
            r = stability(transform(a), transform(b), a.player_id.to_numpy(), b.player_id.to_numpy(), k)
            stab.append(dict(season=season, **r))
    stab = pd.DataFrame(stab)
    bic.to_csv(REPORTS / "role_bic.csv", index=False)
    stab.to_csv(REPORTS / "role_stability.csv", index=False)
    if K is None:
        K = K_CHOSEN

    # ---- as-of fits, aligned
    fits = {}
    drift, kept = {}, {}
    prev = None
    scale_all = transform(ps).std(ddof=0)
    for s_idx in range(1, len(SEASONS)):
        season = SEASONS[s_idx]
        train = ps[ps.season.isin(SEASONS[:s_idx])]
        m = RoleModel(K).fit(transform(train), None if prev is None else prev.means_raw())
        if prev is not None:
            # warm start keeps the order; the Hungarian check only confirms it (drift is reported)
            drift[season] = align(m, prev.means_raw(), scale_all)
            kept[season] = bool((m.perm == np.arange(K)).all())
        fits[season] = m
        prev = m

    # ---- memberships for every outfield squad row from 2024/25 on
    rows = prof[prof.season.isin(SEASONS[1:])].copy()
    P = np.full((len(rows), K), np.nan)
    for season, m in fits.items():
        sel = (rows.season == season).to_numpy()
        P[sel] = m.proba(transform(rows[sel]))
    roles = rows[["match_id", "player_id", "team_id", "season", "date", "player_name", "group"]].copy()
    for r in range(K):
        roles[f"role_{r + 1}"] = P[:, r]
    roles["top_role"] = P.argmax(1) + 1
    roles["top_prob"] = P.max(1)
    roles.to_parquet(PROC / "player_roles_asof.parquet", index=False)

    # ---- reports
    means = []
    for season, m in fits.items():
        mr = m.means_raw()
        mr.insert(0, "role", np.arange(1, K + 1))
        mr.insert(0, "fit_for_season", season)
        mr["weight"] = m.gmm.weights_[m.perm]
        means.append(mr)
    means = pd.concat(means)
    means.to_csv(REPORTS / "role_means.csv", index=False)

    latest = fits[SEASONS[-1]]
    train = ps[ps.season.isin(SEASONS[:-1])].copy()
    pr = latest.proba(transform(train))
    train["role"] = pr.argmax(1) + 1
    train["prob"] = pr.max(1)
    ex = (train.sort_values("season_minutes", ascending=False).groupby("role")
          .head(10)[["role", "season", "player_name", "group", "season_minutes", "prob"]].sort_values(["role", "season_minutes"],
                                                                                                    ascending=[True, False]))
    ex.to_csv(REPORTS / "role_examples.csv", index=False)

    # role vs the slot actually filled, test seasons (starters only)
    slots = pd.read_parquet(PROC / "player_slots.parquet")[["match_id", "player_id", "slot"]]
    rs = roles.merge(slots, on=["match_id", "player_id"])
    rs = rs[rs.season.isin(PR.TEST)]
    ct = pd.crosstab(rs.slot.map(PR.SLOT_GROUP), rs.top_role, normalize="columns")
    ct.to_csv(REPORTS / "role_vs_slot.csv")

    # how persistent roles are: same top role in consecutive seasons (player-season fits under the latest model)
    tr = train.pivot_table(index="player_id", columns="season", values="role", aggfunc="first")
    persist = {}
    for a, b in zip(SEASONS[:2], SEASONS[1:3]):
        both = tr[[a, b]].dropna()
        persist[f"{a}->{b}"] = dict(n=len(both), same_role=float((both[a] == both[b]).mean()))

    summary = dict(
        K=K, n_pc=N_PC, covariance="tied", min_minutes=MIN_MINUTES,
        player_seasons={s: int((ps.season == s).sum()) for s in SEASONS},
        bic_best_K=int(bic.loc[bic.bic.idxmin(), "K"]),
        pca_variance_kept={s: float(m.pca.explained_variance_ratio_.sum()) for s, m in fits.items()},
        stability_tuning=stab[(stab.season == "2024/2025") & (stab.K == K)].iloc[0].drop("season").to_dict(),
        stability_test=stab[(stab.season == "2025/2026") & (stab.K == K)].iloc[0].drop("season").to_dict(),
        season_to_season_same_role=persist,
        refit_mean_shift_sd=drift, refit_order_kept=kept,
        membership_rows=int(len(roles)),
        median_top_prob_test=float(roles[roles.season.isin(PR.TEST)].top_prob.median()),
        share_rows_top_prob_below_0_6_test=float((roles[roles.season.isin(PR.TEST)].top_prob < 0.6).mean()),
        role_names=ROLE_NAMES,
    )
    (REPORTS / "role_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))
    return fits, ps, roles


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
