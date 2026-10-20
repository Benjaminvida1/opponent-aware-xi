"""Step 4: opponent-aware features for the outcome model, one row per directed team-match (A attacking B),
all as of the match date (methodology §5.7, §6, §9.1).

Style clash (L2). The four style scores of both teams (team_style_asof). 2023/24 has no embedding of its
own; its rows are scored with the embedding fitted at the start of 2024/25 (loadings and standardization
from 2023/24 end-of-season own styles). These rows only ever enter fits dated 2024/25 or later, so every
feature still uses only matches played before the date it is used for (§2.3).

Matchups (L3), from the two starting XIs' predicted positions (netpred.py) and as-of profiles (Step 2):
  zone_wide, zone_half, zone_centre   attacking load of A minus defensive cover of B (mirrored) in the
                                      attacking third, Gaussian kernel of bandwidth b; cover divided by B's
                                      lane vulnerability (share of npxG B concedes from that lane vs the
                                      league, shrunk) when `vuln` is on
  dribble                             A's five most advanced vs B's five deepest outfielders, paired by
                                      Hungarian on distance: sum z(dribbles_i) + z(dribbled past_j)
  aerial                              same pairs: sum z(aerial win rate_i) - z(aerial win rate_j)
  press                               z(press_A) * z(build-up exposure_B); press = -own PPDA proxy effect;
                                      exposure = z(GK involvement) + z(CB betweenness) - z(verticality)
Player quality for the zones (strictly positive): q_att = exp(composite / 2), composite = mean z of npxG, xA
and the two progression measures; q_def likewise from tackles, interceptions, blocks, clearances. z-scores
use the means and SDs of outfield profile rows before the date. Each output feature is finally z-scored
with the mean and SD of the rows of the previous 365 days, so coefficients are on a common scale.
Lane vulnerability factors are clipped to [0.5, 2].

Network ablation (§15.3) variants of the same features:
  full      as above
  geometry  style embedding refit without the structure metrics; press exposure without GK involvement
            and betweenness (verticality kept: it comes from positions)
  none      style embedding refit on the §6 team box-score features only; no positions: zone features 0,
            duel pairs = the five most attacking (FW, W, AM groups by q_att) vs the five most defensive
            (CB, FB groups by q_def) without distances; press exposure from long-ball share only

Run:  python src/matchups.py
Output: data/processed/matchup_features.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
SEASONS = ["2023/2024", "2024/2025", "2025/2026", "2026/2027"]
PITCH_X, PITCH_Y = 105.0, 68.0
ZX = np.array([17.5, 52.5, 87.5])
ZY = np.array([6.8, 20.4, 34.0, 47.6, 61.2])
BANDWIDTHS = [6.0, 10.0, 15.0]
VULN_K = 3.0                          # pseudo-npxG pulling a team's lane shares toward the league's
CB_SLOTS = {"LCB", "CB", "RCB"}
GEOMETRY = ["centroid_x", "def_line_height", "att_height", "width", "compactness", "gap_back_mid", "gap_mid_front",
            "back_line_in_possession"]
TEAMFEATS = ["poss", "log_ppda", "box_entries", "cross_tendency", "long_ball_share"]
MATCHUP = ["zone_wide", "zone_half", "zone_centre", "dribble", "aerial", "press"]


def naive(s):
    return s.dt.tz_convert(None) if getattr(s.dt, "tz", None) is not None else s


# --------------------------------------------------------------------------- style scores (incl. ablation embeddings)
def style_scores():
    """As-of style scores for every team-match: the Step 1 embedding ('full') and two refits on subsets of
    metrics ('geometry': no structure metrics; 'none': team box-score features only)."""
    tm = S.load_table()
    metrics = S.STYLE_METRICS
    Z = tm[metrics].to_numpy(float)
    use_poss = np.array([m == "log_ppda" for m in metrics])
    poss_col = metrics.index("poss")
    design = S.Design(tm)
    chosen = pd.read_csv(REPORTS / "style_settings.csv")
    embed_full = __import__("json").loads((REPORTS / "style_summary.json").read_text())["embedding_metrics"]
    sets = {"full": (embed_full, 4),
            "geometry": ([m for m in embed_full if m in GEOMETRY + TEAMFEATS], 4),
            "none": ([m for m in embed_full if m in TEAMFEATS], 3)}
    alphas = {s: S.end_of_season_alphas(design, Z, use_poss, metrics, chosen, s, poss_col) for s in SEASONS[:-1]}
    asof = pd.read_parquet(PROC / "team_style_asof.parquet")
    out = asof[["match_id", "team_id", "season"]].copy()
    for name, (cols, k) in sets.items():
        prev = None
        parts = []
        for j, s in enumerate(SEASONS[1:], start=1):
            A = pd.concat([alphas[p] for p in SEASONS[:j]])
            fit = S.fit_pca(A, prev, k, cols)
            prev = fit
            targets = [s] if s != "2024/2025" else ["2023/2024", "2024/2025"]   # 2023/24 scored with the first fit
            st = asof[asof.season.isin(targets)]
            sc = ((st[[f"own_{m}" for m in cols]].to_numpy() - fit["mu"]) / fit["sd"]) @ fit["L"]
            e = st[["match_id", "team_id"]].copy()
            for i in range(k):
                e[f"{name}_s{i+1}"] = sc[:, i]
            parts.append(e)
        out = out.merge(pd.concat(parts), on=["match_id", "team_id"], how="left")
    if "style_1" in asof:   # the 'full' refit must reproduce Step 1's scores from 2024/25 on
        chk = out.merge(asof[["match_id", "team_id", "style_1"]], on=["match_id", "team_id"]).dropna()
        assert np.allclose(chk.full_s1, chk.style_1, atol=1e-6), "full embedding does not reproduce Step 1"
    return out.drop(columns="season")


# --------------------------------------------------------------------------- lane vulnerability from shots conceded
def lane_vulnerability(m):
    """f[B, lane] as of each match: share of npxG B concedes from the lane over the league's share, shrunk."""
    sh = pd.read_parquet(PROC / "shots.parquet")
    sh = sh[(sh.situation_group != "Penalty") & ~sh.is_own_goal.astype(bool)].copy()
    sh = sh.drop(columns=["date"]).merge(m[["match_id", "date", "home_team_id", "away_team_id"]], on="match_id")
    sh["conceding"] = np.where(sh.team_id == sh.home_team_id, sh.away_team_id, sh.home_team_id)
    sh["lane"] = np.clip((sh.y_m // 13.6).astype(int), 0, 4)
    rows = []
    hist = sh.sort_values("date")
    for day, g in m.groupby(m.date.dt.floor("D")):
        past = hist[hist.date < g.date.min()]
        if len(past) < 200:
            continue
        age = (g.date.min() - past.date).dt.total_seconds() / 86400
        w = past.xg * 0.5 ** (age / 365.0)
        lg = w.groupby(past.lane).sum().reindex(range(5), fill_value=0)
        lg = lg / lg.sum()
        tb = w.groupby([past.conceding, past.lane]).sum().unstack(fill_value=0).reindex(columns=range(5), fill_value=0)
        for t in set(g.home_team_id) | set(g.away_team_id):
            c = tb.loc[t].to_numpy() if t in tb.index else np.zeros(5)
            share = (c + VULN_K * lg.to_numpy()) / (c.sum() + VULN_K)
            rows.append(dict(day=day, team_id=t, **{f"vuln_{k}": float(np.clip(share[k] / lg.iloc[k], 0.5, 2.0))
                                                    for k in range(5)}))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- matchup features
def kernel(xy, b):
    cx, cy = np.meshgrid(ZX, ZY, indexing="ij")          # 3 x 5 zones
    d2 = (xy[:, 0, None, None] - cx) ** 2 + (xy[:, 1, None, None] - cy) ** 2
    K = np.exp(-d2 / (2 * b * b))
    return K / K.sum((1, 2), keepdims=True)              # players x 3 x 5


def zstats_by_day(prof, cols):
    """Mean and SD of each column over outfield profile rows before each match day."""
    days = np.sort(prof.date.dt.floor("D").unique())
    X = prof[cols].to_numpy(float)
    d = prof.date.to_numpy()
    out = {}
    for day in days:
        keep = d < day
        if keep.sum() < 200:
            continue
        out[pd.Timestamp(day)] = (np.nanmean(X[keep], 0), np.nanstd(X[keep], 0))
    return out


def build():
    m = pd.read_parquet(PROC / "matches.parquet")
    m["date"] = naive(m.date)
    pos = pd.read_parquet(PROC / "pred_positions.parquet")
    prof = pd.read_parquet(PROC / "player_profile_asof.parquet")
    prof["date"] = naive(prof.date)
    sty = pd.read_parquet(PROC / "team_style_asof.parquet")
    pcols = ["npxg", "xa", "prog_out", "prog_in", "tackles", "interceptions", "blocks", "clearances", "dribbles",
             "dribbled_past", "aerial_win", "betweenness"]
    zs = zstats_by_day(prof, pcols)
    pr = prof.set_index(["match_id", "player_id"])[pcols + ["group"]]
    pr = pr[~pr.index.duplicated()]
    pos = pos.join(pr, on=["match_id", "player_id"])
    pos = pos.merge(m[["match_id", "date"]], on="match_id")
    vuln = lane_vulnerability(m).set_index(["day", "team_id"])
    sty_k = sty.set_index(["match_id", "team_id"])
    rows = []
    for mt in m.itertuples():
        day = mt.date.floor("D")
        if day not in zs:
            continue
        mu, sd = zs[day]
        sd = np.where(sd > 0, sd, 1)
        P = pos[pos.match_id == mt.match_id]
        sides = {}
        for t in (mt.home_team_id, mt.away_team_id):
            g = P[(P.team_id == t) & ~P.is_gk]
            Zp = np.clip(np.nan_to_num((g[pcols].to_numpy(float) - mu) / sd), -5, 5)   # a few early profiles are extreme
            z = dict(zip(pcols, Zp.T))
            qa = np.exp((z["npxg"] + z["xa"] + z["prog_out"] + z["prog_in"]) / 4 / 2)
            qd = np.exp((z["tackles"] + z["interceptions"] + z["blocks"] + z["clearances"]) / 4 / 2)
            sides[t] = dict(xy=g[["x_pred", "y_pred"]].to_numpy(float), qa=qa, qd=qd, z=z, grp=g.group.to_numpy(),
                            cb_btw=float(np.nanmean(np.where(g.slot_pred.isin(CB_SLOTS), g.betweenness, np.nan)))
                            if g.slot_pred.isin(CB_SLOTS).any() else np.nan)
        for A, B in ((mt.home_team_id, mt.away_team_id), (mt.away_team_id, mt.home_team_id)):
            a, b = sides[A], sides[B]
            if len(a["xy"]) < 10 or len(b["xy"]) < 10:
                continue
            bm = np.column_stack([PITCH_X - b["xy"][:, 0], PITCH_Y - b["xy"][:, 1]])
            rec = dict(match_id=mt.match_id, team_id=A, opponent_id=B, date=mt.date)
            vk = (day, B)
            f = vuln.loc[vk].to_numpy(float) if vk in vuln.index else np.ones(5)
            for bw in BANDWIDTHS:
                Ka, Kb = kernel(a["xy"], bw), kernel(bm, bw)
                att = (Ka * a["qa"][:, None, None]).sum(0)[2]     # attacking third, 5 lanes
                dfn = (Kb * b["qd"][:, None, None]).sum(0)[2]
                for vtag, ff in (("v", f), ("n", np.ones(5))):
                    dl = att - dfn / ff
                    rec[f"zone_wide_b{bw:g}{vtag}"] = dl[[0, 4]].sum()
                    rec[f"zone_half_b{bw:g}{vtag}"] = dl[[1, 3]].sum()
                    rec[f"zone_centre_b{bw:g}{vtag}"] = dl[2]
            # duels: geometric pairing
            ia = np.argsort(-a["xy"][:, 0])[:5]
            ib = np.argsort(b["xy"][:, 0])[:5]
            D = np.hypot(*(a["xy"][ia, None, :] - bm[None, ib, :]).transpose(2, 0, 1))
            r, c = linear_sum_assignment(D)
            pa, pb = ia[r], ib[c]
            rec["dribble"] = float((a["z"]["dribbles"][pa] + b["z"]["dribbled_past"][pb]).sum())
            rec["aerial"] = float((a["z"]["aerial_win"][pa] - b["z"]["aerial_win"][pb]).sum())
            # duels without positions: attacking groups by q_att vs defensive groups by q_def
            att_ok = np.isin(a["grp"], ["FW", "W", "AM"])
            def_ok = np.isin(b["grp"], ["CB", "FB"])
            na = np.flatnonzero(att_ok)[np.argsort(-a["qa"][att_ok])][:5]
            nb = np.flatnonzero(def_ok)[np.argsort(-b["qd"][def_ok])][:5]
            k = min(len(na), len(nb))
            rec["dribble_nopos"] = float((a["z"]["dribbles"][na[:k]]).sum() + b["z"]["dribbled_past"][nb[:k]].sum())
            rec["aerial_nopos"] = float(a["z"]["aerial_win"][na[:k]].sum() - b["z"]["aerial_win"][nb[:k]].sum())
            # press inputs (combined after as-of standardization below)
            sa = sty_k.loc[(mt.match_id, A)] if (mt.match_id, A) in sty_k.index else None
            sb = sty_k.loc[(mt.match_id, B)] if (mt.match_id, B) in sty_k.index else None
            rec["press_raw"] = -sa.own_log_ppda if sa is not None else np.nan
            rec["exp_gk"] = sb.own_gk_share if sb is not None else np.nan
            rec["exp_vert"] = sb.own_verticality if sb is not None else np.nan
            rec["exp_long"] = sb.own_long_ball_share if sb is not None else np.nan
            rec["exp_cbbtw"] = b["cb_btw"]
            rows.append(rec)
    F = pd.DataFrame(rows).sort_values(["date", "match_id", "team_id"]).reset_index(drop=True)

    # as-of standardization of the press inputs: constants from earlier rows only (expanding window)
    def asof_z(col):
        v = F[col].to_numpy(float)
        out = np.full(len(v), np.nan)
        d = F.date.dt.floor("D").to_numpy()
        for day in np.unique(d):
            past = v[(d < day) & (d >= day - np.timedelta64(365, "D")) & ~np.isnan(v)]   # previous 365 days
            if len(past) < 40:
                continue
            sel = d == day
            out[sel] = (v[sel] - past.mean()) / (past.std() or 1)
        return out

    zc = {c: asof_z(c) for c in ["press_raw", "exp_gk", "exp_vert", "exp_long", "exp_cbbtw"]}
    exp_full = zc["exp_gk"] + zc["exp_cbbtw"] - zc["exp_vert"]
    exp_geo = -zc["exp_vert"]
    exp_none = -zc["exp_long"]
    F["press"] = zc["press_raw"] * exp_full
    F["press_geo"] = zc["press_raw"] * exp_geo
    F["press_none"] = zc["press_raw"] * exp_none
    F = F.drop(columns=["press_raw", "exp_gk", "exp_vert", "exp_long", "exp_cbbtw"])
    # final as-of standardization of every matchup column
    feat_cols = [c for c in F.columns if c.startswith(("zone_", "dribble", "aerial", "press"))]
    Fz = F.copy()
    for c in feat_cols:
        Fz[c] = asof_z(c)
    Fz[feat_cols] = Fz[feat_cols].fillna(0.0)
    # style scores of both teams
    sc = style_scores()
    Fz = Fz.merge(sc, on=["match_id", "team_id"], how="left")
    Fz = Fz.merge(sc.rename(columns=lambda c: c if c == "match_id" else ("opponent_id" if c == "team_id" else f"opp_{c}")),
                  on=["match_id", "opponent_id"], how="left")
    Fz.to_parquet(PROC / "matchup_features.parquet", index=False)
    return Fz


if __name__ == "__main__":
    F = build()
    pd.set_option("display.width", 220)
    print(F.shape)
    print(F.describe().T.round(3).to_string())
