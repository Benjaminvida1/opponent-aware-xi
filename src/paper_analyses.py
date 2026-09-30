"""Preliminary analyses reported in the working paper.

Run:  python src/paper_analyses.py

A. Do passing-network features identify teams?  (supports methodology §5–6)
   Per team-match metrics from the network (shape from the 10 outfield starters,
   structure from the full-match graph). For 2025/26 (20 teams x 38 matches):
     - split-half reliability of team means (odd vs even matches by date), Spearman–Brown corrected
     - ICC(1): share of match-to-match variance that is between teams
     - the same after removing the match's possession share (identity beyond possession)
     - variance explained by team, by opponent and by both (two-way OLS; supports the §5.5 decomposition)

B. Which first-half signal best predicts second-half points?  (supports the xG target, §7)
   2025/26 split at round 19: per-match points, goal difference and npxG difference in rounds
   1-19, correlated with points per match in rounds 20-38. Also season points vs expected points.

Outputs: reports/network_reliability.csv, reports/network_variance.csv,
         reports/split_half_prediction.csv, reports/xpts_2025_26.csv, reports/paper_numbers.json
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
PITCH_Y = 68.0

METRIC_LABELS = {
    "def_line_height": "Defensive line height (m)",
    "att_height": "Attacking height (m)",
    "length": "Team length (m)",
    "width": "Team width (m)",
    "centroid_x": "Centroid height (m)",
    "passes": "Completed passes",
    "entropy": "Pass entropy",
    "gini_strength": "Strength inequality (Gini)",
    "reciprocity": "Reciprocity",
    "verticality": "Verticality (m per pass)",
    "forward_share": "Forward-pass share",
    "pass_span": "Mean pass span (m)",
    "flank_share": "Flank share",
    "gk_share": "GK involvement",
    "max_betweenness": "Hub dependence (max betweenness)",
}


def gini(v: np.ndarray) -> float:
    v = np.sort(np.asarray(v, float))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return np.nan
    return float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * v.sum()))


def team_match_metrics(nodes: pd.DataFrame, edges: pd.DataFrame, pm: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    gk = pm[pm.played & pm.is_gk][["match_id", "player_id"]].assign(gk=True)
    nodes = nodes.merge(gk, on=["match_id", "player_id"], how="left")
    nodes["gk"] = nodes.gk.fillna(False).astype(bool)
    nodes["vol"] = nodes.passes_completed + nodes.passes_failed
    poss = pd.concat([
        matches[["match_id", "home_team_id", "home_possession"]].rename(columns={"home_team_id": "team_id", "home_possession": "poss"}),
        matches[["match_id", "away_team_id", "away_possession"]].rename(columns={"away_team_id": "team_id", "away_possession": "poss"}),
    ])
    rows = []
    for (mid, tid), g in nodes.groupby(["match_id", "team_id"]):
        st = g[g.started & ~g.gk]
        if len(st) < 8:
            continue
        w = st.vol.clip(lower=1).to_numpy(float)
        w = w / w.sum()
        x, y = st.x_m.to_numpy(), st.y_m.to_numpy()
        xs = np.sort(x)
        r = dict(match_id=mid, team_id=tid,
                 centroid_x=float(w @ x),
                 def_line_height=float(xs[:4].mean()),
                 att_height=float(xs[-3:].mean()),
                 width=float(2 * np.sqrt(w @ (y - w @ y) ** 2)))
        r["length"] = r["att_height"] - r["def_line_height"]

        ge = edges[(edges.match_id == mid) & (edges.team_id == tid)]
        pos = g.set_index("player_id")[["x_m", "y_m"]]
        ge = ge[ge.passer_id.isin(pos.index) & ge.receiver_id.isin(pos.index)]
        tot = ge.pass_count.sum()
        if tot == 0:
            continue
        pij = ge.pass_count.to_numpy(float) / tot
        xi, xj = pos.x_m.reindex(ge.passer_id).to_numpy(), pos.x_m.reindex(ge.receiver_id).to_numpy()
        yi, yj = pos.y_m.reindex(ge.passer_id).to_numpy(), pos.y_m.reindex(ge.receiver_id).to_numpy()
        r["passes"] = float(tot)
        r["entropy"] = float(-(pij * np.log(pij)).sum() / np.log(len(pij))) if len(pij) > 1 else np.nan
        strength = pd.concat([ge.groupby("passer_id").pass_count.sum(), ge.groupby("receiver_id").pass_count.sum()]).groupby(level=0).sum()
        r["gini_strength"] = gini(strength.to_numpy())
        rev = ge.set_index(["receiver_id", "passer_id"]).pass_count
        back = rev.reindex(list(zip(ge.passer_id, ge.receiver_id))).fillna(0).to_numpy()
        r["reciprocity"] = float(np.minimum(ge.pass_count.to_numpy(), back).sum() / tot)
        r["verticality"] = float(pij @ (xj - xi))
        r["forward_share"] = float(pij[(xj > xi + 5)].sum())
        r["pass_span"] = float(pij @ np.hypot(xj - xi, yj - yi))
        r["flank_share"] = float(pij[np.abs(yj - PITCH_Y / 2) > 20].sum())
        gks = set(g.loc[g.gk, "player_id"])
        r["gk_share"] = float(pij[ge.passer_id.isin(gks).to_numpy() | ge.receiver_id.isin(gks).to_numpy()].sum())
        G = nx.DiGraph()
        for a, b, c in zip(ge.passer_id, ge.receiver_id, ge.pass_count):
            G.add_edge(a, b, distance=1.0 / c)
        r["max_betweenness"] = float(max(nx.betweenness_centrality(G, weight="distance", normalized=True).values()))
        rows.append(r)
    df = pd.DataFrame(rows).merge(poss, on=["match_id", "team_id"])
    m = matches[["match_id", "season", "round", "date", "home_team_id", "away_team_id"]]
    df = df.merge(m, on="match_id")
    df["home"] = (df.team_id == df.home_team_id).astype(int)
    df["opponent_id"] = np.where(df.home == 1, df.away_team_id, df.home_team_id)
    return df.drop(columns=["home_team_id", "away_team_id"])


def icc1(values: pd.Series, groups: pd.Series) -> float:
    d = pd.DataFrame({"v": values, "g": groups}).dropna()
    k = d.groupby("g").size().mean()
    grand = d.v.mean()
    msb = (d.groupby("g").v.apply(lambda s: len(s) * (s.mean() - grand) ** 2).sum()) / (d.g.nunique() - 1)
    msw = d.groupby("g").v.apply(lambda s: ((s - s.mean()) ** 2).sum()).sum() / (len(d) - d.g.nunique())
    return float((msb - msw) / (msb + (k - 1) * msw))


def split_half(df: pd.DataFrame, col: str) -> tuple[float, float]:
    d = df.sort_values("date").copy()
    d["half"] = d.groupby("team_id").cumcount() % 2
    means = d.groupby(["team_id", "half"])[col].mean().unstack()
    r = float(np.corrcoef(means[0], means[1])[0, 1])
    return r, 2 * r / (1 + r)


def r2_ols(y: np.ndarray, X: np.ndarray) -> float:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return float(1 - resid.var() / y.var())


def dummies(s: pd.Series) -> np.ndarray:
    return pd.get_dummies(s, drop_first=True).to_numpy(float)


def network_analysis(df: pd.DataFrame):
    s = df[df.season == "2025/2026"].copy()
    rel, var = [], []
    one = np.ones((len(s), 1))
    for col, label in METRIC_LABELS.items():
        v = s[col]
        # residual after possession (linear) for the "beyond possession" check
        Xp = np.column_stack([one, s.poss])
        beta, *_ = np.linalg.lstsq(Xp, v.to_numpy(), rcond=None)
        s[col + "_res"] = v - Xp @ beta
        r, sb = split_half(s, col)
        rr, sbr = split_half(s, col + "_res")
        rel.append(dict(metric=label, key=col, mean=v.mean(), between_team_sd=s.groupby("team_id")[col].mean().std(),
                        within_team_sd=s.groupby("team_id")[col].std().mean(),
                        split_half_r=r, reliability=sb, icc1=icc1(v, s.team_id),
                        reliability_beyond_possession=sbr, icc1_beyond_possession=icc1(s[col + "_res"], s.team_id)))
        y = v.to_numpy()
        T, O, H = dummies(s.team_id), dummies(s.opponent_id), s.home.to_numpy(float)[:, None]
        var.append(dict(metric=label, key=col,
                        r2_team=r2_ols(y, np.hstack([one, T])),
                        r2_opponent=r2_ols(y, np.hstack([one, O])),
                        r2_team_opponent_home=r2_ols(y, np.hstack([one, T, O, H])),
                        r2_possession=r2_ols(y, Xp)))
    return pd.DataFrame(rel), pd.DataFrame(var)


def result_points(gf, ga):
    return np.where(gf > ga, 3, np.where(gf == ga, 1, 0))


def xpts(lam_for: float, lam_against: float) -> float:
    g = np.arange(11)
    m = np.outer(poisson.pmf(g, lam_for), poisson.pmf(g, lam_against))
    m /= m.sum()
    return float(3 * np.tril(m, -1).sum() + np.trace(m))


def xg_analysis(matches: pd.DataFrame, shots: pd.DataFrame):
    s = shots[shots.is_own_goal == 0]
    tot = s.groupby(["match_id", "team_id"]).xg.sum()
    np_ = s[s.situation_group != "penalty"].groupby(["match_id", "team_id"]).xg.sum()
    rows = []
    for r in matches[matches.season == "2025/2026"].itertuples():
        for tid, oid, team, gf, ga in [(r.home_team_id, r.away_team_id, r.home_team, r.home_score, r.away_score),
                                       (r.away_team_id, r.home_team_id, r.away_team, r.away_score, r.home_score)]:
            xf, xa = tot.get((r.match_id, tid), 0.0), tot.get((r.match_id, oid), 0.0)
            rows.append(dict(match_id=r.match_id, round=r.round, date=r.date, team=team, gf=gf, ga=ga,
                             pts=int(result_points(gf, ga)), npxg_for=np_.get((r.match_id, tid), 0.0),
                             npxg_against=np_.get((r.match_id, oid), 0.0), xpts=xpts(xf, xa)))
    tm = pd.DataFrame(rows)
    tm["half"] = np.where(tm["round"] <= 19, "first", "second")
    agg = tm.groupby(["team", "half"]).agg(pts=("pts", "mean"), gd=("gf", "mean"), ga=("ga", "mean"),
                                            npxgd=("npxg_for", "mean"), npxga=("npxg_against", "mean"),
                                            xpts=("xpts", "mean")).unstack()
    first = pd.DataFrame({
        "Points per match": agg[("pts", "first")],
        "Goal difference per match": agg[("gd", "first")] - agg[("ga", "first")],
        "npxG difference per match": agg[("npxgd", "first")] - agg[("npxga", "first")],
        "Expected points per match": agg[("xpts", "first")],
    })
    target = agg[("pts", "second")]
    pred = []
    for name, v in first.items():
        pred.append(dict(predictor=name, r=float(np.corrcoef(v, target)[0, 1]),
                         r2=float(np.corrcoef(v, target)[0, 1] ** 2)))
    pred = pd.DataFrame(pred)
    # bootstrap the difference between npxG-diff and points as predictors (resample teams)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(target), (5000, len(target)))
    a, b, t = first["npxG difference per match"].to_numpy(), first["Points per match"].to_numpy(), target.to_numpy()
    diffs = np.array([np.corrcoef(a[i], t[i])[0, 1] - np.corrcoef(b[i], t[i])[0, 1] for i in idx])
    season = tm.groupby("team").agg(points=("pts", "sum"), xpts=("xpts", "sum"),
                                    goals_for=("gf", "sum"), goals_against=("ga", "sum"),
                                    npxg_for=("npxg_for", "sum"), npxg_against=("npxg_against", "sum")).reset_index()
    season["points_minus_xpts"] = season.points - season.xpts
    season = season.sort_values("points", ascending=False).reset_index(drop=True)
    season.insert(0, "position", np.arange(1, len(season) + 1))
    boot = dict(diff_r=float(np.median(diffs)), ci_low=float(np.quantile(diffs, 0.025)), ci_high=float(np.quantile(diffs, 0.975)))
    return pred, season, boot


def main():
    matches = pd.read_parquet(PROC / "matches.parquet")
    nodes = pd.read_parquet(PROC / "network_nodes.parquet")
    edges = pd.read_parquet(PROC / "network_edges.parquet")
    pm = pd.read_parquet(PROC / "player_match.parquet")
    shots = pd.read_parquet(PROC / "shots.parquet")

    tm = team_match_metrics(nodes, edges, pm, matches)
    tm.to_parquet(PROC / "team_match_network_metrics.parquet", index=False)
    rel, var = network_analysis(tm)
    rel.to_csv(REPORTS / "network_reliability.csv", index=False)
    var.to_csv(REPORTS / "network_variance.csv", index=False)

    pred, season, boot = xg_analysis(matches, shots)
    pred.to_csv(REPORTS / "split_half_prediction.csv", index=False)
    season.to_csv(REPORTS / "xpts_2025_26.csv", index=False)

    numbers = dict(team_matches_with_metrics=int(len(tm)), npxg_minus_points_r=boot,
                   corr_points_xpts_season=float(np.corrcoef(season.points, season.xpts)[0, 1]))
    (REPORTS / "paper_numbers.json").write_text(json.dumps(numbers, indent=2))
    return rel, var, pred, season, numbers


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    rel, var, pred, season, numbers = main()
    print(rel.drop(columns="key").round(3).to_string(index=False))
    print(var.drop(columns="key").round(3).to_string(index=False))
    print(pred.round(3).to_string(index=False))
    print(season.round(1).to_string(index=False))
    print(json.dumps(numbers, indent=2))
