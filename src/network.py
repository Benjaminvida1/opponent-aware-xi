"""Passing-network features per team-match (methodology §5.1–5.4, §6).

Run:  python src/network.py

For every team-match this computes
  * shape (§5.2) from the 10 outfield starters' average positions,
  * each starter's slot (§5.3): Hungarian assignment within the teamsheet formation's template, and
    the shape in possession: lines found by 1-D k-means on depth (deviation from §5.3, methodology v0.6),
  * structure (§5.4) from the full-match graph of every player who appeared,
  * team features (§6): possession, PPDA proxy, box entries, crossing tendency,
  * context used by the style decomposition (§5.5): game-state shares, early red card flag.

Nothing here looks across matches, so there is no as-of issue: each row uses only its own match.

Outputs (cached, regenerate with this script):
  data/processed/team_match_network.parquet   one row per team-match
  data/processed/player_slots.parquet         one row per outfield starter: formation template, slot, line in possession
  reports/formation_checks.json, reports/formation_vs_teamsheet.csv, reports/slot_vs_label.csv
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
PITCH_X, PITCH_Y = 105.0, 68.0

# --------------------------------------------------------------------------- formation templates (§5.3)
# Each template: lines from back to front; each line is a list of (slot, x, y) in metres on the team's
# own frame (own goal x = 0, y = 68 on the team's left). Only the relative layout matters: observed and
# template coordinates are both standardized per axis before matching. Slot type D/M/F feeds §13.
_W4 = [56, 42, 26, 12]
TEMPLATES: dict[str, list[list[tuple[str, float, float]]]] = {
    "4-4-2": [[("LB", 32, 56), ("LCB", 28, 42), ("RCB", 28, 26), ("RB", 32, 12)],
              [("LM", 54, 56), ("LCM", 50, 42), ("RCM", 50, 26), ("RM", 54, 12)],
              [("LS", 72, 41), ("RS", 72, 27)]],
    "4-3-3": [[("LB", 32, 56), ("LCB", 28, 42), ("RCB", 28, 26), ("RB", 32, 12)],
              [("DM", 44, 34), ("LCM", 54, 45), ("RCM", 54, 23)],
              [("LW", 70, 56), ("ST", 74, 34), ("RW", 70, 12)]],
    "4-2-3-1": [[("LB", 32, 56), ("LCB", 28, 42), ("RCB", 28, 26), ("RB", 32, 12)],
                [("LDM", 46, 41), ("RDM", 46, 27)],
                [("LAM", 64, 54), ("CAM", 63, 34), ("RAM", 64, 14)],
                [("ST", 75, 34)]],
    "4-1-4-1": [[("LB", 32, 56), ("LCB", 28, 42), ("RCB", 28, 26), ("RB", 32, 12)],
                [("DM", 43, 34)],
                [("LM", 60, 56), ("LCM", 56, 42), ("RCM", 56, 26), ("RM", 60, 12)],
                [("ST", 74, 34)]],
    "4-4-1-1": [[("LB", 32, 56), ("LCB", 28, 42), ("RCB", 28, 26), ("RB", 32, 12)],
                [("LM", 52, 56), ("LCM", 48, 42), ("RCM", 48, 26), ("RM", 52, 12)],
                [("SS", 64, 34)],
                [("ST", 74, 34)]],
    "3-4-3": [[("LCB", 29, 50), ("CB", 26, 34), ("RCB", 29, 18)],
              [("LWB", 52, 61), ("LCM", 48, 42), ("RCM", 48, 26), ("RWB", 52, 7)],
              [("LW", 70, 52), ("ST", 74, 34), ("RW", 70, 16)]],
    "3-5-2": [[("LCB", 29, 50), ("CB", 26, 34), ("RCB", 29, 18)],
              [("LWB", 54, 62), ("LCM", 52, 45), ("CM", 46, 34), ("RCM", 52, 23), ("RWB", 54, 6)],
              [("LS", 72, 41), ("RS", 72, 27)]],
    "3-4-2-1": [[("LCB", 29, 50), ("CB", 26, 34), ("RCB", 29, 18)],
                [("LWB", 52, 61), ("LCM", 47, 41), ("RCM", 47, 27), ("RWB", 52, 7)],
                [("LAM", 65, 45), ("RAM", 65, 23)],
                [("ST", 75, 34)]],
    "4-3-1-2": [[("LB", 32, 56), ("LCB", 28, 42), ("RCB", 28, 26), ("RB", 32, 12)],
                [("DM", 44, 34), ("LCM", 52, 46), ("RCM", 52, 22)],
                [("CAM", 63, 34)],
                [("LS", 73, 41), ("RS", 73, 27)]],
    "5-3-2": [[("LWB", 38, 61), ("LCB", 28, 47), ("CB", 26, 34), ("RCB", 28, 21), ("RWB", 38, 7)],
              [("LCM", 52, 46), ("CM", 48, 34), ("RCM", 52, 22)],
              [("LS", 70, 41), ("RS", 70, 27)]],
    "5-4-1": [[("LWB", 38, 61), ("LCB", 28, 47), ("CB", 26, 34), ("RCB", 28, 21), ("RWB", 38, 7)],
              [("LM", 54, 56), ("LCM", 50, 42), ("RCM", 50, 26), ("RM", 54, 12)],
              [("ST", 70, 34)]],
}


def _template_arrays():
    out = {}
    for name, lines in TEMPLATES.items():
        slots, xy, line_idx, stype = [], [], [], []
        for k, line in enumerate(lines):
            for s, x, y in line:
                slots.append(s)
                xy.append((x, y))
                line_idx.append(k)
                stype.append("D" if k == 0 else ("F" if k == len(lines) - 1 else "M"))
        xy = np.array(xy, float)
        assert len(slots) == 10, name
        out[name] = dict(slots=slots, xy=xy, z=_standardize(xy), line=np.array(line_idx), stype=stype)
    return out


def _standardize(xy: np.ndarray) -> np.ndarray:
    c = xy - xy.mean(0)
    sd = c.std(0)
    sd[sd == 0] = 1
    return c / sd


TEMPLATE_ARR = None

# Teamsheet formations outside the library map to the nearest library template (rare: 53 of 2,380 team-matches).
SHEET_TO_TEMPLATE = {"3-2-4-1": "3-4-2-1", "3-4-1-2": "3-5-2", "4-1-2-3": "4-3-3", "4-5-1": "4-1-4-1",
                     "4-3-2-1": "4-3-3", "4-1-3-2": "4-3-1-2", "4-1-2-1-2": "4-3-1-2", "4-2-2-2": "4-4-2",
                     "3-1-4-2": "3-5-2", "3-5-1-1": "3-5-2", "4-2-4": "4-4-2", "5-3-1-1": "5-3-2"}


def sheet_template(f: str) -> str | None:
    f = SHEET_TO_TEMPLATE.get(f, f)
    return f if f in TEMPLATES else None


def infer_formation(xy: np.ndarray, only: list[str] | None = None) -> dict:
    """Hungarian assignment of 10 outfield positions to each template; lowest standardized cost wins.
    `only` restricts the search (e.g. to the teamsheet formation's template)."""
    global TEMPLATE_ARR
    if TEMPLATE_ARR is None:
        TEMPLATE_ARR = _template_arrays()
    z = _standardize(xy)
    best = None
    costs = {}
    for name, t in TEMPLATE_ARR.items():
        if only is not None and name not in only:
            continue
        d = ((z[:, None, :] - t["z"][None, :, :]) ** 2).sum(-1)
        r, c = linear_sum_assignment(d)
        cost = float(d[r, c].sum())
        costs[name] = cost
        if best is None or cost < best[1]:
            best = (name, cost, c)
    name, cost, col = best
    t = TEMPLATE_ARR[name]
    ranked = sorted(costs.values())
    return dict(formation=name, cost=cost, margin=(ranked[1] - ranked[0]) if len(ranked) > 1 else np.nan,
                slots=[t["slots"][j] for j in col], line=t["line"][col], stype=[t["stype"][j] for j in col])


def detect_lines(x: np.ndarray) -> tuple[str, float, np.ndarray]:
    """Optimal 1-D k-means on outfield x for k in {3, 4}, k chosen by silhouette.
    Returns the line structure 'a-b-c(-d)' read back to front, the silhouette, and each player's line
    (0 = deepest), in the input order."""
    order = np.argsort(x, kind="stable")
    xs = x[order]
    n = len(xs)
    best = None
    for k in (3, 4):
        bk = None
        for cuts in itertools.combinations(range(1, n), k - 1):
            groups = np.split(xs, cuts)
            sse = sum(((g - g.mean()) ** 2).sum() for g in groups)
            if bk is None or sse < bk[0]:
                bk = (sse, cuts)
        labels = np.repeat(np.arange(k), np.diff(np.r_[0, bk[1], n]))
        sil = _silhouette_1d(xs, labels)
        if best is None or sil > best[1]:
            lab = np.empty(n, int)
            lab[order] = labels
            best = ("-".join(str(c) for c in np.bincount(labels)), sil, lab)
    return best


def _silhouette_1d(x: np.ndarray, lab: np.ndarray) -> float:
    d = np.abs(x[:, None] - x[None, :])
    s = []
    for i in range(len(x)):
        own = lab == lab[i]
        if own.sum() <= 1:
            s.append(0.0)
            continue
        a = d[i, own].sum() / (own.sum() - 1)
        b = min(d[i, lab == k].mean() for k in np.unique(lab) if k != lab[i])
        s.append((b - a) / max(a, b))
    return float(np.mean(s))


# --------------------------------------------------------------------------- helpers
def gini(v: np.ndarray) -> float:
    v = np.sort(np.asarray(v, float))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return np.nan
    return float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * v.sum()))


def game_state_shares(matches: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    """Share of the 90 minutes each side spent trailing, level and leading (goals from shots, §3)."""
    goals = shots[shots.is_goal == 1][["match_id", "minute", "minute_added", "period", "scoring_team_id"]].copy()
    goals["t"] = goals.minute.clip(upper=90).astype(float)
    rows = []
    gb = {k: v.sort_values(["t", "minute_added"]) for k, v in goals.groupby("match_id")}
    for r in matches.itertuples():
        g = gb.get(r.match_id)
        times = [0.0] + ([] if g is None else list(g.t)) + [90.0]
        diff, diffs = 0, [0]
        if g is not None:
            for sid in g.scoring_team_id:
                diff += 1 if sid == r.home_team_id else -1
                diffs.append(diff)
        dur = np.diff(times)
        d = np.array(diffs)
        lead, level, trail = dur[d > 0].sum() / 90, dur[d == 0].sum() / 90, dur[d < 0].sum() / 90
        rows.append(dict(match_id=r.match_id, team_id=r.home_team_id, share_leading=lead, share_level=level, share_trailing=trail))
        rows.append(dict(match_id=r.match_id, team_id=r.away_team_id, share_leading=trail, share_level=level, share_trailing=lead))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- per team-match
METRICS = {
    # shape (§5.2), outfield starters
    "centroid_x": "Centroid height (m)",
    "def_line_height": "Defensive line height (m)",
    "att_height": "Attacking height (m)",
    "length": "Team length (m)",
    "width": "Team width (m)",
    "width_range": "Width, widest to narrowest (m)",
    "compactness": "Hull area of average positions (m²)",
    "gap_back_mid": "Gap back line to midfield (m)",
    "gap_mid_front": "Gap midfield to front line (m)",
    "fb_asymmetry": "Full-back asymmetry, left minus right (m)",
    "lateral_bias": "Lateral bias, + = left (m)",
    # structure (§5.4), full graph
    "passes": "Completed passes",
    "tempo": "Completed passes per possession share",
    "entropy": "Pass entropy",
    "gini_strength": "Strength inequality (Gini)",
    "reciprocity": "Reciprocity",
    "clustering": "Weighted clustering",
    "verticality": "Verticality (m per pass)",
    "forward_share": "Forward-pass share",
    "pass_span": "Mean pass span (m)",
    "flank_share": "Flank share",
    "lr_balance": "Left-right balance, + = left",
    "gk_share": "GK involvement",
    "max_betweenness": "Hub dependence (max betweenness)",
    "alg_connectivity": "Algebraic connectivity",
    # team features (§6)
    "poss": "Possession share",
    "log_ppda": "PPDA proxy (log)",
    "box_entries": "Box touches per 90 of possession",
    "cross_tendency": "Crosses per pass into final third",
    "long_ball_share": "Long balls per pass attempted",
}


def team_match_table(nodes, edges, pm, matches, shots, events):
    gk = pm[pm.played & pm.is_gk][["match_id", "player_id"]].assign(gk=True)
    nodes = nodes.merge(gk, on=["match_id", "player_id"], how="left")
    nodes["gk"] = nodes.gk.fillna(False).astype(bool)
    nodes["vol"] = nodes.passes_completed + nodes.passes_failed

    # §6 team box-score features
    played = pm[pm.played]
    tb = played.groupby(["match_id", "team_id"]).agg(
        passes_att=("passes_attempted", "sum"), def_actions=("tackles", "sum"), interceptions=("interceptions", "sum"),
        fouls=("fouls_committed", "sum"), box_touches=("touches_opp_box", "sum"), crosses=("crosses_attempted", "sum"),
        final_third=("passes_into_final_third", "sum"), long_balls=("accurate_long_balls_attempted", "sum"),
        poss_pct=("team_possession_pct", "first")).reset_index()
    tb["def_actions"] = tb.def_actions + tb.interceptions + tb.fouls
    opp = tb[["match_id", "team_id", "passes_att"]].rename(columns={"team_id": "opp_team", "passes_att": "opp_passes_att"})
    tb = tb.merge(opp, on="match_id")
    tb = tb[tb.team_id != tb.opp_team].drop(columns="opp_team")

    poss = pd.concat([
        matches[["match_id", "home_team_id", "home_possession"]].rename(columns={"home_team_id": "team_id", "home_possession": "poss"}),
        matches[["match_id", "away_team_id", "away_possession"]].rename(columns={"away_team_id": "team_id", "away_possession": "poss"}),
    ])
    edge_groups = {k: v for k, v in edges.groupby(["match_id", "team_id"])}
    sheet = {}
    for r_ in matches.itertuples():
        sheet[(r_.match_id, r_.home_team_id)] = r_.home_formation
        sheet[(r_.match_id, r_.away_team_id)] = r_.away_formation
    rows, slot_rows = [], []
    for (mid, tid), g in nodes.groupby(["match_id", "team_id"]):
        st = g[g.started & ~g.gk & g.x_m.notna()]
        r = dict(match_id=mid, team_id=tid, n_outfield_starters_with_position=len(st))
        if len(st) == 10:
            x, y = st.x_m.to_numpy(float), st.y_m.to_numpy(float)
            w = st.vol.clip(lower=1).to_numpy(float)
            w = w / w.sum()
            # Slots: Hungarian assignment within the teamsheet formation's template (the unit the optimizer fills).
            tmpl = sheet_template(sheet[(mid, tid)])
            fs = infer_formation(np.column_stack([x, y]), only=[tmpl])
            # Shape in possession: lines found in the data (1-D k-means on depth), not a template.
            lines, sil, line = detect_lines(x)
            back, front = line == 0, line == line.max()
            second, penult = line == 1, line == line.max() - 1
            deep5 = np.argsort(x)[:5]
            left, right = deep5[np.argmax(y[deep5])], deep5[np.argmin(y[deep5])]
            yc = float(w @ y)
            r.update(formation=tmpl, formation_fit_cost=fs["cost"],
                     shape_in_possession=lines, lines_silhouette=sil, n_lines=int(line.max() + 1),
                     back_line_in_possession=int(back.sum()), front_line_in_possession=int(front.sum()),
                     centroid_x=float(w @ x), centroid_y=yc,
                     def_line_height=float(np.sort(x)[:4].mean()),
                     att_height=float(np.sort(x)[-3:].mean()),
                     width=float(2 * np.sqrt(w @ (y - yc) ** 2)),
                     width_range=float(y.max() - y.min()),
                     compactness=float(ConvexHull(np.column_stack([x, y])).volume),
                     gap_back_mid=float(x[second].mean() - x[back].mean()),
                     gap_mid_front=float(x[front].mean() - x[penult].mean()),
                     fb_asymmetry=float(x[left] - x[right]),
                     lateral_bias=yc - PITCH_Y / 2)
            r["length"] = r["att_height"] - r["def_line_height"]
            for pid, s, ln, ty, l2, px, py in zip(st.player_id, fs["slots"], fs["line"], fs["stype"], line, x, y):
                slot_rows.append(dict(match_id=mid, team_id=tid, player_id=pid, formation=tmpl,
                                      slot=s, slot_line=int(ln), slot_type=ty,
                                      shape_in_possession=lines, line_in_possession=int(l2), x_m=px, y_m=py))

        # structure on the full graph (every player who appeared and has a position)
        ge = edge_groups.get((mid, tid), edges.iloc[:0])
        pos = g.set_index("player_id")[["x_m", "y_m"]].dropna()
        ge = ge[ge.passer_id.isin(pos.index) & ge.receiver_id.isin(pos.index) & (ge.pass_count > 0)]
        tot = ge.pass_count.sum()
        if tot > 0:
            pc = ge.pass_count.to_numpy(float)
            pij = pc / tot
            # Edge weights are match totals, not per-90 rates over shared minutes (deviation from §5.1,
            # methodology v0.6): per-90 rates let short substitute spells dominate strength-based metrics and
            # cut their split-half reliability on 2023/24-2024/25 (Gini 0.92 -> 0.34, clustering 0.90 -> 0.11).
            wt = pc
            xi, xj = pos.x_m.reindex(ge.passer_id).to_numpy(), pos.x_m.reindex(ge.receiver_id).to_numpy()
            yi, yj = pos.y_m.reindex(ge.passer_id).to_numpy(), pos.y_m.reindex(ge.receiver_id).to_numpy()
            r["passes"] = float(tot)
            r["entropy"] = float(-(pij * np.log(pij)).sum() / np.log(len(pij))) if len(pij) > 1 else np.nan
            ew = pd.DataFrame({"a": ge.passer_id.to_numpy(), "b": ge.receiver_id.to_numpy(), "w": wt})
            strength = pd.concat([ew.groupby("a").w.sum(), ew.groupby("b").w.sum()]).groupby(level=0).sum()
            r["gini_strength"] = gini(strength.to_numpy())
            rev = ge.set_index(["receiver_id", "passer_id"]).pass_count
            back_ = rev.reindex(list(zip(ge.passer_id, ge.receiver_id))).fillna(0).to_numpy()
            r["reciprocity"] = float(np.minimum(pc, back_).sum() / tot)
            r["verticality"] = float(pij @ (xj - xi))
            r["forward_share"] = float(pij[(xj > xi + 5)].sum())
            r["pass_span"] = float(pij @ np.hypot(xj - xi, yj - yi))
            r["flank_share"] = float(pij[np.abs(yj - PITCH_Y / 2) > 20].sum())
            r["lr_balance"] = float(pij[yj > PITCH_Y / 2].sum() - pij[yj < PITCH_Y / 2].sum())
            gks = set(g.loc[g.gk, "player_id"])
            r["gk_share"] = float(pij[ge.passer_id.isin(gks).to_numpy() | ge.receiver_id.isin(gks).to_numpy()].sum())
            G = nx.DiGraph()
            for a, b, c in zip(ge.passer_id, ge.receiver_id, wt):
                G.add_edge(a, b, distance=1.0 / c)
            r["max_betweenness"] = float(max(nx.betweenness_centrality(G, weight="distance", normalized=True).values()))
            # Undirected graph W~ + W~^T, scaled to per-90 shares of the team's total rate so that
            # algebraic connectivity is comparable across teams with different pass volumes.
            U = nx.Graph()
            for a, b, c in zip(ge.passer_id, ge.receiver_id, wt / wt.sum()):
                if a == b:
                    continue
                U.add_edge(a, b, weight=U[a][b]["weight"] + c if U.has_edge(a, b) else c)
            mx = max(d["weight"] for *_, d in U.edges(data=True))
            for *_, d in U.edges(data=True):
                d["wn"] = d["weight"] / mx
            r["clustering"] = float(np.mean(list(nx.clustering(U, weight="wn").values())))
            L = nx.laplacian_matrix(U, weight="weight").toarray()
            ev = np.sort(np.linalg.eigvalsh(L))
            r["alg_connectivity"] = float(ev[1]) if len(ev) > 1 else np.nan
        rows.append(r)

    df = pd.DataFrame(rows).merge(poss, on=["match_id", "team_id"]).merge(tb, on=["match_id", "team_id"], how="left")
    df["tempo"] = df.passes / df.poss
    df["log_ppda"] = np.log(df.opp_passes_att / df.def_actions.clip(lower=1))
    df["box_entries"] = df.box_touches / df.poss
    df["cross_tendency"] = df.crosses / df.final_third.clip(lower=1)
    df["long_ball_share"] = df.long_balls / df.passes_att.clip(lower=1)

    m = matches[["match_id", "season", "round", "date", "home_team_id", "away_team_id", "home_formation", "away_formation"]]
    df = df.merge(m, on="match_id")
    df["home"] = (df.team_id == df.home_team_id).astype(int)
    df["opponent_id"] = np.where(df.home == 1, df.away_team_id, df.home_team_id)
    df["teamsheet_formation"] = np.where(df.home == 1, df.home_formation, df.away_formation)
    df = df.drop(columns=["home_team_id", "away_team_id", "home_formation", "away_formation"])
    df = df.merge(game_state_shares(matches, shots), on=["match_id", "team_id"], how="left")
    early_red = events[(events.event == "red_card") & (events.minute < 60)].match_id.unique()
    df["red_before_60"] = df.match_id.isin(early_red)
    return df, pd.DataFrame(slot_rows)


# --------------------------------------------------------------------------- validation
def back_count(f: str) -> int | float:
    try:
        return int(str(f).split("-")[0])
    except ValueError:
        return np.nan


LABEL_TO_GROUP = {"LB": "L-back", "RB": "R-back", "LCB": "CB", "RCB": "CB", "CB": "CB",
                  "LWB": "L-back", "RWB": "R-back", "ST": "ST", "LS": "ST", "RS": "ST",
                  "LW": "L-wide att", "LAM": "L-wide att", "LM": "L-wide att",
                  "RW": "R-wide att", "RAM": "R-wide att", "RM": "R-wide att",
                  "CAM": "C-mid", "SS": "C-mid", "CM": "C-mid", "LCM": "C-mid", "RCM": "C-mid",
                  "DM": "C-mid", "LDM": "C-mid", "RDM": "C-mid"}


def validate(tm: pd.DataFrame, slots: pd.DataFrame, pm: pd.DataFrame) -> dict:
    d = tm.dropna(subset=["shape_in_possession"]).copy()
    d["back_inferred"] = d.back_line_in_possession
    d["back_sheet"] = d.teamsheet_formation.map(back_count)
    ct = pd.crosstab(d.teamsheet_formation, d.shape_in_possession)
    ct = ct.loc[ct.sum(1).sort_values(ascending=False).index]
    ct.to_csv(REPORTS / "formation_vs_teamsheet.csv")

    # slot vs source starting-position label, where the source has one
    lab = pm[pm.started & pm.starting_position.notna() & ~pm.is_gk][["match_id", "player_id", "starting_position"]]
    sv = slots.merge(lab, on=["match_id", "player_id"])
    sv["group_label"] = sv.starting_position.map(LABEL_TO_GROUP)
    # wide attackers in a back three are wing-backs in our slots; compare at the coarse side level too
    sv["group_slot"] = sv.slot.map(LABEL_TO_GROUP)
    side = lambda s: s.str[0].where(s.str[:2].isin(["L-", "R-"]), "C")
    sv["side_label"], sv["side_slot"] = side(sv.group_label), side(sv.group_slot)
    pd.crosstab(sv.starting_position, sv.slot).to_csv(REPORTS / "slot_vs_label.csv")
    out = dict(
        team_matches=int(len(tm)),
        with_formation=int(len(d)),
        missing_formation_reason="fewer than 10 outfield starters with a position (red card or no passes)",
        shape_in_possession_shares=d.shape_in_possession.value_counts(normalize=True).head(12).round(3).to_dict(),
        back_line_in_possession_shares=d.back_line_in_possession.value_counts(normalize=True).round(3).to_dict(),
        teamsheet_shares=d.teamsheet_formation.value_counts(normalize=True).head(12).round(3).to_dict(),
        back_line_in_possession_by_teamsheet_back=pd.crosstab(d.back_sheet, d.back_inferred, normalize="index").round(3).to_dict("index"),
        slot_label_n=int(len(sv)),
        slot_group_agreement=float((sv.group_label == sv.group_slot).mean()),
        slot_side_agreement=float((sv.side_label == sv.side_slot).mean()),
        slot_group_agreement_by_label=sv.groupby("starting_position").apply(lambda s: (s.group_label == s.group_slot).mean()).round(3).to_dict(),
    )
    return out


def main():
    nodes = pd.read_parquet(PROC / "network_nodes.parquet")
    edges = pd.read_parquet(PROC / "network_edges.parquet")
    pm = pd.read_parquet(PROC / "player_match.parquet")
    matches = pd.read_parquet(PROC / "matches.parquet")
    shots = pd.read_parquet(PROC / "shots.parquet")
    events = pd.read_parquet(PROC / "match_events.parquet")
    tm, slots = team_match_table(nodes, edges, pm, matches, shots, events)
    tm.to_parquet(PROC / "team_match_network.parquet", index=False)
    slots.to_parquet(PROC / "player_slots.parquet", index=False)
    checks = validate(tm, slots, pm)
    checks["metric_missing"] = {k: int(tm[k].isna().sum()) for k in METRICS}
    (REPORTS / "formation_checks.json").write_text(json.dumps(checks, indent=2, default=float))
    print(json.dumps(checks, indent=2, default=float))


if __name__ == "__main__":
    main()
