"""Dashboard-ready data for the visuals: shots, passing network, team shape (convex hull), radars.

Everything reads the processed tables written by src/ingest.py (run it first). All coordinates are in
metres on a 105 x 68 pitch from the team's own perspective: own goal at x = 0, attacking toward x = 105,
y = 68 on the team's left (CLAUDE.md, §2.2). Each function returns plain DataFrames / dicts so the same
numbers can feed the matplotlib figures (viz_plots.py, viz_radar.py) and a dashboard (viz_export.py).

These are descriptive views of what happened; nothing here feeds a forecast, so the as-of rule does not
apply. The one model output used (L1 player impact) is read from reports/player_ratings_current.csv.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

# ---------------------------------------------------------------- vocabularies
# Situation filter shown to the user -> source `situation` values
SITUATIONS = {
    "Open play": ["RegularPlay", "IndividualPlay"],
    "Fast break": ["FastBreak"],
    "Set piece": ["FromCorner", "SetPiece", "ThrowInSetPiece", "FreeKick"],
    "Penalty": ["Penalty"],
}
SITUATION_OF = {s: k for k, v in SITUATIONS.items() for s in v}
OUTCOMES = ["Goal", "Saved", "Blocked", "Off target"]   # blocked shots are "saved" + is_blocked in the source
CB_SLOTS = {"LCB", "CB", "RCB"}
GROUP_OF_SLOT = {"LCB": "CB", "RCB": "CB", "CB": "CB", "LB": "FB", "RB": "FB", "LWB": "FB", "RWB": "FB",
                 "CM": "CM", "DM": "CM", "CAM": "AM", "LW": "W", "RW": "W", "ST": "FW"}
GROUP_NAMES = {"CB": "centre-backs", "FB": "full-backs", "CM": "central midfielders",
               "AM": "attacking midfielders", "W": "wingers", "FW": "strikers"}


@lru_cache(maxsize=None)
def table(name: str) -> pd.DataFrame:
    path = PROC / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run `python src/ingest.py` first")
    return pd.read_parquet(path)


def surname(full: str) -> str:
    parts = str(full).replace(".", ". ").split()
    return parts[-1] if parts else str(full)


def match_info(match_id: int) -> dict:
    m = table("matches").set_index("match_id").loc[match_id]
    date = pd.Timestamp(m.date)
    return {"match_id": int(match_id), "season": m.season, "round": int(m["round"]),
            "date": date.strftime("%Y-%m-%d"), "home_team": m.home_team, "away_team": m.away_team,
            "home_score": int(m.home_score), "away_score": int(m.away_score),
            "label": (f"{m.home_team} {int(m.home_score)}–{int(m.away_score)} {m.away_team}  ·  "
                      f"Premier League {m.season.replace('/20', '/')}, round {int(m['round'])}  ·  "
                      f"{date.day} {date:%b %Y}")}


def opponent(match_id: int, team: str) -> str:
    m = match_info(match_id)
    return m["away_team"] if m["home_team"] == team else m["home_team"]


# ---------------------------------------------------------------- shots
def shots(match_id: int | None = None, team: str | None = None, situations=None,
          season: str | None = None) -> pd.DataFrame:
    """One row per shot (own goals dropped) with the labels the shot map uses.

    situations: iterable of keys of SITUATIONS (None = all). This is the dashboard's situation filter.
    """
    s = table("shots")
    s = s[~s.is_own_goal.astype(bool)]
    if match_id is not None:
        s = s[s.match_id == match_id]
    if team is not None:
        s = s[s.team == team]
    if season is not None:
        s = s[s.season == season]
    s = s.copy()
    s["situation_label"] = s.situation.map(SITUATION_OF).fillna("Open play")
    s["outcome_label"] = np.select(
        [s.outcome.eq("goal"), s.is_blocked.astype(bool), s.outcome.eq("saved")],
        ["Goal", "Blocked", "Saved"], "Off target")
    added = s.minute_added.fillna(0).astype(int)
    s["minute_label"] = np.where(added > 0, s.minute.astype(int).astype(str) + "+" + added.astype(str),
                                 s.minute.astype(int).astype(str)) + "'"
    s["player_short"] = s.player_name.map(surname)
    if situations is not None:
        bad = set(situations) - set(SITUATIONS)
        if bad:
            raise ValueError(f"unknown situations {bad}; choose from {list(SITUATIONS)}")
        s = s[s.situation_label.isin(list(situations))]
    cols = ["match_id", "season", "team", "opponent", "shot_id", "player_id", "player_name", "player_short",
            "period", "minute", "minute_added", "minute_label", "x_m", "y_m", "xg", "xgot", "shot_type",
            "situation", "situation_label", "outcome", "outcome_label", "is_from_inside_box"]
    return s[cols].sort_values(["match_id", "minute", "shot_id"]).reset_index(drop=True)


def shot_summary(s: pd.DataFrame) -> dict:
    n = len(s)
    return {"shots": n, "goals": int((s.outcome_label == "Goal").sum()),
            "xg": float(s.xg.sum()), "npxg": float(s.loc[s.situation != "Penalty", "xg"].sum()),
            "on_target": int(s.outcome_label.isin(["Goal", "Saved"]).sum()),
            "blocked": int((s.outcome_label == "Blocked").sum()),
            "xg_per_shot": float(s.xg.mean()) if n else 0.0}


# ---------------------------------------------------------------- passing network
def network(match_id: int, team: str, min_link: int = 4) -> dict:
    """Starting XI at average positions plus links between starters (both directions combined).

    Average positions and pass counts cover each starter's whole time on the pitch (the source gives
    match totals, no timestamps); substitutes are left out, the usual convention.
    """
    n = table("network_nodes")
    n = n[(n.match_id == match_id) & (n.team == team) & n.started].copy()
    if len(n) != 11:
        raise ValueError(f"{team} in {match_id}: {len(n)} starters in the network, expected 11")
    pm = table("player_match")
    pm = pm[(pm.match_id == match_id) & (pm.team == team)][["player_id", "player_name", "starting_position",
                                                            "is_gk", "passes_attempted"]]
    n = n.drop(columns=["player_name"]).merge(pm, on="player_id", how="left")
    n["name"] = n.player_name.map(surname)
    n["passes_attempted_net"] = n.passes_completed + n.passes_failed
    n["subbed_off"] = n.minutes_played < 89
    nodes = n[["player_id", "player_name", "name", "shirt_number", "starting_position", "is_gk", "x_m", "y_m",
               "passes_completed", "passes_attempted_net", "minutes_played", "subbed_off"]].reset_index(drop=True)

    e = table("network_edges")
    ids = set(nodes.player_id)
    e = e[(e.match_id == match_id) & (e.team == team) & e.passer_id.isin(ids) & e.receiver_id.isin(ids)].copy()
    e["a"] = e[["passer_id", "receiver_id"]].min(axis=1)
    e["b"] = e[["passer_id", "receiver_id"]].max(axis=1)
    links = e.groupby(["a", "b"], as_index=False).pass_count.sum()
    pos = nodes.set_index("player_id")[["x_m", "y_m"]]
    links = links.join(pos, on="a").join(pos, on="b", rsuffix="_b").sort_values("pass_count")
    names = nodes.set_index("player_id").name
    top = e.sort_values("pass_count").iloc[-1]
    hub = nodes.sort_values("passes_completed").iloc[-1]
    att = nodes.passes_attempted_net.sum()
    summary = {"passes_completed": int(nodes.passes_completed.sum()),
               "completion": float(nodes.passes_completed.sum() / att) if att else np.nan,
               "links_shown": int((links.pass_count >= min_link).sum()), "min_link": min_link,
               "hub": hub["name"], "hub_passes": int(hub.passes_completed),
               "top_link": f"{names[top.passer_id]} → {names[top.receiver_id]}",
               "top_link_passes": int(top.pass_count),
               "substituted": [(r.name, int(r.minutes_played)) for r in
                               nodes[nodes.subbed_off].sort_values("minutes_played").itertuples()]}
    return {"nodes": nodes, "links": links.reset_index(drop=True), "summary": summary}


# ---------------------------------------------------------------- team shape (convex hull)
def _shape(g: pd.DataFrame) -> dict:
    """Shape metrics for one team-match from its outfield starters (x_m, y_m, starting_position)."""
    pts = g[["x_m", "y_m"]].to_numpy()
    h = ConvexHull(pts)
    cb = g[g.starting_position.isin(CB_SLOTS)]
    if len(cb) < 2:  # no slot labels: use the two deepest outfield players
        cb = g.nsmallest(2, "x_m")
    return {"area_m2": float(h.volume), "width_m": float(np.ptp(pts[:, 1])), "depth_m": float(np.ptp(pts[:, 0])),
            "cb_height_m": float(cb.x_m.mean()), "centre_x_m": float(pts[:, 0].mean()),
            "centre_y_m": float(pts[:, 1].mean()), "hull_idx": h.vertices}


def shape_table() -> pd.DataFrame:
    """Shape metrics for every team-match (outfield starters). Cached in data/processed/dash_shapes.parquet."""
    path = PROC / "dash_shapes.parquet"
    if path.exists():
        return pd.read_parquet(path)
    n = table("network_nodes")
    pm = table("player_match")[["match_id", "player_id", "is_gk", "starting_position"]]
    n = n[n.started].merge(pm, on=["match_id", "player_id"], how="left")
    n = n[~n.is_gk.fillna(False).astype(bool)].dropna(subset=["x_m", "y_m"])
    rows = []
    for (mid, team), g in n.groupby(["match_id", "team"]):
        if len(g) >= 8:
            d = _shape(g)
            d.pop("hull_idx")
            rows.append({"match_id": mid, "team": team, "n_players": len(g), **d})
    out = pd.DataFrame(rows).merge(table("matches")[["match_id", "season", "date"]], on="match_id")
    out.to_parquet(path, index=False)
    return out


def team_shape(match_id: int, team: str) -> dict:
    """Convex hull of the 10 outfield starters' average positions, with metrics and a league reference."""
    nodes = network(match_id, team)["nodes"]
    of = nodes[~nodes.is_gk.fillna(False).astype(bool)].reset_index(drop=True)
    d = _shape(of)
    ref = shape_table()
    d["area_pct"] = float((ref.area_m2 < d["area_m2"]).mean())
    d["n_reference"] = int(len(ref))
    d["hull"] = of.loc[d.pop("hull_idx"), ["x_m", "y_m"]].reset_index(drop=True)
    d["nodes"] = nodes
    return d


# ---------------------------------------------------------------- radars
# (label, column, section)
RADAR_METRICS = [
    ("Non-penalty xG", "npxg", "Shooting & creating"),
    ("xA", "xa", "Shooting & creating"),
    ("Chances created", "chances_created", "Shooting & creating"),
    ("Touches in\nopposition box", "touches_opp_box", "Shooting & creating"),
    ("Successful\ndribbles", "successful_dribbles", "Shooting & creating"),
    ("Accurate passes", "accurate_passes", "Passing"),
    ("Pass\ncompletion", "pass_pct", "Passing"),
    ("Line-breaking\npasses", "line_breaking_passes", "Passing"),
    ("Passes into\nfinal third", "passes_into_final_third", "Passing"),
    ("Tackles +\ninterceptions*", "tkl_int_padj", "Defending"),
    ("Recoveries*", "recoveries_padj", "Defending"),
    ("Ground duel\nwin rate", "ground_duel_pct", "Defending"),
]
RATE_COLS = {"pass_pct", "ground_duel_pct"}
_SUM_COLS = ["npxg", "xa", "chances_created", "touches_opp_box", "successful_dribbles", "accurate_passes",
             "passes_attempted", "line_breaking_passes", "passes_into_final_third", "tkl_int_padj",
             "recoveries_padj", "ground_duels_won", "ground_duels_attempted"]


def radar_table(season: str, min_minutes: int = 900) -> pd.DataFrame:
    """Season per-90 rates for outfield players with >= min_minutes, wide format (one row per player).

    Non-penalty xG; tackles + interceptions and recoveries adjusted to 50% opponent possession.
    Group = the position group of the player's most frequent starting slot that season.
    """
    p = table("player_match")
    p = p[(p.season == season) & p.played & ~p.is_gk.astype(bool) & (p.minutes_played > 0)].copy()
    f = 50 / (100 - p.team_possession_pct)
    p["tkl_int_padj"] = (p.tackles + p.interceptions) * f
    p["recoveries_padj"] = p.recoveries * f
    sums = p.groupby("player_id").agg(name=("player_name", "last"), team=("team", "last"),
                                      minutes=("minutes_played", "sum"), **{c: (c, "sum") for c in _SUM_COLS})
    slot = p.dropna(subset=["starting_position"]).groupby("player_id").starting_position.agg(
        lambda s: s.value_counts().index[0])
    sums["group"] = slot.map(GROUP_OF_SLOT)
    per90 = sums.minutes / 90
    out = sums[["name", "team", "minutes", "group"]].copy()
    for _, c, _ in RADAR_METRICS:
        if c == "pass_pct":
            out[c] = 100 * sums.accurate_passes / sums.passes_attempted
        elif c == "ground_duel_pct":
            out[c] = 100 * sums.ground_duels_won / sums.ground_duels_attempted
        else:
            out[c] = sums[c] / per90
    out = out[(out.minutes >= min_minutes) & out.group.notna()]
    out.insert(0, "season", season)
    return out


def radar_long(season: str, min_minutes: int = 900) -> pd.DataFrame:
    """Long format for the dashboard: one row per player x metric with value, percentile, z and peer stats."""
    w = radar_table(season, min_minutes)
    rows = []
    for _, c, sec in RADAR_METRICS:
        g = w.groupby("group")[c]
        mean, sd, med = g.transform("mean"), g.transform("std"), g.transform("median")
        pct = g.rank(pct=False, method="average").sub(0.5).div(g.transform("count")) * 100
        rows.append(pd.DataFrame({"season": season, "player_id": w.index, "name": w.name, "team": w.team,
                                  "group": w.group, "minutes": w.minutes, "metric": c, "section": sec,
                                  "value": w[c], "percentile": pct, "z": (w[c] - mean) / sd,
                                  "peer_mean": mean, "peer_median": med,
                                  "n_peers": g.transform("count")}))
    return pd.concat(rows, ignore_index=True)


def player_radar(season: str, player: str | int, min_minutes: int = 900) -> dict:
    """Values, percentiles and z-scores against the player's position group."""
    lg = radar_long(season, min_minutes)
    me = lg[lg.player_id == player] if isinstance(player, (int, np.integer)) else lg[lg.name == player]
    if me.empty:
        raise ValueError(f"{player} not found with >= {min_minutes} minutes in {season}")
    me = me.set_index("metric").loc[[c for _, c, _ in RADAR_METRICS]]
    first = me.iloc[0]
    return {"player_id": int(first.player_id), "name": first["name"], "team": first.team, "group": first.group,
            "group_name": GROUP_NAMES[first.group], "minutes": int(first.minutes),
            "n_peers": int(first.n_peers), "season": season, "metrics": me.reset_index()}


def player_impact(player: str | int) -> dict | None:
    """L1 player effects (Step 3) relative to the player's position group, from player_ratings_current.csv.

    att = change in own team's npxG per 90, def = opponent npxG per 90 prevented, net = att + def; each
    minus the group mean, plus the player's percentile in the group. Window: the 365 days before the
    ratings date (the end of the data). Returns None if the player has no rating (< 900 minutes).
    """
    path = REPORTS / "player_ratings_current.csv"
    if not path.exists():
        return None
    r = pd.read_csv(path)
    r = r[~r.is_gk & r.group.notna()].copy()
    for c in ["att", "def", "net"]:
        col = f"{c}_npxg_per90"
        r[f"{c}_vs_group"] = r[col] - r.groupby("group")[col].transform("mean")
        g = r.groupby("group")[col]
        r[f"{c}_pct"] = (g.rank(method="average") - 0.5) / g.transform("count") * 100
    me = r[r.player_id == player] if isinstance(player, (int, np.integer)) else r[r.player_name == player]
    if me.empty:
        return None
    m = me.iloc[0]
    return {"att_vs_group": float(m.att_vs_group), "def_vs_group": float(m.def_vs_group),
            "net_vs_group": float(m.net_vs_group), "att_pct": float(m.att_pct), "def_pct": float(m.def_pct),
            "net_pct": float(m.net_pct), "group": m.group, "n_group": int((r.group == m.group).sum()),
            "minutes_365d": int(m.minutes_365d)}
