"""Ingest the scraped CSVs into the Parquet tables of methodology §2.1.

Run:  python src/ingest.py            (from the project root)

Outputs (data/processed/):
    matches.parquet        one row per match
    player_match.parquet   one row per player per match (cleaned, blanks -> 0 where appropriate)
    shots.parquet          one row per shot, metres, situation groups
    network_nodes.parquet  one row per player per team-match, FotMob ids, metres
    network_edges.parquet  one row per passer->receiver pair, FotMob ids
    match_events.parquet   substitutions and red cards, one row per player event
    id_crosswalk.parquet   network-source player ids -> FotMob player ids

Decisions implemented here (see findings log, 2026-09-28):
  * Canonical ids are the player_match / shot ids (FotMob). The network uses three
    other id schemes (source A = 2025/26 R1-25, B = R26-38, C = 2026/27); they are
    mapped via (match_id, team, shirt_number).
  * Blanks in player_match are zeros for players who played (the source omits zeros),
    except: physical stats in matches with no physical data stay missing; xgot,
    shots_on_target, successful_dribbles and big_chances_created are empty in the
    source and are dropped (xgot and shots on target are rebuilt from shots).
  * The shot table is the source of truth for xG, npxG, xGOT and shots on target.
  * The passing network is the full match (all players who appeared).
  * Coordinates in metres on a 105 x 68 pitch, own goal at x = 0, attacking toward
    x = 105, y = 68 on the attacking team's left.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

PITCH_X, PITCH_Y = 105.0, 68.0
TEAM_ALIASES = {
    "Bournemouth": "AFC Bournemouth",
    "Brighton and Hove Albion": "Brighton & Hove Albion",
}

# Counts / values that the source leaves blank when zero.
ZERO_FILL = [
    "goals", "total_shots", "touches_opp_box", "touches", "accurate_passes",
    "passes_attempted", "assists", "xa", "chances_created", "line_breaking_passes",
    "passes_into_final_third", "accurate_crosses", "crosses_attempted", "interceptions",
    "blocks", "recoveries", "clearances", "dribbled_past", "ground_duels_won",
    "aerial_duels_won", "fouls_committed", "was_fouled", "tackles", "red_cards",
]
PHYSICAL_ALWAYS = ["top_speed_kmh", "distance_covered_m", "running_distance_m"]
PHYSICAL_ZERO_OK = ["num_sprints", "sprint_distance_m"]  # blank = 0 when the match has physical data
EMPTY_IN_SOURCE = ["xgot", "shots_on_target", "successful_dribbles", "big_chances_created"]

OPEN_PLAY = {"RegularPlay", "FastBreak", "IndividualPlay"}
SET_PIECE = {"FromCorner", "SetPiece", "ThrowInSetPiece", "FreeKick"}


def _minute_parts(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """'90+3' -> (90, 3); '67' -> (67, 0); NaN stays NaN."""
    s = s.astype("string")
    base = pd.to_numeric(s.str.split("+").str[0], errors="coerce")
    added = pd.to_numeric(s.str.split("+").str[1], errors="coerce").fillna(0).where(s.notna())
    return base, added


def load_raw() -> dict[str, pd.DataFrame]:
    return {
        "pm": pd.read_csv(RAW / "player_match_stats.csv"),
        "shots": pd.read_csv(RAW / "shot_map.csv"),
        "nodes": pd.read_csv(RAW / "pass_network_nodes.csv", dtype=str),
        "edges": pd.read_csv(RAW / "pass_network_edges.csv", dtype=str),
    }


# --------------------------------------------------------------------------- matches
def build_matches(pm: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    g = pm.groupby("match_id")
    m = g.agg(
        season=("season", "first"), round=("round", "first"), date=("match_date_utc", "first"),
        home_team=("home_team", "first"), away_team=("away_team", "first"),
        home_team_id=("home_team_id", "first"), away_team_id=("away_team_id", "first"),
        home_score=("home_score_ft", "first"), away_score=("away_score_ft", "first"),
        home_score_ht=("home_score_ht", "first"), away_score_ht=("away_score_ht", "first"),
    ).reset_index()
    m["date"] = pd.to_datetime(m["date"], utc=True)
    poss = pm.dropna(subset=["team"]).groupby(["match_id", "team"]).team_possession_pct.first()
    m["home_possession"] = [poss.get((r.match_id, r.home_team), np.nan) / 100 for r in m.itertuples()]
    m["away_possession"] = 1 - m["home_possession"]
    phys = pm[pm.minutes_played.notna()].groupby("match_id").distance_covered_m.apply(lambda s: s.notna().any())
    m["has_physical"] = m.match_id.map(phys).fillna(False).astype(bool)
    src = nodes.groupby("match_id").team_id.first()
    src.index = src.index.astype(int)
    m["network_source"] = m.match_id.map(
        lambda i: "A" if str(src[i]).startswith("t") else ("B" if str(src[i]).isdigit() else "C")
    )
    for c in ["home_team_id", "away_team_id", "home_score", "away_score", "home_score_ht", "away_score_ht"]:
        m[c] = m[c].astype(int)
    return m.sort_values(["date", "match_id"]).reset_index(drop=True)


# --------------------------------------------------------------------------- shots
def build_shots(shots: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    s = shots.copy()
    s["date"] = pd.to_datetime(s.match_date_utc, utc=True)
    s = s.rename(columns={"team_name": "team"})
    opp = matches.set_index("match_id")
    s["opponent"] = np.where(s.is_home == 1, s.match_id.map(opp.away_team), s.match_id.map(opp.home_team))
    s["opponent_id"] = np.where(s.is_home == 1, s.match_id.map(opp.away_team_id), s.match_id.map(opp.home_team_id))
    s["minute_added"] = s.minute_added.fillna(0).astype(int)
    s["situation_group"] = np.select(
        [s.situation.eq("Penalty"), s.situation.isin(SET_PIECE), s.situation.isin(OPEN_PLAY)],
        ["penalty", "set_piece", "open_play"], default="other",
    )
    s["is_goal"] = s.outcome.eq("goal").astype(int)
    # FotMob marks blocked shots as on target; strict on-target = goal or saved by the keeper.
    s["on_target_strict"] = ((s.is_on_target == 1) & (s.is_blocked == 0)).astype(int)
    # Team credited with the goal (own goals count for the opponent).
    s["scoring_team_id"] = np.where(s.is_own_goal == 1, s.opponent_id, s.team_id)
    s["x_m"], s["y_m"] = s.x, s.y  # already on 105 x 68
    s["period_order"] = s.period.map({"FirstHalf": 1, "SecondHalf": 2})
    s = s.sort_values(["match_id", "period_order", "minute", "minute_added", "shot_id"])
    keep = ["match_id", "season", "round", "date", "shot_id", "player_id", "player_name", "team_id", "team",
            "opponent_id", "opponent", "is_home", "period", "minute", "minute_added", "x_m", "y_m", "xg", "xgot",
            "shot_type", "situation", "situation_group", "outcome", "is_goal", "is_blocked", "is_on_target",
            "on_target_strict", "is_from_inside_box", "is_own_goal", "scoring_team_id"]
    return s[keep].reset_index(drop=True)


# --------------------------------------------------------------------------- player_match
def build_player_match(pm: pd.DataFrame, shots: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    p = pm[pm.team.notna()].copy()                      # 6 rows with no team and no stats
    p["date"] = pd.to_datetime(p.match_date_utc, utc=True)
    p["played"] = p.minutes_played.notna()
    p["started"] = p.started_or_subbed.eq("started")
    p["is_gk"] = p.starting_position.eq("GK")
    # GK flag for substitute keepers: a player who has started in goal anywhere.
    gk_ids = set(p.loc[p.is_gk, "player_id"])
    p["is_gk"] = p.player_id.isin(gk_ids)
    p["xa_was_blank"] = p.played & p.xa.isna() & p.chances_created.gt(0)
    p["xg_fotmob"] = p.xg  # keep the rounded source value for reference

    played = p.played
    for c in ZERO_FILL:
        p.loc[played, c] = p.loc[played, c].fillna(0)
    has_phys = p.match_id.map(matches.set_index("match_id").has_physical)
    for c in PHYSICAL_ZERO_OK:
        p.loc[played & has_phys, c] = p.loc[played & has_phys, c].fillna(0)
    p = p.drop(columns=EMPTY_IN_SOURCE + ["xg", "npxg"])

    # Shot-based xG family (source of truth). Own goals are excluded from the shooter.
    s = shots[shots.is_own_goal == 0]
    agg = s.groupby(["match_id", "player_id"]).agg(
        xg=("xg", "sum"),
        npxg=("xg", lambda v: v[s.loc[v.index, "situation_group"] != "penalty"].sum()),
        xg_open_play=("xg", lambda v: v[s.loc[v.index, "situation_group"] == "open_play"].sum()),
        xg_set_piece=("xg", lambda v: v[s.loc[v.index, "situation_group"] == "set_piece"].sum()),
        xgot=("xgot", "sum"),
        shots_on_target=("on_target_strict", "sum"),
        headed_shots=("shot_type", lambda v: (v == "Header").sum()),
        penalties_taken=("situation_group", lambda v: (v == "penalty").sum()),
    ).reset_index()
    p = p.merge(agg, on=["match_id", "player_id"], how="left")
    played = p.played
    shot_cols = ["xg", "npxg", "xg_open_play", "xg_set_piece", "xgot", "shots_on_target", "headed_shots", "penalties_taken"]
    p.loc[played, shot_cols] = p.loc[played, shot_cols].fillna(0)

    # Timing: substitutions and red cards, on the 0-90 clock (stoppage time not in minutes_played).
    p["on_minute"] = np.where(p.started, 0.0, np.where(played, 90 - p.minutes_played, np.nan))
    p["off_minute"] = np.where(played, p.on_minute + p.minutes_played, np.nan)
    p["red_card_minute"] = np.where(p.red_cards.gt(0), p.off_minute, np.nan)

    int_cols = [c for c in ZERO_FILL + ["shots_on_target", "headed_shots", "penalties_taken"] if c not in ("xa",)]
    for c in int_cols:
        p[c] = p[c].astype("Int64")
    p["player_id"] = p.player_id.astype(int)
    p["team_id"] = p.team_id.astype(int)
    order = ["match_id", "season", "round", "date", "team_id", "team", "opponent", "player_id", "player_name",
             "shirt_number", "played", "started", "started_or_subbed", "starting_position", "is_gk",
             "minutes_played", "on_minute", "off_minute", "red_cards", "red_card_minute", "team_possession_pct"]
    rest = [c for c in p.columns if c not in order and c not in (
        "match_date_utc", "competition", "home_team", "away_team", "home_team_id", "away_team_id",
        "home_score_ft", "away_score_ft", "home_score_ht", "away_score_ht")]
    return p[order + rest].sort_values(["date", "match_id", "team", "started", "player_name"],
                                       ascending=[True, True, True, False, True]).reset_index(drop=True)


# --------------------------------------------------------------------------- network
def build_network(nodes: pd.DataFrame, edges: pd.DataFrame, pm: pd.DataFrame, matches: pd.DataFrame):
    n = nodes.copy()
    n["match_id"] = n.match_id.astype(int)
    n["network_source"] = n.match_id.map(matches.set_index("match_id").network_source)
    n["team"] = n.team_name.map(lambda t: TEAM_ALIASES.get(t, t))
    n["shirt_number"] = pd.to_numeric(n.shirt_number)
    for c in ["avg_x", "avg_y", "passes_completed", "passes_failed"]:
        n[c] = pd.to_numeric(n[c])

    ref = pm[["match_id", "team", "team_id", "shirt_number", "player_id", "player_name", "played",
              "started", "minutes_played", "on_minute", "off_minute"]].rename(
        columns={"player_id": "fm_player_id", "player_name": "fm_player_name", "team_id": "fm_team_id"})
    n = n.merge(ref, on=["match_id", "team", "shirt_number"], how="left")
    n_unmatched = n.fm_player_id.isna()
    unmatched = n[n_unmatched][["match_id", "team", "player_name", "position_label"]]

    crosswalk = (n[~n_unmatched][["network_source", "match_id", "team_id", "player_id", "fm_team_id", "fm_player_id",
                                  "player_name", "fm_player_name"]]
                 .rename(columns={"team_id": "src_team_id", "player_id": "src_player_id",
                                  "player_name": "src_player_name"}))

    # Keep players who actually appeared (source C lists unused subs with blank coordinates).
    n = n[~n_unmatched & n.played.fillna(False).astype(bool)].copy()
    n["x_m"] = n.avg_x * PITCH_X / 100
    n["y_m"] = n.avg_y * PITCH_Y / 100
    n["sub_on_base"], n["sub_on_added"] = _minute_parts(n.sub_on_minute)
    n["sub_off_base"], n["sub_off_added"] = _minute_parts(n.sub_off_minute)
    nodes_out = n.rename(columns={"fm_player_id": "player_id_fm"})[[
        "match_id", "network_source", "fm_team_id", "team", "player_id_fm", "fm_player_name", "shirt_number",
        "started", "minutes_played", "on_minute", "off_minute", "position_label", "x_m", "y_m",
        "passes_completed", "passes_failed", "sub_on_base", "sub_on_added", "sub_off_base", "sub_off_added",
    ]].rename(columns={"fm_team_id": "team_id", "player_id_fm": "player_id", "fm_player_name": "player_name"})
    nodes_out["team_id"] = nodes_out.team_id.astype(int)
    nodes_out["player_id"] = nodes_out.player_id.astype(int)

    e = edges.copy()
    e["match_id"] = e.match_id.astype(int)
    e["pass_count"] = e.pass_count.astype(int)
    cw = crosswalk.set_index(["match_id", "src_team_id", "src_player_id"])
    key_p = list(zip(e.match_id, e.team_id, e.passer_id))
    key_r = list(zip(e.match_id, e.team_id, e.receiver_id))
    e["passer_id_fm"] = cw.fm_player_id.reindex(key_p).to_numpy()
    e["receiver_id_fm"] = cw.fm_player_id.reindex(key_r).to_numpy()
    e["team_id_fm"] = cw.fm_team_id.reindex(key_p).to_numpy()
    edge_unmapped = int(e[["passer_id_fm", "receiver_id_fm"]].isna().any(axis=1).sum())
    e = e.dropna(subset=["passer_id_fm", "receiver_id_fm"])
    edges_out = pd.DataFrame({
        "match_id": e.match_id, "team_id": e.team_id_fm.astype(int),
        "team": e.team_name.map(lambda t: TEAM_ALIASES.get(t, t)),
        "passer_id": e.passer_id_fm.astype(int), "receiver_id": e.receiver_id_fm.astype(int),
        "pass_count": e.pass_count,
    }).reset_index(drop=True)
    return nodes_out.reset_index(drop=True), edges_out, crosswalk.reset_index(drop=True), unmatched, edge_unmapped


# --------------------------------------------------------------------------- events
def build_match_events(pm: pd.DataFrame, nodes_out: pd.DataFrame) -> pd.DataFrame:
    p = pm[pm.played]
    rows = []
    subs_on = p[~p.started]
    rows.append(pd.DataFrame({"match_id": subs_on.match_id, "team_id": subs_on.team_id,
                              "player_id": subs_on.player_id, "event": "sub_on", "minute": subs_on.on_minute}))
    off = p[p.off_minute.lt(90) & p.red_cards.eq(0)]
    rows.append(pd.DataFrame({"match_id": off.match_id, "team_id": off.team_id,
                              "player_id": off.player_id, "event": "sub_off", "minute": off.off_minute}))
    red = p[p.red_cards.gt(0)]
    rows.append(pd.DataFrame({"match_id": red.match_id, "team_id": red.team_id,
                              "player_id": red.player_id, "event": "red_card", "minute": red.red_card_minute}))
    ev = pd.concat(rows, ignore_index=True)
    # Where the network gives an exact minute with stoppage time (2025/26), attach it.
    nn = nodes_out.set_index(["match_id", "player_id"])
    k = list(zip(ev.match_id, ev.player_id))
    on_add = nn.sub_on_added.reindex(k).to_numpy()
    off_add = nn.sub_off_added.reindex(k).to_numpy()
    ev["minute_added"] = np.where(ev.event.eq("sub_on"), on_add, np.where(ev.event.eq("sub_off"), off_add, np.nan))
    return ev.sort_values(["match_id", "minute", "event"]).reset_index(drop=True)


# --------------------------------------------------------------------------- checks
def validate(t: dict[str, pd.DataFrame]) -> list[str]:
    m, s, p, n, e = t["matches"], t["shots"], t["player_match"], t["network_nodes"], t["network_edges"]
    out = []
    goals = s[s.is_goal == 1].groupby(["match_id", "scoring_team_id"]).size()
    ok = all(goals.get((r.match_id, r.home_team_id), 0) == r.home_score and
             goals.get((r.match_id, r.away_team_id), 0) == r.away_score for r in m.itertuples())
    out.append(f"shot goals reconcile with scores: {ok}")
    out.append(f"possession sums to 1 in every match: {bool(np.allclose(m.home_possession + m.away_possession, 1))}")
    st = p[p.started].groupby(["match_id", "team_id"]).size()
    out.append(f"11 starters per team-match: {bool((st == 11).all())} ({len(st)} team-matches)")
    starters_in_net = n[n.started].groupby(["match_id", "team_id"]).size()
    out.append(f"11 starters in every network: {bool((starters_in_net == 11).all())}")
    diff = n.merge(p[["match_id", "player_id", "accurate_passes"]], on=["match_id", "player_id"])
    d = diff.passes_completed - diff.accurate_passes.astype(float)
    out.append(f"network passes_completed - accurate_passes: median {d.median():.0f}, |diff|<=2 in {(d.abs() <= 2).mean():.1%}")
    out.append(f"edges referencing a node player: {e.set_index(['match_id','passer_id']).index.isin(n.set_index(['match_id','player_id']).index).mean():.1%}")
    out.append(f"x_m range [{n.x_m.min():.1f}, {n.x_m.max():.1f}], y_m range [{n.y_m.min():.1f}, {n.y_m.max():.1f}]")
    played = p[p.played]
    remaining = played[[c for c in ZERO_FILL if c in played]].isna().sum().sum()
    out.append(f"blank counts left among played players (should be 0): {int(remaining)}")
    return out


def main() -> dict[str, pd.DataFrame]:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    matches = build_matches(raw["pm"], raw["nodes"])
    shots = build_shots(raw["shots"], matches)
    pm = build_player_match(raw["pm"], shots, matches)
    nodes, edges, crosswalk, unmatched, edge_unmapped = build_network(raw["nodes"], raw["edges"], pm, matches)
    events = build_match_events(pm, nodes)
    tables = {"matches": matches, "shots": shots, "player_match": pm, "network_nodes": nodes,
              "network_edges": edges, "match_events": events, "id_crosswalk": crosswalk}
    for name, df in tables.items():
        df.to_parquet(OUT / f"{name}.parquet", index=False)
        print(f"{name:14s} {df.shape}")
    print(f"network players unmatched to player_match: {len(unmatched)} (unused subs)")
    print(f"edges dropped for unmapped ids: {edge_unmapped}")
    for line in validate(tables):
        print("check:", line)
    return tables


if __name__ == "__main__":
    main()
