"""Ingest data v2 (2023/24 to 2026/27 R5, 1,190 matches) into the Parquet tables of methodology §2.1.

Run:  python src/ingest.py            (from the repo root)

Outputs (data/processed/):
    matches.parquet        one row per match: season, round, date, teams, scores, possession,
                           teamsheet formations, managers, Opta id, has_physical
    player_match.parquet   one row per squad member per match (unused subs: played = False)
    shots.parquet          one row per shot, metres, situation groups, strict on-target flag
    network_nodes.parquet  one row per player who appeared per match, metres
    network_edges.parquet  one row per passer -> receiver pair per match (completed passes)
    match_events.parquet   substitutions and red cards, one row per player event
    odds.parquet           Bet365 closing 1X2 odds and margin-free implied probabilities
    fixtures.parquet       every fixture of every PL team in all competitions, with rest days
    players.parquet        date of birth, preferred foot, height
    ingest_checks.json     the data checks (also printed)

Rules implemented here (methodology §2.6, findings log):
  * Every file uses FotMob player and team ids; no crosswalk is needed. Two stale FotMob ids
    in 2023/24 (four player rows with no team) are remapped to the ids the network uses;
    minutes and passes agree exactly (see STALE_IDS).
  * Blank counts are zeros for players who appeared (v2 already fills them; kept as a guard).
  * The shot table is the source of truth for xG, npxG, xGOT and shots on target. The source
    player values are kept as xg_fotmob, xgot_fotmob, shots_on_target_fotmob for reference.
  * Physical stats are kept in player_match but not used anywhere in the model.
  * Coordinates: shots arrive in metres (105 x 68); network positions arrive on 0-100 and are
    rescaled. Own goal at x = 0, attacking toward x = 105, y = 68 on the team's left.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

PITCH_X, PITCH_Y = 105.0, 68.0

# Stale FotMob ids in 2023/24. Evidence: the network node with the new id in the same match has
# the same minutes (90 - sub_on) and the same completed passes; the player is listed in the shots
# under the old id or with id -1.
STALE_IDS = {538210: (1396784, "Andre Brooks", "Sheffield United"),
             501284: (1130797, "Michael Olakigbe", "Brentford")}       # listed as "Dakota Ochsenham"
SHOT_NAME_IDS = {"Dakota Ochsenham": 1130797, "Ehor Yarmolyuk": 1157236}   # shots with player_id = -1

COUNTS = [
    "goals", "total_shots", "touches_opp_box", "successful_dribbles", "successful_dribbles_attempted",
    "big_chances_created", "touches", "accurate_passes", "passes_attempted", "assists", "chances_created",
    "line_breaking_passes", "passes_into_final_third", "accurate_crosses", "crosses_attempted",
    "accurate_long_balls", "accurate_long_balls_attempted", "interceptions", "blocks", "recoveries",
    "clearances", "dribbled_past", "ground_duels_won", "ground_duels_attempted", "aerial_duels_won",
    "aerial_duels_attempted", "fouls_committed", "was_fouled", "tackles", "red_cards", "shots_on_target",
]
VALUES = ["xg", "npxg", "xa", "xgot"]
PHYSICAL = ["top_speed_kmh", "distance_covered_m", "num_sprints", "sprint_distance_m", "running_distance_m"]

OPEN_PLAY = {"RegularPlay", "FastBreak", "IndividualPlay"}
SET_PIECE = {"FromCorner", "SetPiece", "ThrowInSetPiece", "FreeKick"}


def _minute_parts(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """'90+3' -> (90, 3); '67' -> (67, 0); blank stays blank."""
    s = s.astype("string")
    base = pd.to_numeric(s.str.split("+").str[0], errors="coerce")
    added = pd.to_numeric(s.str.split("+").str[1], errors="coerce").fillna(0).where(s.notna())
    return base, added


def _decode_formation(v) -> str:
    """Formations saved through Excel become date serials (37714 = 3 Apr 2003 = '3-4-3')."""
    s = str(v)
    if s.isdigit():
        d = pd.Timestamp("1899-12-30") + pd.Timedelta(days=int(s))
        return f"{d.month}-{d.day}-{d.year % 10}"
    return s


def load_raw() -> dict[str, pd.DataFrame]:
    return {
        "matches": pd.read_csv(RAW / "matches.csv", dtype={"opta_id": str, "home_formation": str,
                                                           "away_formation": str}),
        "pm": pd.read_csv(RAW / "player_match_stats.csv"),
        "shots": pd.read_csv(RAW / "shot_map.csv"),
        "nodes": pd.read_csv(RAW / "pass_network_nodes.csv"),
        "edges": pd.read_csv(RAW / "pass_network_edges.csv"),
        "odds": pd.read_csv(RAW / "closing_odds.csv"),
        "fixtures": pd.read_csv(RAW / "fixtures_all_competitions.csv"),
        "players": pd.read_csv(RAW / "players.csv"),
    }


# --------------------------------------------------------------------------- matches
def build_matches(raw_m: pd.DataFrame, pm: pd.DataFrame) -> pd.DataFrame:
    m = raw_m.rename(columns={
        "fotmob_id": "match_id", "match_date_utc": "date", "home_score_ft": "home_score",
        "away_score_ft": "away_score"}).copy()
    m["date"] = pd.to_datetime(m["date"], utc=True)
    m["home_formation"] = m.home_formation.map(_decode_formation)
    m["away_formation"] = m.away_formation.map(_decode_formation)
    m["home_possession"] = m.home_possession_pct / 100
    m["away_possession"] = 1 - m.home_possession
    played = pm[pm.minutes_played.notna()]
    phys = played.groupby("match_id").distance_covered_m.apply(lambda s: s.notna().any())
    m["has_physical"] = m.match_id.map(phys).fillna(False).astype(bool)
    cols = ["match_id", "opta_id", "season", "round", "date", "home_team", "away_team", "home_team_id",
            "away_team_id", "home_score", "away_score", "home_score_ht", "away_score_ht", "home_possession",
            "away_possession", "home_formation", "away_formation", "home_manager", "home_manager_id",
            "away_manager", "away_manager_id", "has_physical"]
    return m[cols].sort_values(["date", "match_id"]).reset_index(drop=True)


def half_time_from_shots(m: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    """The source half-time score omits first-half stoppage-time goals; rebuild it from shots."""
    m = m.rename(columns={"home_score_ht": "home_score_ht_source", "away_score_ht": "away_score_ht_source"})
    g = s[(s.is_goal == 1) & (s.period == "FirstHalf")].groupby(["match_id", "scoring_team_id"]).size()
    m["home_score_ht"] = [int(g.get((r.match_id, r.home_team_id), 0)) for r in m.itertuples()]
    m["away_score_ht"] = [int(g.get((r.match_id, r.away_team_id), 0)) for r in m.itertuples()]
    return m


# --------------------------------------------------------------------------- shots
def build_shots(shots: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    s = shots.copy()
    s["date"] = pd.to_datetime(s.match_date_utc, utc=True)
    s = s.rename(columns={"team_name": "team"})
    mm = matches.set_index("match_id")
    home = s.is_home == 1
    s["opponent"] = np.where(home, s.match_id.map(mm.away_team), s.match_id.map(mm.home_team))
    s["opponent_id"] = np.where(home, s.match_id.map(mm.away_team_id), s.match_id.map(mm.home_team_id))
    s["minute_added"] = s.minute_added.fillna(0).astype(int)
    s["situation_group"] = np.select(
        [s.situation.eq("Penalty"), s.situation.isin(SET_PIECE), s.situation.isin(OPEN_PLAY)],
        ["penalty", "set_piece", "open_play"], default="other")
    s["is_goal"] = s.outcome.eq("goal").astype(int)
    # The source marks blocked shots as on target; strict on target = goal or saved by the keeper.
    s["on_target_strict"] = ((s.is_on_target == 1) & (s.is_blocked == 0)).astype(int)
    s["scoring_team_id"] = np.where(s.is_own_goal == 1, s.opponent_id, s.team_id)
    s["x_m"], s["y_m"] = s.x, s.y          # already metres on 105 x 68 (checked: penalties at x = 94, y = 34)
    s["period_order"] = s.period.map({"FirstHalf": 1, "SecondHalf": 2})
    s = s.sort_values(["match_id", "period_order", "minute", "minute_added", "shot_id"])
    keep = ["match_id", "season", "round", "date", "shot_id", "player_id", "player_name", "team_id", "team",
            "opponent_id", "opponent", "is_home", "period", "minute", "minute_added", "x_m", "y_m", "xg", "xgot",
            "shot_type", "situation", "situation_group", "outcome", "is_goal", "is_blocked", "is_on_target",
            "on_target_strict", "is_from_inside_box", "is_own_goal", "scoring_team_id"]
    return s[keep].reset_index(drop=True)


# --------------------------------------------------------------------------- player_match
def fix_stale_ids(pm: pd.DataFrame, shots: pd.DataFrame, matches: pd.DataFrame):
    """Remap stale ids in player_match and shots; give the orphaned player rows their team."""
    p, s = pm.copy(), shots.copy()
    mm = matches.set_index("match_id")
    for old, (new, name, team) in STALE_IDS.items():
        for k in p.index[p.player_id == old]:
            r = mm.loc[p.at[k, "match_id"]]
            home = r.home_team == team
            p.loc[k, ["player_id", "player_name", "team", "team_id", "opponent"]] = [
                new, name, team, r.home_team_id if home else r.away_team_id, r.away_team if home else r.home_team]
            if pd.isna(p.at[k, "started_or_subbed"]):
                p.at[k, "started_or_subbed"] = "sub"
        k = s.player_id == old
        s.loc[k, ["player_id", "player_name"]] = [new, name]
    for name, pid in SHOT_NAME_IDS.items():
        s.loc[(s.player_id == -1) & (s.player_name == name), "player_id"] = pid
    return p, s


def build_player_match(pm: pd.DataFrame, shots: pd.DataFrame, matches: pd.DataFrame,
                       sub_on: pd.Series | None = None) -> pd.DataFrame:
    p = pm[pm.team.notna()].copy()                       # remaining rows have no team and no stats
    p["date"] = pd.to_datetime(p.match_date_utc, utc=True)
    p["played"] = p.minutes_played.notna()
    p["started"] = p.started_or_subbed.eq("started")
    gk_ids = set(p.loc[p.starting_position.eq("GK"), "player_id"])
    p["is_gk"] = p.player_id.isin(gk_ids)              # includes keepers who only came off the bench
    p["xa_was_blank"] = p.played & p.xa.isna() & p.chances_created.gt(0)
    p = p.rename(columns={"xg": "xg_fotmob", "npxg": "npxg_fotmob", "xgot": "xgot_fotmob",
                          "shots_on_target": "shots_on_target_fotmob"})

    played = p.played
    guard = [c for c in COUNTS + VALUES if c in p and c not in ("shots_on_target", "xgot", "xg", "npxg")]
    guard += ["xg_fotmob", "npxg_fotmob", "xgot_fotmob", "shots_on_target_fotmob"]
    p.loc[played, guard] = p.loc[played, guard].fillna(0)

    # Shot-based xG family (source of truth). Own goals are not credited to the shooter.
    s = shots[shots.is_own_goal == 0].copy()
    s["npxg"] = s.xg.where(s.situation_group != "penalty", 0.0)
    s["xg_open_play"] = s.xg.where(s.situation_group == "open_play", 0.0)
    s["xg_set_piece"] = s.xg.where(s.situation_group == "set_piece", 0.0)
    s["xgot_strict"] = s.xgot.fillna(0).where(s.on_target_strict == 1, 0.0)
    s["header"] = s.shot_type.eq("Header").astype(int)
    s["pen"] = s.situation_group.eq("penalty").astype(int)
    agg = s.groupby(["match_id", "player_id"]).agg(
        xg=("xg", "sum"), npxg=("npxg", "sum"), xg_open_play=("xg_open_play", "sum"),
        xg_set_piece=("xg_set_piece", "sum"), xgot=("xgot_strict", "sum"),
        shots_on_target=("on_target_strict", "sum"), headed_shots=("header", "sum"),
        penalties_taken=("pen", "sum")).reset_index()
    p = p.merge(agg, on=["match_id", "player_id"], how="left")
    shot_cols = ["xg", "npxg", "xg_open_play", "xg_set_piece", "xgot", "shots_on_target", "headed_shots",
                 "penalties_taken"]
    p.loc[p.played, shot_cols] = p.loc[p.played, shot_cols].fillna(0)

    # Timing on the 0-90 clock (minutes_played excludes stoppage time, §2.6 rule 8).
    p["on_minute"] = np.where(p.started, 0.0, np.where(p.played, 90 - p.minutes_played, np.nan))
    # Substitutes who were later taken off (or sent off, or injured with no replacement left) played
    # fewer than 90 - on minutes. Where the network records the sub-on minute (2023/24 to 2025/26),
    # use it when it disagrees with 90 - minutes_played by more than 2 minutes.
    if sub_on is not None:
        net_on = pd.MultiIndex.from_frame(p[["match_id", "player_id"]]).map(sub_on.to_dict().get)
        net_on = pd.to_numeric(pd.Series(net_on, index=p.index), errors="coerce").clip(upper=90)
        fix = ~p.started & p.played & net_on.notna() & (p.on_minute - net_on).abs().gt(2)
        p.loc[fix, "on_minute"] = net_on[fix]
        p["on_minute_from_network"] = fix
    p["off_minute"] = np.where(p.played, p.on_minute + p.minutes_played, np.nan)
    p["red_card_minute"] = np.where(p.red_cards.gt(0), p.off_minute, np.nan)

    for c in [c for c in COUNTS if c != "shots_on_target"] + ["shots_on_target", "headed_shots",
                                                             "penalties_taken", "shots_on_target_fotmob"]:
        p[c] = p[c].round().astype("Int64")
    for c in ["player_id", "team_id"]:
        p[c] = p[c].astype(int)
    order = ["match_id", "season", "round", "date", "team_id", "team", "opponent", "player_id", "player_name",
             "shirt_number", "played", "started", "started_or_subbed", "starting_position", "is_gk",
             "minutes_played", "on_minute", "off_minute", "red_cards", "red_card_minute", "team_possession_pct"]
    drop = {"match_date_utc", "competition", "home_team", "away_team", "home_team_id", "away_team_id",
            "home_score_ft", "away_score_ft", "home_score_ht", "away_score_ht"}
    rest = [c for c in p.columns if c not in order and c not in drop]
    return p[order + rest].sort_values(["date", "match_id", "team", "started", "player_name"],
                                       ascending=[True, True, True, False, True]).reset_index(drop=True)


# --------------------------------------------------------------------------- network
def build_network(nodes: pd.DataFrame, edges: pd.DataFrame, pm: pd.DataFrame):
    n = nodes.dropna(subset=["player_id"]).copy()       # one unused 2026/27 youth sub has no FotMob id
    n["player_id"] = n.player_id.astype(int)
    ref = pm[["match_id", "player_id", "team_id", "played", "started", "minutes_played", "on_minute",
              "off_minute"]]
    n = n.merge(ref, on=["match_id", "player_id", "team_id"], how="left")
    n = n[n.played.fillna(False).astype(bool)].copy()   # 2026/27 lists unused subs with blank coordinates
    n["x_m"] = n.avg_x * PITCH_X / 100
    n["y_m"] = n.avg_y * PITCH_Y / 100
    n["sub_on_base"], n["sub_on_added"] = _minute_parts(n.sub_on_minute)
    n["sub_off_base"], n["sub_off_added"] = _minute_parts(n.sub_off_minute)
    n = n.rename(columns={"team_name": "team"})
    nodes_out = n[["match_id", "team_id", "team", "player_id", "player_name", "shirt_number", "started",
                   "minutes_played", "on_minute", "off_minute", "position_label", "x_m", "y_m",
                   "passes_completed", "passes_failed", "sub_on_base", "sub_on_added", "sub_off_base",
                   "sub_off_added"]].reset_index(drop=True)
    nodes_out["started"] = nodes_out.started.astype(bool)

    e = edges.rename(columns={"team_name": "team"})[["match_id", "team_id", "team", "passer_id",
                                                     "receiver_id", "pass_count"]].copy()
    return nodes_out, e.reset_index(drop=True)


# --------------------------------------------------------------------------- events
def build_match_events(pm: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    p = pm[pm.played]
    subs_on = p[~p.started]
    off = p[p.off_minute.lt(90) & p.red_cards.eq(0)]
    red = p[p.red_cards.gt(0)]
    ev = pd.concat([
        pd.DataFrame({"match_id": subs_on.match_id, "team_id": subs_on.team_id, "player_id": subs_on.player_id,
                      "event": "sub_on", "minute": subs_on.on_minute}),
        pd.DataFrame({"match_id": off.match_id, "team_id": off.team_id, "player_id": off.player_id,
                      "event": "sub_off", "minute": off.off_minute}),
        pd.DataFrame({"match_id": red.match_id, "team_id": red.team_id, "player_id": red.player_id,
                      "event": "red_card", "minute": red.red_card_minute}),
    ], ignore_index=True)
    nn = nodes.set_index(["match_id", "player_id"])
    k = pd.MultiIndex.from_arrays([ev.match_id, ev.player_id])
    on_add = nn.sub_on_added.reindex(k).to_numpy()
    off_add = nn.sub_off_added.reindex(k).to_numpy()
    ev["minute_added"] = np.where(ev.event.eq("sub_on"), on_add, np.where(ev.event.eq("sub_off"), off_add, np.nan))
    return ev.sort_values(["match_id", "minute", "event"]).reset_index(drop=True)


# --------------------------------------------------------------------------- odds, fixtures, players
def build_odds(odds: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    o = matches[["match_id", "season", "date", "home_team", "away_team"]].merge(
        odds.drop(columns=["date"]), on=["season", "home_team", "away_team"], how="left")
    inv = 1 / o[["closing_home_odds", "closing_draw_odds", "closing_away_odds"]].to_numpy()
    o["overround"] = inv.sum(1)
    # Proportional normalisation: the simplest margin removal; enough for a reference benchmark.
    o[["p_home", "p_draw", "p_away"]] = inv / inv.sum(1, keepdims=True)
    return o[["match_id", "closing_home_odds", "closing_draw_odds", "closing_away_odds", "odds_source",
              "overround", "p_home", "p_draw", "p_away"]]


def build_fixtures(fx: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    # The 2025/26 cup rounds appear twice (once under each season label); keep one row.
    f = fx.drop_duplicates(["team", "date_utc", "competition", "opponent"]).rename(columns={"date_utc": "date"})
    f["date"] = pd.to_datetime(f.date, utc=True)
    ids = pd.concat([matches[["home_team", "home_team_id"]].set_axis(["team", "team_id"], axis=1),
                     matches[["away_team", "away_team_id"]].set_axis(["team", "team_id"], axis=1)]
                    ).drop_duplicates().set_index("team").team_id
    f["team_id"] = f.team.map(ids).astype("Int64")
    # Link league fixtures to match_id (one 2024/25 kick-off differs by one second).
    home = matches[["match_id", "season", "home_team", "away_team"]]
    key_h = home.rename(columns={"home_team": "team", "away_team": "opponent"}).assign(home_away="home")
    key_a = home.rename(columns={"away_team": "team", "home_team": "opponent"}).assign(home_away="away")
    keys = pd.concat([key_h, key_a]).assign(competition="Premier League")
    n0 = len(f)
    f = f.merge(keys, on=["season", "competition", "team", "opponent", "home_away"], how="left")
    assert len(f) == n0
    f["match_id"] = f.match_id.astype("Int64")
    f = f.sort_values(["team", "date"]).reset_index(drop=True)
    g = f.groupby("team").date
    f["rest_days"] = (f.date - g.shift(1)).dt.total_seconds() / 86400
    # Fixtures in the 14 days before kick-off (any competition).
    f["fixtures_prev_14d"] = 0
    for t, idx in f.groupby("team").groups.items():
        d = f.loc[idx, "date"].dt.tz_convert(None).to_numpy(dtype="datetime64[ns]")
        lo = np.searchsorted(d, d - np.timedelta64(14, "D"), side="left")
        f.loc[idx, "fixtures_prev_14d"] = np.arange(len(d)) - lo
    return f[["team", "team_id", "season", "date", "competition", "opponent", "home_away", "match_id",
              "rest_days", "fixtures_prev_14d"]]


def build_players(players: pd.DataFrame) -> pd.DataFrame:
    p = players.rename(columns={"name": "player_name"}).copy()
    p["date_of_birth"] = pd.to_datetime(p.date_of_birth)
    return p


# --------------------------------------------------------------------------- checks
def validate(t: dict[str, pd.DataFrame], raw: dict[str, pd.DataFrame]) -> dict:
    m, s, p, n, e = t["matches"], t["shots"], t["player_match"], t["network_nodes"], t["network_edges"]
    c: dict = {}
    c["matches"] = int(len(m))
    c["matches_by_season"] = m.season.value_counts().sort_index().to_dict()
    ids = set(m.match_id)
    c["every_table_covers_every_match"] = {k: set(t[k].match_id) == ids for k in
                                           ["player_match", "shots", "network_nodes", "network_edges", "odds"]}
    goals = s[s.is_goal == 1].groupby(["match_id", "scoring_team_id"]).size()
    c["shot_goals_match_final_scores"] = int(sum(
        goals.get((r.match_id, r.home_team_id), 0) == r.home_score and
        goals.get((r.match_id, r.away_team_id), 0) == r.away_score for r in m.itertuples()))
    st = p[p.started].groupby(["match_id", "team_id"]).size()
    c["team_matches_with_11_starters"] = f"{int((st == 11).sum())} of {len(st)}"
    sq = p.groupby(["match_id", "team_id"]).size()
    c["squad_size_counts"] = {int(k): int(v) for k, v in sq.value_counts().sort_index().items()}
    c["player_match_rows_dropped (no team, no stats)"] = int(len(raw["pm"]) - len(p))
    c["source_half_time_score_agrees_with_shots"] = int((m.home_score_ht_source.eq(m.home_score_ht) & m.away_score_ht_source.eq(m.away_score_ht)).sum())
    played = p[p.played]
    num = [x for x in COUNTS + ["xa", "xg", "npxg", "xgot"] if x in played]
    c["blank_counts_among_players_who_appeared"] = int(played[num].isna().sum().sum())
    c["players_who_appeared"] = int(len(played))
    c["unused_subs"] = int((~p.played).sum())
    c["red_cards"] = int(p.red_cards.fillna(0).sum())
    c["matches_with_physical_data"] = m.groupby("season").has_physical.sum().astype(int).to_dict()

    # Joins.
    pk = pd.MultiIndex.from_frame(p[["match_id", "player_id"]])
    c["shots_joined_to_player_match"] = float(pd.MultiIndex.from_frame(
        s[["match_id", "player_id"]]).isin(pk).mean())
    raw_n = raw["nodes"].dropna(subset=["player_id"])
    c["raw_network_nodes_joined_to_player_match"] = float(pd.MultiIndex.from_arrays(
        [raw_n.match_id, raw_n.player_id.astype(int)]).isin(pk).mean())
    nk = pd.MultiIndex.from_frame(n[["match_id", "player_id"]])
    c["players_who_appeared_with_a_node"] = f"{int(pd.MultiIndex.from_frame(played[['match_id','player_id']]).isin(nk).sum())} of {len(played)}"
    c["nodes_without_coordinates (appeared, no passes)"] = int(n.x_m.isna().sum())
    sn = n[n.started].groupby(["match_id", "team_id"]).size()
    c["networks_with_11_starters"] = f"{int((sn == 11).sum())} of {len(sn)}"
    c["edges_with_both_players_as_nodes"] = float(
        pd.MultiIndex.from_frame(e[["match_id", "passer_id"]]).isin(nk).mean() *
        pd.MultiIndex.from_frame(e[["match_id", "receiver_id"]]).isin(nk).mean())
    out = e.groupby(["match_id", "passer_id"]).pass_count.sum()
    nn = n.set_index(["match_id", "player_id"]).passes_completed
    d = (nn - out.reindex(nn.index).fillna(0))
    c["edge_out_sum_vs_node_passes_within_2"] = float((d.abs() <= 2).mean())
    j = n.merge(p[["match_id", "player_id", "accurate_passes"]], on=["match_id", "player_id"])
    d = j.passes_completed - j.accurate_passes.astype(float)
    c["node_passes_vs_player_accurate_passes"] = dict(median=float(d.median()), within_2=float((d.abs() <= 2).mean()))

    # Coordinates and orientation.
    c["node_x_m_range"] = [float(n.x_m.min()), float(n.x_m.max())]
    c["node_y_m_range"] = [float(n.y_m.min()), float(n.y_m.max())]
    lab = p[p.started][["match_id", "player_id", "starting_position"]].merge(n, on=["match_id", "player_id"])
    c["mean_node_y_m_by_label"] = lab.groupby("starting_position").y_m.mean().round(1).reindex(
        ["LB", "LCB", "RCB", "RB", "LW", "RW", "GK", "ST"]).to_dict()
    c["penalty_shot_median_xy_m"] = s[s.situation == "Penalty"][["x_m", "y_m"]].median().round(1).to_dict()
    c["starters_missing_position_label"] = int(p[p.started].starting_position.isna().sum())

    # Timing.
    sub_net = n[~n.started & n.sub_on_base.notna()]
    gap = (sub_net.on_minute - sub_net.sub_on_base.clip(upper=90)).abs()
    c["sub_on_minute_within_1_of_network (where network has it)"] = float(gap.le(1).mean())
    c["sub_on_minute_taken_from_network"] = int(p.get("on_minute_from_network", pd.Series(False)).sum())

    # Odds, fixtures, players.
    o = t["odds"]
    c["odds_matched"] = int(o.closing_home_odds.notna().sum())
    c["odds_overround_median"] = float(o.overround.median())
    f = t["fixtures"]
    c["league_matches_in_fixtures_for_both_teams"] = int((f.match_id.dropna().value_counts() == 2).sum())
    c["fixtures_by_competition"] = f.competition.value_counts().to_dict()
    c["fixture_duplicates_dropped"] = int(len(raw["fixtures"]) - len(f))
    pl = t["players"]
    c["player_match_ids_in_players_table"] = float(p.player_id.drop_duplicates().isin(pl.player_id).mean())
    c["players_missing"] = pl[["date_of_birth", "preferred_foot", "height_cm"]].isna().mean().round(3).to_dict()
    c["formations_valid"] = bool(pd.concat([m.home_formation, m.away_formation]).str.fullmatch(r"\d(-\d)+").all())
    return c


def main() -> dict[str, pd.DataFrame]:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    pm_raw, shots_raw = fix_stale_ids(raw["pm"], raw["shots"], raw["matches"].rename(columns={"fotmob_id": "match_id"}))
    matches = build_matches(raw["matches"], pm_raw)
    shots = build_shots(shots_raw, matches)
    matches = half_time_from_shots(matches, shots)
    raw_n = raw["nodes"].dropna(subset=["player_id"])
    sub_on = _minute_parts(raw_n.sub_on_minute)[0]
    sub_on.index = pd.MultiIndex.from_arrays([raw_n.match_id, raw_n.player_id.astype(int)])
    pm = build_player_match(pm_raw, shots, matches, sub_on.dropna())
    nodes, edges = build_network(raw["nodes"], raw["edges"], pm)
    events = build_match_events(pm, nodes)
    tables = {"matches": matches, "shots": shots, "player_match": pm, "network_nodes": nodes,
              "network_edges": edges, "match_events": events, "odds": build_odds(raw["odds"], matches),
              "fixtures": build_fixtures(raw["fixtures"], matches), "players": build_players(raw["players"])}
    for old in ["id_crosswalk.parquet"]:
        (OUT / old).unlink(missing_ok=True)
    for name, df in tables.items():
        df.to_parquet(OUT / f"{name}.parquet", index=False)
        print(f"{name:14s} {df.shape}")
    checks = validate(tables, raw)
    (OUT / "ingest_checks.json").write_text(json.dumps(checks, indent=2, default=str))
    for k, v in checks.items():
        print(f"check: {k}: {v}")
    return tables


if __name__ == "__main__":
    main()
