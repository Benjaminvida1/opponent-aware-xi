"""Build the data for the dashboard site in docs/ (GitHub Pages).

Reads the processed tables and reports in this repo and writes compact JSON to docs/data/:
  meta.json            colours, headline results, labels
  matches.json         every match: score, model and market forecasts (test matches)
  match/<id>.json      one match: shots, passing networks, team shapes, lineup calls (test matches)
  players.json         radar values per player-season, model impact
  teams.json           strength over time, season summaries, style dimensions

    python src/build_site.py        # about a minute
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

sys.path.insert(0, str(Path(__file__).resolve().parent))
import viz_colors as C  # noqa: E402
import viz_data as D  # noqa: E402
import viz_export as X  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REP = ROOT / "reports"
OUT = ROOT / "docs" / "data"

SIT = list(D.SITUATIONS)
OUTC = D.OUTCOMES
STYLE_DIMS = [  # (column, name, low end, high end)
    ("style_1", "Control vs direct", "Direct", "Control"),
    ("style_2", "Central vs wide", "Wide", "Central"),
    ("style_3", "Forward reach", "Short", "Long"),
    ("style_4", "Build-up through the keeper", "Rarely", "Often"),
]
NOT_COLLECTED = {"2023/2024": ["line_breaking_passes"], "2024/2025": ["line_breaking_passes"]}
SLOT_NAMES = {"GK": "Goalkeeper", "LB": "Left-back", "RB": "Right-back", "LCB": "Centre-back", "RCB": "Centre-back",
              "CB": "Centre-back", "LWB": "Left wing-back", "RWB": "Right wing-back", "DM": "Defensive mid",
              "LDM": "Defensive mid", "RDM": "Defensive mid", "CM": "Central mid", "LCM": "Central mid",
              "RCM": "Central mid", "LM": "Left mid", "RM": "Right mid", "CAM": "Attacking mid",
              "LAM": "Left attacking mid", "RAM": "Right attacking mid", "LW": "Left wing", "RW": "Right wing",
              "ST": "Striker", "LST": "Striker", "RST": "Striker", "CF": "Striker"}


def r(x, k=2):
    return None if x is None or pd.isna(x) else round(float(x), k)


def dump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
    return path.stat().st_size


# ---------------------------------------------------------------- meta
def build_meta(matches):
    t = {row.model: row.rps for row in pd.concat([pd.read_csv(REP / "l0_test_scores.csv"),
                                                    pd.read_csv(REP / "l23_test_scores.csv")]).itertuples()}
    base, market = t["Base rates"], t["Market (Bet365 closing)"]
    ladder = []
    for key, name, blurb in [
        ("Base rates", "Guessing from history", "Home win, draw and away win at their usual frequencies"),
        ("L0", "Team strength", "Each team's strength from its chances created and conceded"),
        ("L1", "+ who plays", "Adds the players in each lineup, rated from their stats and time on the pitch"),
        ("L2", "+ style clash", "Adds how the two teams' playing styles fit together"),
        ("L3", "+ player matchups", "Adds who faces whom: wingers against full-backs, strikers against centre-backs"),
        ("Market (Bet365 closing)", "Bookmakers", "Closing betting odds: they also know team news and injuries"),
    ]:
        rps = t[key]
        ladder.append({"key": key, "name": name, "blurb": blurb, "rps": r(rps, 4),
                       "share_of_market_gain": r((base - rps) / (base - market), 3)})
    step5 = json.loads((REP / "step5_summary.json").read_text())
    return {
        "clubs": C.all_clubs(),
        "tokens": json.loads(C.write_json().read_text())["tokens"],
        "situations": SIT, "outcomes": OUTC,
        "radar": {"metrics": [{"key": c, "label": lab.replace("\n", " "), "section": sec}
                              for lab, c, sec in D.RADAR_METRICS],
                  "rates": sorted(D.RADAR_METRICS and D.RATE_COLS), "groups": D.GROUP_NAMES},
        "style_dims": [{"key": k, "name": n, "low": lo, "high": hi} for k, n, lo, hi in STYLE_DIMS],
        "ladder": ladder,
        "clash": {a: [b for b in C.CLUBS if b != a and C.delta_e(C.color(a), C.color(b)) < C.CLASH_DE]
                  for a in C.CLUBS},
        "facts": {"matches": int(len(matches)), "seasons": sorted(matches.season.unique().tolist()),
                  "test_matches": 430, "team_selections": int(step5["decisions"]["team_matches"]),
                  "recommended_change": r(step5["decisions"]["recommended_change"], 3),
                  "opponent_changes_best": r(step5["decisions"]["best_differs_from_neutral"], 3),
                  "usual_overlap": 9.0, "recommended_overlap": 8.2,
                  "shots": int(len(D.shots())), "players": int(D.table("player_match").query("played").player_id.nunique())},
    }


# ---------------------------------------------------------------- match list with forecasts
def build_matches():
    m = D.table("matches").copy()
    l1 = pd.read_parquet(PROC / "l1_predictions.parquet").set_index("match_id")[["p_home", "p_draw", "p_away"]]
    od = D.table("odds").set_index("match_id")[["p_home", "p_draw", "p_away"]]
    m = m.join(l1, on="match_id").join(od, on="match_id", rsuffix="_mkt")
    rows = []
    for x in m.sort_values("date").itertuples():
        hc, ac = C.pair_colors(x.home_team, x.away_team)
        rows.append({"id": int(x.match_id), "season": x.season, "col": [hc, ac, C.ink_on(hc), C.ink_on(ac)], "round": int(x.round),
                     "date": pd.Timestamp(x.date).strftime("%Y-%m-%d"), "home": x.home_team, "away": x.away_team,
                     "hs": int(x.home_score), "as": int(x.away_score),
                     "model": None if np.isnan(x.p_home) else [r(x.p_home, 3), r(x.p_draw, 3), r(x.p_away, 3)],
                     "market": None if np.isnan(x.p_home_mkt) else [r(x.p_home_mkt, 3), r(x.p_draw_mkt, 3),
                                                                    r(x.p_away_mkt, 3)]})
    return m, rows


def forecast_stats(rows):
    """How often the favourite won, for the model and the market (test matches)."""
    out = {}
    for k in ["model", "market"]:
        t = [x for x in rows if x["model"] is not None]
        res = [0 if x["hs"] > x["as"] else 1 if x["hs"] == x["as"] else 2 for x in t]
        fav = [int(np.argmax(x[k])) for x in t]
        out[k] = r(np.mean([a == b for a, b in zip(res, fav)]), 3)
    return out


# ---------------------------------------------------------------- per-match files
def lineup_calls():
    d = pd.read_parquet(PROC / "step5_decisions.parquet")
    sw = pd.read_parquet(PROC / "step5_swaps.parquet")
    sw = sw[sw.xi == "best"]
    names = D.table("player_match").drop_duplicates("player_id").set_index("player_id").player_name
    out = {}
    for x in d.itertuples():
        def xi(col, slots_col=None):
            ids = ast.literal_eval(getattr(x, col)) if isinstance(getattr(x, col), str) else list(getattr(x, col))
            slots = json.loads(getattr(x, slots_col)) if slots_col and isinstance(getattr(x, slots_col), str) else {}
            return [{"id": int(i), "name": names.get(int(i), str(i)), "slot": slots.get(str(i))} for i in ids]
        usual, actual = xi("usual_xi", "usual_slots"), xi("actual_xi")
        rec = xi("recommended_xi", "recommended_slots")
        changes = []
        if x.recommended_changes > 0:
            s = sw[(sw.match_id == x.match_id) & (sw.team_id == x.team_id)]
            changes = [{"out": a.out_name, "in": a.in_name, "slot": SLOT_NAMES.get(a.slot, a.slot),
                        "dEP": r(a.dEP, 3), "attack": r(a.dEP_attack, 3), "defence": r(a.dEP_defence, 3)}
                       for a in s.itertuples()]
        out[(int(x.match_id), int(x.team_id))] = {
            "formation": x.formation, "p_better": r(x.p_better, 3),
            "recommended": bool(x.recommended_changes > 0),
            "usual_ep": r(x.usual_ep, 3), "recommended_ep": r(x.recommended_ep, 3), "actual_ep": r(x.actual_ep, 3),
            "usual_win": r(x.usual_p_win, 3), "recommended_win": r(x.recommended_p_win, 3),
            "actual_win": r(x.actual_p_win, 3),
            "usual_overlap_with_actual": int(x.usual_overlap),
            "rec_overlap_with_actual": int(x.recommended_overlap),
            "usual": usual, "recommended_xi": rec, "actual": actual, "changes": changes,
            "points": r(x.points, 0)}
    return out


def build_match_files(match_rows):
    shots = D.shots()
    nodes, links = X.network_tables()
    shapes = D.shape_table().copy()
    shapes["area_pct"] = shapes.area_m2.rank(pct=True, method="average")
    shapes = shapes.set_index(["match_id", "team"])
    calls = lineup_calls()
    team_ids = D.table("network_nodes").drop_duplicates(["match_id", "team"]).set_index(["match_id", "team"]).team_id
    sh_g = dict(tuple(shots.groupby("match_id")))
    nd_g = dict(tuple(nodes.groupby(["match_id", "team"])))
    lk_g = dict(tuple(links.groupby(["match_id", "team"])))
    total = 0
    for mr in match_rows:
        mid = mr["id"]
        teams = {}
        for side in ["home", "away"]:
            team = mr[side]
            s = sh_g.get(mid, shots.iloc[:0])
            s = s[s.team == team]
            n = nd_g[(mid, team)].reset_index(drop=True)
            idx = {pid: i for i, pid in enumerate(n.player_id)}
            lk = lk_g.get((mid, team), links.iloc[:0])
            of = n[~n.is_gk.fillna(False).astype(bool)]
            hull = [int(of.index[i]) for i in ConvexHull(of[["x_m", "y_m"]].to_numpy()).vertices]
            sp = shapes.loc[(mid, team)]
            tid = int(team_ids[(mid, team)])
            teams[side] = {
                "team": team,
                "shots": [[r(a.x_m, 1), r(a.y_m, 1), r(a.xg, 3), OUTC.index(a.outcome_label),
                           SIT.index(a.situation_label), a.player_short, a.minute_label] for a in s.itertuples()],
                "nodes": [[int(a.player_id), a.name, int(a.shirt_number) if not pd.isna(a.shirt_number) else None,
                           r(a.x_m, 1), r(a.y_m, 1), int(a.passes_completed), int(a.passes_failed),
                           int(a.minutes_played), int(bool(a.is_gk))] for a in n.itertuples()],
                "links": [[idx[a.player_a], idx[a.player_b], int(a.pass_count)] for a in lk.itertuples()],
                "shape": {"hull": hull, "area": r(sp.area_m2, 0), "width": r(sp.width_m, 1),
                          "depth": r(sp.depth_m, 1), "cb": r(sp.cb_height_m, 1), "centre": r(sp.centre_x_m, 1),
                          "area_pct": r(sp.area_pct, 3)},
                "call": calls.get((mid, tid)),
            }
        total += dump({"id": mid, "teams": teams}, OUT / "match" / f"{mid}.json")
    return total


# ---------------------------------------------------------------- players
def build_players():
    lg = pd.read_parquet(PROC / "dash_radar.parquet") if (PROC / "dash_radar.parquet").exists() else \
        pd.concat([D.radar_long(s) for s in sorted(D.table("player_match").season.unique())])
    keys = [c for _, c, _ in D.RADAR_METRICS]
    out = []
    for (season, pid), g in lg.groupby(["season", "player_id"]):
        g = g.set_index("metric").loc[keys].copy()
        if season in NOT_COLLECTED:
            g.loc[NOT_COLLECTED[season], ["value", "percentile", "z", "peer_median"]] = np.nan
        f = g.iloc[0]
        out.append({"id": int(pid), "season": season, "name": f["name"], "team": f.team, "group": f.group,
                    "minutes": int(f.minutes), "n": int(f.n_peers),
                    "v": [r(v, 3) for v in g.value], "p": [r(v, 1) for v in g.percentile],
                    "z": [r(v, 2) for v in g.z], "med": [r(v, 3) for v in g.peer_median]})
    impact = {}
    rat = REP / "player_ratings_current.csv"
    if rat.exists():
        rr = pd.read_csv(rat)
        for pid in rr[~rr.is_gk & rr.group.notna()].player_id:
            im = D.player_impact(int(pid))
            if im:
                impact[int(pid)] = {k: (r(v, 4) if isinstance(v, float) else v) for k, v in im.items()}
    return {"players": out, "impact": impact, "impact_note": "L1 player effects as of the end of the data "
            "(365 days before 21 Sep 2026), relative to the average rated player in the same position group."}


# ---------------------------------------------------------------- teams
def build_teams(matches):
    tm = pd.concat([matches[["home_team_id", "home_team"]].set_axis(["team_id", "team"], axis=1),
                    matches[["away_team_id", "away_team"]].set_axis(["team_id", "team"], axis=1)]).drop_duplicates()
    name = tm.set_index("team_id").team
    rat = pd.read_parquet(PROC / "team_ratings_asof.parquet")
    rat["team"] = rat.team_id.map(name)
    style = pd.read_parquet(PROC / "team_style_asof.parquet")
    style["team"] = style.team_id.map(name)
    xp = pd.read_csv(REP / "xpts_by_season.csv")
    cur = pd.read_csv(REP / "team_ratings_current.csv")
    teams = {}
    for team in sorted(name.unique()):
        rt = rat[rat.team == team].sort_values("day")
        st = style[style.team == team].dropna(subset=["style_1"]).sort_values("date")
        seasons = []
        for row in xp[xp.team == team].sort_values("season").itertuples():
            s_end = st[st.season == row.season]
            seasons.append({"season": row.season, "position": int(row.position), "points": int(row.points),
                            "xpts": r(row.xpts, 1), "gf": int(row.goals_for), "ga": int(row.goals_against),
                            "npxg_for": r(row.npxg_for, 1), "npxg_against": r(row.npxg_against, 1),
                            "style": [r(v, 2) for v in s_end.iloc[-1][[k for k, *_ in STYLE_DIMS]]]
                            if len(s_end) else None})
        c = cur[cur.team == team]
        teams[team] = {
            "strength": [[pd.Timestamp(d).strftime("%Y-%m-%d"), r(v, 3)] for d, v in zip(rt.day, rt.rating)],
            "seasons": seasons,
            "style_now": [r(v, 2) for v in st.iloc[-1][[k for k, *_ in STYLE_DIMS]]] if len(st) else None,
            "style_date": pd.Timestamp(st.iloc[-1].date).strftime("%Y-%m-%d") if len(st) else None,
            "current": None if c.empty else {"npxg_for": r(c.npxg_for_per_match.iloc[0], 2),
                                             "npxg_against": r(c.npxg_against_per_match.iloc[0], 2),
                                             "npxg_diff": r(c.npxg_diff_per_match.iloc[0], 2)},
        }
    # league ranges of the style scores (latest per team) for scaling the bars
    return {"teams": teams, "current_order": cur.sort_values("npxg_diff_per_match", ascending=False).team.tolist()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X.build_tables()   # refresh the dash_* tables
    m, rows = build_matches()
    meta = build_meta(m)
    meta["forecast_hit_rate"] = forecast_stats(rows)
    sizes = {"meta.json": dump(meta, OUT / "meta.json"), "matches.json": dump(rows, OUT / "matches.json"),
             "players.json": dump(build_players(), OUT / "players.json"),
             "teams.json": dump(build_teams(m), OUT / "teams.json")}
    sizes["match/*.json"] = build_match_files(rows)
    for k, v in sizes.items():
        print(f"{k:16s} {v / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
