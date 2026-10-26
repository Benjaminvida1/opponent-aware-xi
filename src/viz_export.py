"""Dashboard tables and example payloads for the visuals.

Writes (regenerated, not committed):
  data/processed/dash_shots.parquet          every shot with situation / outcome labels (own goals dropped)
  data/processed/dash_network_nodes.parquet  starting XI per team-match at average positions, display names
  data/processed/dash_network_links.parquet  links between starters, both directions combined
  data/processed/dash_shapes.parquet         convex-hull shape metrics per team-match (+ area percentile)
  data/processed/dash_radar.parquet          radar metrics per player-season: value, percentile, z, peer stats
  data/processed/dash_impact.parquet         L1 player impact vs position group (from player_ratings_current.csv)
Writes (committed, small):
  reports/dashboard/team_colors.json          club colours and chart tokens
  reports/dashboard/example_match_<id>.json   everything one match view needs, both teams
  reports/dashboard/example_radar.json        one radar payload (both modes + impact)

    python src/viz_export.py                 # all tables + examples (about 30 seconds)
    python src/viz_export.py --match 5795435
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import viz_colors as C  # noqa: E402
import viz_data as D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
DASH = ROOT / "reports" / "dashboard"


def _records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records", force_ascii=False))


# ---------------------------------------------------------------- full tables
def network_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    n = D.table("network_nodes")
    n = n[n.started].drop(columns=["player_name"])
    pm = D.table("player_match")[["match_id", "player_id", "player_name", "starting_position", "is_gk"]]
    n = n.merge(pm, on=["match_id", "player_id"], how="left")
    n["name"] = n.player_name.map(D.surname)
    n["subbed_off"] = n.minutes_played < 89
    nodes = n[["match_id", "team_id", "team", "player_id", "player_name", "name", "shirt_number",
               "starting_position", "is_gk", "x_m", "y_m", "passes_completed", "passes_failed",
               "minutes_played", "subbed_off"]].reset_index(drop=True)

    e = D.table("network_edges")
    key = nodes[["match_id", "team_id", "player_id"]]
    e = e.merge(key.rename(columns={"player_id": "passer_id"}), on=["match_id", "team_id", "passer_id"])
    e = e.merge(key.rename(columns={"player_id": "receiver_id"}), on=["match_id", "team_id", "receiver_id"])
    e["a"] = e[["passer_id", "receiver_id"]].min(axis=1)
    e["b"] = e[["passer_id", "receiver_id"]].max(axis=1)
    links = e.groupby(["match_id", "team_id", "team", "a", "b"], as_index=False).pass_count.sum()
    pos = nodes.set_index(["match_id", "player_id"])[["x_m", "y_m"]]
    links = links.join(pos, on=["match_id", "a"]).join(pos, on=["match_id", "b"], rsuffix="_b")
    return nodes, links.rename(columns={"a": "player_a", "b": "player_b"})


def build_tables() -> dict[str, pd.DataFrame]:
    out = {"dash_shots": D.shots()}
    out["dash_network_nodes"], out["dash_network_links"] = network_tables()
    sh = D.shape_table().copy()
    sh["area_pct"] = sh.area_m2.rank(pct=True, method="average")
    out["dash_shapes"] = sh
    seasons = sorted(D.table("player_match").season.unique())
    out["dash_radar"] = pd.concat([D.radar_long(s) for s in seasons], ignore_index=True)
    rat = ROOT / "reports" / "player_ratings_current.csv"
    if rat.exists():
        r = pd.read_csv(rat)
        r = r[~r.is_gk & r.group.notna()]
        rows = [{"player_id": int(pid), "player_name": nm, "team": tm, **D.player_impact(int(pid))}
                for pid, nm, tm in zip(r.player_id, r.player_name, r.team)]
        out["dash_impact"] = pd.DataFrame(rows)
    for name, df in out.items():
        df.to_parquet(PROC / f"{name}.parquet", index=False)
    return out


# ---------------------------------------------------------------- example payloads
def match_payload(match_id: int) -> dict:
    m = D.match_info(match_id)
    home_c, away_c = C.pair_colors(m["home_team"], m["away_team"])
    teams = {}
    for team, colour in [(m["home_team"], home_c), (m["away_team"], away_c)]:
        s = D.shots(match_id, team)
        net = D.network(match_id, team)
        sh = D.team_shape(match_id, team)
        teams[team] = {
            "colour": colour, "colour_tokens": C.club(team),
            "shots": _records(s.drop(columns=["season"])),
            "shot_summary_by_situation": {k: D.shot_summary(s[s.situation_label == k]) for k in D.SITUATIONS},
            "shot_summary": D.shot_summary(s),
            "network": {"nodes": _records(net["nodes"]), "links": _records(net["links"]),
                        "summary": net["summary"]},
            "shape": {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in sh.items()
                      if k not in ("hull", "nodes")} | {"hull": _records(sh["hull"])},
        }
    return {"match": m, "situations": list(D.SITUATIONS), "outcomes": D.OUTCOMES, "teams": teams}


def radar_payload(season: str, player: str) -> dict:
    r = D.player_radar(season, player)
    return {k: v for k, v in r.items() if k != "metrics"} | {
        "colour": C.color(r["team"]), "metrics": _records(r["metrics"]),
        "labels": {c: lab.replace("\n", " ") for lab, c, _ in D.RADAR_METRICS},
        "impact": D.player_impact(r["player_id"])}


def main(match_id: int = 5795435, season: str = "2025/2026", player: str = "Declan Rice", tables: bool = True):
    DASH.mkdir(parents=True, exist_ok=True)
    print("wrote", C.write_json())
    if tables:
        for name, df in build_tables().items():
            print(f"wrote data/processed/{name}.parquet  {df.shape}")
    p = DASH / f"example_match_{match_id}.json"
    p.write_text(json.dumps(match_payload(match_id), indent=1, ensure_ascii=False, default=str))
    print("wrote", p)
    p = DASH / "example_radar.json"
    p.write_text(json.dumps(radar_payload(season, player), indent=1, ensure_ascii=False, default=str))
    print("wrote", p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", type=int, default=5795435)
    ap.add_argument("--season", default="2025/2026")
    ap.add_argument("--player", default="Declan Rice")
    ap.add_argument("--no-tables", action="store_true")
    a = ap.parse_args()
    main(a.match, a.season, a.player, not a.no_tables)
