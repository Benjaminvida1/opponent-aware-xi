"""Checks for the visuals (viz_colors, viz_data, viz_plots, viz_radar, viz_export). Run from the repo root:
python tests/test_viz.py

Needs the processed tables from src/ingest.py.
"""
from __future__ import annotations

import itertools
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")
import viz_colors as C  # noqa: E402
import viz_data as D  # noqa: E402

MATCH, TEAM = 5795435, "Arsenal"   # Arsenal 2-1 Chelsea, 2026/27 R3


def test_colours():
    teams = set(D.table("matches").home_team)
    assert teams <= set(C.CLUBS), f"clubs without colours: {teams - set(C.CLUBS)}"
    for t in C.CLUBS:
        c = C.club(t)
        assert C.contrast(c["chart"], C.SURFACE) >= C.MIN_CONTRAST - 0.05, t
        assert C.contrast(c["chart_alt"], C.SURFACE) >= C.MIN_CONTRAST - 0.05, t
        assert C.contrast(c["ink"], c["chart"]) >= 3.0, t
    for a, b in itertools.permutations(C.CLUBS, 2):
        h, w = C.pair_colors(a, b)
        assert h == C.color(a) and C.delta_e(h, w) >= C.CLASH_DE, (a, b)
    assert C.pair_colors("Arsenal", "Liverpool")[1] != C.color("Liverpool")   # red v red switches


def test_shots():
    s = D.shots()
    raw = D.table("shots")
    assert len(s) == (~raw.is_own_goal.astype(bool)).sum()
    assert set(s.situation_label) == set(D.SITUATIONS) and set(s.outcome_label) == set(D.OUTCOMES)
    # the situation filter partitions the shots
    parts = sum(len(D.shots(MATCH, TEAM, [k])) for k in D.SITUATIONS)
    assert parts == len(D.shots(MATCH, TEAM))
    # goals (non-own) per team-match agree with the processed shots table
    g = s[s.outcome_label == "Goal"].groupby(["match_id", "team"]).size()
    g2 = raw[raw.is_goal.astype(bool) & ~raw.is_own_goal.astype(bool)].groupby(["match_id", "team"]).size()
    assert g.sort_index().equals(g2.sort_index())
    # blocked shots are never counted on target
    sm = D.shot_summary(D.shots(MATCH, TEAM))
    assert sm == {**sm, "shots": 16, "goals": 2, "on_target": 9, "blocked": 6}
    try:
        D.shots(MATCH, TEAM, ["Corners"])
        raise AssertionError("unknown situation accepted")
    except ValueError:
        pass


def test_network():
    net = D.network(MATCH, TEAM)
    n, links = net["nodes"], net["links"]
    assert len(n) == 11 and n.is_gk.sum() == 1
    e = D.table("network_edges")
    e = e[(e.match_id == MATCH) & (e.team == TEAM) & e.passer_id.isin(n.player_id) & e.receiver_id.isin(n.player_id)]
    assert links.pass_count.sum() == e.pass_count.sum()          # both directions combined, nothing lost
    assert (links.a < links.b).all() and not links.duplicated(["a", "b"]).any()
    assert net["summary"]["hub"] == "Rice" and net["summary"]["top_link_passes"] == 13
    # all team-matches: 11 starters each
    nodes = pd.read_parquet(D.PROC / "dash_network_nodes.parquet") if (D.PROC / "dash_network_nodes.parquet").exists() else None
    if nodes is not None:
        assert (nodes.groupby(["match_id", "team"]).size() == 11).all()


def test_shape():
    sh = D.team_shape(MATCH, TEAM)
    of = sh["nodes"][~sh["nodes"].is_gk]
    assert len(of) == 10 and 3 <= len(sh["hull"]) <= 10
    assert 0 < sh["width_m"] <= 68 and 0 < sh["depth_m"] <= 105 and 0 <= sh["area_pct"] <= 1
    # area of the hull polygon (shoelace) equals the reported area
    x, y = sh["hull"].x_m.to_numpy(), sh["hull"].y_m.to_numpy()
    assert abs(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) - sh["area_m2"]) < 1e-6
    t = D.shape_table()
    assert len(t) == 2 * len(D.table("matches")) and (t.width_m <= 68).all()


def test_radar():
    lg = D.radar_long("2025/2026")
    assert lg.percentile.between(0, 100).all()
    z = lg.groupby(["group", "metric"]).z
    assert (z.mean().abs() < 1e-9).all() and ((z.std() - 1).abs() < 1e-9).all()
    r = D.player_radar("2025/2026", "Declan Rice")
    assert r["group"] == "CM" and len(r["metrics"]) == len(D.RADAR_METRICS)
    # percentile = share of peers below + half the ties
    w = D.radar_table("2025/2026")
    peers = w[w.group == "CM"]
    me = peers[peers.name == "Declan Rice"].iloc[0]
    exp = ((peers.xa < me.xa).mean() + 0.5 * (peers.xa == me.xa).mean()) * 100
    got = r["metrics"].set_index("metric").loc["xa", "percentile"]
    assert abs(exp - got) < 1e-9
    imp = D.player_impact("Bukayo Saka")
    assert imp is not None and imp["group"] == "W" and 0 < imp["net_pct"] < 100


def test_figures_and_export():
    import viz_export as X
    import viz_plots as P
    import viz_radar as R
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for f in [P.shot_map(MATCH, TEAM, out=d / "a.png")[0],
                  P.shot_map(MATCH, TEAM, ["Set piece"], out=d / "b.png")[0],
                  P.passing_network(MATCH, TEAM, out=d / "c.png")[0],
                  P.convex_hull(MATCH, TEAM, out=d / "d.png")[0],
                  R.radar_figure("2025/2026", ["Declan Rice", "Bukayo Saka"], "zscore", out=d / "e.png")[0]]:
            assert f.exists() and f.stat().st_size > 50_000
    p = X.match_payload(MATCH)
    assert set(p["teams"]) == {"Arsenal", "Chelsea"}
    a, c = (p["teams"][t]["colour"] for t in ("Arsenal", "Chelsea"))
    assert C.delta_e(a, c) >= C.CLASH_DE


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("all viz checks passed")
