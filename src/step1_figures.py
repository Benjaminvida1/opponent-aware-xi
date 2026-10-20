"""Figures for the Step 1 memo and checkpoint (reports/figures/). Run after network.py and style.py.

  formations_<match>.png   both teams' passing networks (starters), slot from the teamsheet formation,
                           shape in possession from the network; one file per showcase match
  style_loadings.png       what each style dimension is made of
  style_map.png            teams on the first two style dimensions (as of the latest match)
  style_prediction.png     how well each metric is predicted before kick-off: league average, own style only,
                           + strength gap, + style imposed by the opponent (test seasons)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from style import STYLE_NAMES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REP = ROOT / "reports"
FIG = REP / "figures"
INK, MUTED, GRID = "#1f2328", "#6e7781", "#d0d7de"
C_D, C_M, C_F = "#2f6fbd", "#2a9d6f", "#c2571a"
SHOWCASE = {
    4813480: "Man City 3-0 Liverpool (2025/26 R11)",
    4813486: "Arsenal 4-1 Tottenham (2025/26 R12)",
    4813449: "Liverpool 1-2 Man United (2025/26 R8)",
    4813701: "Man City 2-1 Arsenal (2025/26 R33)",
    5795435: "Arsenal 2-1 Chelsea (2026/27 R3)",
    4813543: "Tottenham 1-2 Liverpool (2025/26 R17)",
}


def pitch(ax):
    ax.set_xlim(-2, 107)
    ax.set_ylim(-2, 70)
    ax.set_aspect("equal")
    ax.axis("off")
    kw = dict(color=GRID, lw=1, zorder=0)
    ax.plot([0, 105, 105, 0, 0], [0, 0, 68, 68, 0], **kw)
    ax.plot([52.5, 52.5], [0, 68], **kw)
    ax.add_patch(plt.Circle((52.5, 34), 9.15, fill=False, **kw))
    for x0, s in [(0, 1), (105, -1)]:
        ax.plot([x0, x0 + s * 16.5, x0 + s * 16.5, x0], [13.85, 13.85, 54.15, 54.15], **kw)
        ax.plot([x0, x0 + s * 5.5, x0 + s * 5.5, x0], [24.85, 24.85, 43.15, 43.15], **kw)


def formation_figures():
    nodes = pd.read_parquet(PROC / "network_nodes.parquet")
    edges = pd.read_parquet(PROC / "network_edges.parquet")
    slots = pd.read_parquet(PROC / "player_slots.parquet")
    tm = pd.read_parquet(PROC / "team_match_network.parquet")
    pm = pd.read_parquet(PROC / "player_match.parquet")
    names = pm[["match_id", "player_id", "player_name"]]
    colors = {"D": C_D, "M": C_M, "F": C_F}
    for mid, title in SHOWCASE.items():
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.9))
        for ax, (tid, g) in zip(axes, slots[slots.match_id == mid].groupby("team_id")):
            pitch(ax)
            info = tm[(tm.match_id == mid) & (tm.team_id == tid)].iloc[0]
            team = nodes[(nodes.match_id == mid) & (nodes.team_id == tid)].team.iloc[0]
            g = g.merge(names, on=["match_id", "player_id"])
            pos = g.set_index("player_id")[["x_m", "y_m"]]
            e = edges[(edges.match_id == mid) & (edges.team_id == tid) & edges.passer_id.isin(pos.index)
                      & edges.receiver_id.isin(pos.index)]
            pair = e.assign(a=np.minimum(e.passer_id, e.receiver_id), b=np.maximum(e.passer_id, e.receiver_id)) \
                .groupby(["a", "b"]).pass_count.sum().reset_index()
            for r in pair[pair.pass_count >= 6].itertuples():
                ax.plot([pos.x_m[r.a], pos.x_m[r.b]], [pos.y_m[r.a], pos.y_m[r.b]], color=MUTED,
                        lw=0.4 + r.pass_count / 8, alpha=0.45, zorder=1)
            vol = nodes[(nodes.match_id == mid) & (nodes.team_id == tid)].set_index("player_id")
            for r in g.itertuples():
                v = vol.passes_completed.get(r.player_id, 20) + vol.passes_failed.get(r.player_id, 0)
                ax.scatter(r.x_m, r.y_m, s=60 + 4 * v, color=colors[r.slot_type], edgecolor="white", lw=1.2, zorder=3)
                ax.text(r.x_m, r.y_m, r.slot, ha="center", va="center", fontsize=6.5, color="white", weight="bold", zorder=4)
                ax.text(r.x_m, r.y_m - 4.2, r.player_name.split()[-1], ha="center", va="top", fontsize=7, color=INK, zorder=4)
            ax.annotate("", xy=(100, -1), xytext=(80, -1), arrowprops=dict(arrowstyle="->", color=MUTED))
            ax.set_title(f"{team}\nteamsheet {info.teamsheet_formation}  ·  lines in possession {info.shape_in_possession}"
                         f"", fontsize=9.5, color=INK)
        fig.suptitle(title, fontsize=12, color=INK, y=1.0)
        fig.text(0.5, 0.01, "Average positions of the starting outfielders; slot labels from the teamsheet formation "
                 "(blue defence, green midfield, orange attack); lines = passes between the pair (6+). Own goal on the left.",
                 ha="center", fontsize=8, color=MUTED)
        fig.tight_layout()
        fig.savefig(FIG / f"formations_{mid}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def style_figures():
    L = pd.read_csv(REP / "style_embedding_loadings.csv")
    last = L[L.season_fit == L.season_fit.max()]
    comps = sorted(last.component.unique())
    fig, axes = plt.subplots(1, len(comps), figsize=(3.1 * len(comps), 7), sharey=True)
    order = last[last.component == 1].sort_values("loading").label.tolist()
    for ax, c in zip(axes, comps):
        d = last[last.component == c].set_index("label").loc[order]
        ax.barh(range(len(d)), d.loading, color=np.where(d.loading > 0, C_D, C_F))
        ax.axvline(0, color=GRID)
        ax.set_title(f"{c}. {STYLE_NAMES.get(c, '')}", fontsize=10)
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels(d.index, fontsize=8)
        ax.tick_params(labelsize=8)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    fig.suptitle(f"Style dimensions: loadings (fit at the start of {last.season_fit.iloc[0]})", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "style_loadings.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    st = pd.read_parquet(PROC / "team_style_asof.parquet")
    nodes = pd.read_parquet(PROC / "network_nodes.parquet")[["team_id", "team"]].drop_duplicates("team_id")
    latest = st.sort_values("date").groupby("team_id").tail(1)
    latest = latest[latest.season == latest.season.max()].merge(nodes, on="team_id")
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(latest.style_1, latest.style_2, color=C_D)
    for r in latest.itertuples():
        ax.text(r.style_1, r.style_2 + 0.12, r.team, fontsize=7.5, ha="center", color=INK)
    ax.axhline(0, color=GRID, lw=0.8)
    ax.axvline(0, color=GRID, lw=0.8)
    ax.set_xlabel(f"{STYLE_NAMES[1]}  (direct  ←   →  control)")
    ax.set_ylabel(f"{STYLE_NAMES[2]}  (wide  ←   →  central)")
    ax.set_title("Own style of the 2026/27 teams, as of their latest match", fontsize=11)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "style_map.png", dpi=150)
    plt.close(fig)

    p = pd.read_csv(REP / "style_prediction.csv")
    p = p[p.split == "test"].sort_values("r2_full")
    fig, ax = plt.subplots(figsize=(8, 8))
    y = np.arange(len(p))
    ax.scatter(p.r2_league, y, color=MUTED, label="League average", s=18)
    ax.scatter(p.r2_own_only, y, color=C_M, label="Own style only", s=18)
    ax.scatter(p.r2_with_strength, y, color=C_F, label="+ strength gap", s=18)
    ax.scatter(p.r2_full, y, color=C_D, label="+ style imposed by opponent", s=22)
    ax.set_yticks(y)
    ax.set_yticklabels(p.label, fontsize=8)
    ax.axvline(0, color=GRID)
    ax.set_xlabel("Share of match-to-match variation predicted before kick-off (test seasons)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "style_prediction.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    formation_figures()
    style_figures()
