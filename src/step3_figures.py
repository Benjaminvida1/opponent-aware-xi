"""Step 3 memo tables and figures: player ratings within position group, L1 vs L0.

Player effects only compare players in the same position group: across groups the split of a team's npxG
between positions is fixed by the intercept and partly by game-state confounding (defenders come on to
protect leads). So each rating is shown relative to the average rated player of his group.

Run after run_l1.py:  python src/step3_figures.py
Outputs: reports/player_ratings_by_group.csv (top and bottom 5 per group),
         reports/figures/l1_vs_l0.png, reports/figures/player_ratings_by_group.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
GROUP_NAMES = {"GK": "Goalkeepers", "CB": "Centre-backs", "FB": "Full-backs", "CM": "Central midfielders",
               "AM": "Attacking midfielders", "W": "Wingers", "FW": "Strikers"}


def by_group(r: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    r = r.dropna(subset=["group"]).copy()
    for c in ["att_npxg_per90", "def_npxg_per90", "net_npxg_per90"]:
        r[c.replace("_npxg_per90", "_vs_group")] = r[c] - r.groupby("group")[c].transform("mean")
    out = []
    for g, d in r.groupby("group"):
        d = d.sort_values("net_vs_group", ascending=False)
        out.append(d.head(k).assign(end="top"))
        out.append(d.tail(k).assign(end="bottom"))
    cols = ["group", "end", "player_name", "team", "minutes_365d", "att_vs_group", "def_vs_group", "net_vs_group"]
    return pd.concat(out)[cols].reset_index(drop=True)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    r = pd.read_csv(REPORTS / "player_ratings_current.csv")
    t = by_group(r)
    t.round(3).to_csv(REPORTS / "player_ratings_by_group.csv", index=False)

    groups = [g for g in ["FW", "W", "AM", "CM", "FB", "CB", "GK"] if g in set(t.group)]
    fig, axes = plt.subplots(len(groups), 1, figsize=(8, 2.1 * len(groups)))
    for ax, g in zip(np.atleast_1d(axes), groups):
        d = t[t.group == g].sort_values("net_vs_group")
        cols = np.where(d.net_vs_group > 0, "#2b6cb0", "#c05621")
        ax.barh(d.player_name + " (" + d.team.str.replace(" & Hove Albion", "").str[:14] + ")", d.net_vs_group, color=cols)
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_title(GROUP_NAMES[g], fontsize=10, loc="left")
        ax.tick_params(labelsize=7)
    np.atleast_1d(axes)[-1].set_xlabel("Net effect vs group average (team npxG difference per 90)")
    fig.tight_layout()
    fig.savefig(FIG / "player_ratings_by_group.png", dpi=150)

    s = pd.read_csv(REPORTS / "l1_test_scores.csv").set_index("model")
    mkt = 0.2048
    order = ["L0", "Team effects on stints", "L1 without profile prior", "L1 profile prior only", "L1"]
    fig, ax = plt.subplots(figsize=(7, 3))
    v = s.loc[order, "rps"]
    ax.barh(order, v, color=["#888"] + ["#a0aec0"] * 3 + ["#2b6cb0"])
    ax.axvline(mkt, color="#c05621", ls="--", lw=1)
    ax.text(mkt, -0.6, " market", color="#c05621", fontsize=8)
    ax.set_xlim(0.200, 0.212)
    ax.set_xlabel("RPS on test (2025/26 + 2026/27 R1-5; lower is better)")
    for y, x in enumerate(v):
        ax.text(x + 0.0001, y, f"{x:.4f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "l1_vs_l0.png", dpi=150)
    print(t.round(3).to_string())


if __name__ == "__main__":
    main()
