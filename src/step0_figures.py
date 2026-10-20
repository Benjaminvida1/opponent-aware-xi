"""Figures for the Step 0 memo (reports/figures/). Run after run_l0.py.

  l0_vs_market.png      test-period RPS of base rates, goals ratings, L0 and the market, with 95% intervals
  l0_calibration.png    reliability of L0 and market W/D/L probabilities on the test period
  season_carryover.png  team rating in one season vs the next (the omega estimate)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
FIG = REP / "figures"
INK, MUTED, GRID = "#1f2328", "#6e7781", "#d0d7de"
C_L0, C_MKT, C_OTHER = "#2f6fbd", "#c2571a", "#8c959f"


def style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def l0_vs_market():
    t = pd.read_csv(REP / "l0_test_scores.csv").set_index("model")
    order = ["Base rates", "Goals ratings", "L0", "Market (Bet365 closing)"]
    labels = ["Base rates", "Ratings on goals", "L0 (ratings on npxG)", "Market (closing odds)"]
    colors = [C_OTHER, C_OTHER, C_L0, C_MKT]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    v = t.loc[order, "rps"].to_numpy()
    ax.barh(range(4), v, color=colors, height=0.6)
    for i, x in enumerate(v):
        ax.text(x + 0.0005, i, f"{x:.4f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(4), labels, color=INK, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(min(v) - 0.01, max(v) + 0.006)
    ax.set_xlabel("Ranked probability score on the test period (lower is better)", color=MUTED, fontsize=9)
    style(ax)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "l0_vs_market.png", dpi=180)
    plt.close(fig)


def calibration():
    c = pd.read_csv(REP / "l0_calibration.csv")
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.plot([0, 0.8], [0, 0.8], color=GRID, lw=1)
    for name, col, off in [("L0", C_L0, -0.004), ("Market", C_MKT, 0.004)]:
        d = c[c.model == name]
        ax.errorbar(d.predicted + off, d.observed, yerr=[d.observed - d.ci_low, d.ci_high - d.observed],
                    fmt="o", color=col, ms=5, capsize=2, lw=1, label=name)
    ax.set_xlabel("Predicted probability", color=MUTED, fontsize=9)
    ax.set_ylabel("Observed frequency", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=9)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "l0_calibration.png", dpi=180)
    plt.close(fig)


def carryover():
    d = pd.read_csv(REP / "season_carryover.csv")
    s = json.loads((REP / "l0_summary.json").read_text())["season_carryover_omega"]["pooled"]
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for (a, b), g in d.groupby(["season_from", "season_to"]):
        ax.scatter(g.rating_from, g.rating_to, s=18, alpha=0.8, label=f"{a[:4]}/{a[7:]} → {b[:4]}/{b[7:]}")
    x = np.linspace(d.rating_from.min(), d.rating_from.max(), 10)
    ax.plot(x, x, color=GRID, lw=1)
    xm, ym = d.rating_from.mean(), d.rating_to.mean()
    ax.plot(x, ym + s["omega"] * (x - xm), color=INK, lw=1.2)
    ax.text(0.03, 0.95, f"slope (omega) = {s['omega']:.2f}", transform=ax.transAxes, fontsize=9, color=INK, va="top")
    ax.set_xlabel("npxG difference per match, season N", color=MUTED, fontsize=9)
    ax.set_ylabel("npxG difference per match, season N+1", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "season_carryover.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    l0_vs_market()
    calibration()
    carryover()
    print("figures written to", FIG)
