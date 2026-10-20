"""Step 4 memo figures: the model ladder on the test seasons, and the network predictor's position accuracy.

Run after netpred.py and run_l23.py:  python src/step4_figures.py
Outputs: reports/figures/l23_ladder.png, reports/figures/netpred_positions.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
INK, MUTED, ACCENT, GRID = "#1f2933", "#7b8794", "#2f6fb0", "#e4e7eb"


def ladder():
    c = pd.read_csv(REPORTS / "l23_comparisons.csv")
    c = c[c.metric == "rps"].set_index("comparison")
    rows = [("L2 (style clash) vs L1", "L2 minus L1"), ("L3 (+ matchups) vs L2", "L3 minus L2"),
            ("L3 vs L1", "L3 minus L1"), ("Matchups alone vs L1", "L1 + matchups only minus L1"),
            ("L3 vs L3 without network", "L3 minus L3 without network"),
            ("L3 vs geometry only", "L3 minus L3 network geometry only")]
    l1 = pd.read_csv(REPORTS / "l1_comparisons.csv").set_index("comparison")
    l1 = l1[l1.metric == "rps"].loc["L1 minus L0"]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    labels = ["L1 vs L0 (Step 3, for scale)"] + [r[0] for r in rows]
    vals = [(l1["diff"], l1.ci_low, l1.ci_high)] + [tuple(c.loc[k, ["diff", "ci_low", "ci_high"]]) for _, k in rows]
    for i, (d, lo, hi) in enumerate(vals):
        col = MUTED if i == 0 else ACCENT
        ax.plot([lo, hi], [i, i], color=col, lw=2.2, solid_capstyle="round")
        ax.plot(d, i, "o", color=col, ms=6)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(range(len(labels)), labels, fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("Change in ranked probability score (negative = better), 95% interval", fontsize=9, color=INK)
    ax.grid(axis="x", color=GRID)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title("Test seasons, 430 matches: opponent-aware layers add nothing measurable", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "l23_ladder.png", dpi=160)


def positions():
    v = pd.read_csv(REPORTS / "netpred_validation.csv")
    v = v[(v.part == "positions") & (v.split == "test") & v.rmse_x.notna()]
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    names = {"slot average only": "League slot average", "own recent average": "Player's own recent average",
             "model": "Network predictor"}
    for i, r in enumerate(v.itertuples()):
        ax.barh(i - 0.18, r.rmse_x, 0.34, color=ACCENT if r.variant == "model" else MUTED)
        ax.barh(i + 0.18, r.rmse_y, 0.34, color=ACCENT if r.variant == "model" else MUTED, alpha=0.55)
        ax.text(r.rmse_x + 0.1, i - 0.18, f"depth {r.rmse_x:.1f} m", va="center", fontsize=8, color=INK)
        ax.text(r.rmse_y + 0.1, i + 0.18, f"width {r.rmse_y:.1f} m", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(v)), [names[x] for x in v.variant], fontsize=9, color=INK)
    ax.set_xlabel("Typical error of a starter's pre-match position (m), test seasons", fontsize=9, color=INK)
    ax.set_xlim(0, 11.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "netpred_positions.png", dpi=160)


if __name__ == "__main__":
    ladder()
    positions()
