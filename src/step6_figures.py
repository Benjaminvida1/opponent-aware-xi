"""Step 6 figure: the model ladder with match-level and round-level bootstrap intervals (methodology §15.2).

Run: python src/step6_figures.py  ->  reports/figures/step6_ladder_robustness.png
Reads reports/robustness_round_bootstrap.csv (written by src/robustness.py).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = pd.read_csv(ROOT / "reports" / "robustness_round_bootstrap.csv")
R = R[R.metric == "rps"].set_index("comparison")
ROWS = [("L1 minus L0", "L1 vs L0 (players)"),
        ("Match-level L1 (L1 settings) minus L0", "Match-level L1 vs L0 (no stints)"),
        ("Match-level L1 (L1 settings) minus L1", "Match-level L1 vs L1"),
        ("L2 minus L1", "L2 vs L1 (style clash)"),
        ("L3 minus L2", "L3 vs L2 (matchups)"),
        ("L3 minus L3 without network", "L3 vs L3 without network")]
ROWS = [(k, lab) for k, lab in ROWS if k in R.index]
fig, ax = plt.subplots(figsize=(9, 0.62 * len(ROWS) + 1.2))
for i, (k, lab) in enumerate(ROWS):
    r = R.loc[k]
    y = len(ROWS) - 1 - i
    ax.plot([r.ci_low_match, r.ci_high_match], [y + 0.13] * 2, color="#2b6cb0", lw=3, solid_capstyle="round")
    ax.plot([r.ci_low_round, r.ci_high_round], [y - 0.13] * 2, color="#c05621", lw=3, solid_capstyle="round")
    ax.plot([r["diff"]] * 2, [y + 0.13, y - 0.13], "o", color="#1a202c", ms=5)
ax.axvline(0, color="#1a202c", lw=1)
ax.set_yticks(range(len(ROWS)))
ax.set_yticklabels([lab for _, lab in ROWS][::-1])
ax.plot([], [], color="#2b6cb0", lw=3, label="resampling matches")
ax.plot([], [], color="#c05621", lw=3, label="resampling whole rounds")
ax.legend(loc="upper right", fontsize=8, frameon=False)
ax.set_xlabel("Change in ranked probability score (negative = better), 95% interval, 430 test matches")
ax.grid(axis="x", color="#e2e8f0")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out = ROOT / "reports" / "figures" / "step6_ladder_robustness.png"
fig.savefig(out, dpi=150)
print(out)
