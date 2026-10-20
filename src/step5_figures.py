"""Step 5 memo figure: is L1's sense of lineup changes right out of sample? (methodology §15.4)

Team-matches where the manager's XI differs from the usual XI, binned by L1's predicted change in npxG
(manager's XI minus usual XI); realized npxG minus the usual XI's forecast, with 95% intervals.
Run: python src/step5_figures.py  ->  reports/figures/step5_lineup_calibration.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = pd.read_parquet(ROOT / "data" / "processed" / "step5_decisions.parquet")
D = D[D.actual_xi.apply(lambda s: set(json.loads(s))) != D.usual_xi.apply(lambda s: set(json.loads(s)))]
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, side, title in zip(axes, ("for", "against"), ("npxG for", "npxG against")):
    x = D[f"actual_raw_npxg_{side}"] - D[f"usual_raw_npxg_{side}"]
    y = D[f"npxg_{side}_actual"] - D[f"usual_raw_npxg_{side}"]
    q = pd.qcut(x, 5)
    g = pd.DataFrame(dict(x=x, y=y, q=q)).groupby("q", observed=True)
    mx, my, se = g.x.mean(), g.y.mean(), g.y.std() / np.sqrt(g.size())
    ax.errorbar(mx, my - y.mean(), yerr=1.96 * se, fmt="o", color="#2b6cb0", capsize=3)
    lim = max(abs(mx).max() * 1.2, 0.05)
    ax.plot([-lim, lim], [-lim, lim], "--", color="grey", lw=1, label="realized = predicted")
    ax.axhline(0, color="lightgrey", lw=0.8)
    ax.set_title(f"{title} (n = {len(D)})")
    ax.set_xlabel("L1's predicted change: manager's XI − usual XI")
    ax.legend(fontsize=8, loc="upper left")
axes[0].set_ylabel("Realized minus usual-XI forecast\n(centred)")
fig.suptitle("Manager deviations from the usual XI: realized chances move with L1's prediction, noisily (test seasons)")
fig.tight_layout()
out = ROOT / "reports" / "figures" / "step5_lineup_calibration.png"
fig.savefig(out, dpi=150)
print(out)
