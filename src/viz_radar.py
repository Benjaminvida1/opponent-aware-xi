"""Player radars in club colours, against same-position players over one season.

Two scales (`mode`):
  "percentile"  rank in the position group, 0-100 (50 = group median). Easy to read; hides how far
                ahead a player is.
  "zscore"      standard deviations from the group mean, drawn from -2.5 (centre) to +2.5 (edge) with the
                group mean on the dashed ring. Shows distance, not just order; values beyond ±2.5 sit on
                the edge and are labelled with their true value.

`impact=True` adds the L1 model's player effects (Step 3) under the radar: change in own team's npxG
(attack), opponent npxG prevented (defence) and net, per 90 and relative to the position group, over the
365 days before the ratings date. They answer a different question from the radar (on-pitch impact vs
what the player does), so they sit beside it rather than on it.

    python src/viz_radar.py                                   # Rice and Saka, 2025/26, both modes
    python src/viz_radar.py 2025/2026 zscore "Declan Rice" "Bukayo Saka"
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import viz_colors as C  # noqa: E402
import viz_data as D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures" / "viz"
plt.rcParams["font.family"] = ["Inter", "DejaVu Sans"]
Z_LIM = 2.5


def fmt_value(c, v):
    if c in D.RATE_COLS:
        return f"{v:.0f}%"
    return f"{v:.2f}" if abs(v) < 1 else f"{v:.1f}"


def radius(mets, mode):
    if mode == "percentile":
        return mets.percentile.to_numpy(), [f"{p:.0f}" for p in mets.percentile]
    z = mets.z.to_numpy()
    return 50 + np.clip(z, -Z_LIM, Z_LIM) / Z_LIM * 50, [f"{v:+.1f}" for v in z]


def draw_radar(ax, mets, colour, mode="percentile"):
    n = len(mets)
    ang = np.pi / 2 - np.arange(n) * 2 * np.pi / n
    step = 2 * np.pi / n
    ax.set_aspect("equal"); ax.axis("off")
    for i, sec in enumerate(mets.section):
        if sec == "Passing":
            th = np.linspace(ang[i] + step / 2, ang[i] - step / 2, 30)
            ax.fill(np.r_[0, 100 * np.cos(th), 0], np.r_[0, 100 * np.sin(th), 0], color=C.MUTED_FILL, zorder=0, lw=0)
    rings = (25, 50, 75, 100) if mode == "percentile" else (10, 30, 50, 70, 90, 100)
    t = np.linspace(0, 2 * np.pi, 200)
    for r in rings:
        ax.plot(r * np.cos(t), r * np.sin(t), color=C.LINE, lw=1.4 if r == 100 else 0.8,
                ls=(0, (3, 3)) if r == 50 else "-", zorder=1)
    for a in ang:
        ax.plot([0, 100 * np.cos(a)], [0, 100 * np.sin(a)], color=C.LINE, lw=0.8, zorder=1)
    ax.text(2, 50, "50th" if mode == "percentile" else "group mean", fontsize=7, color=C.INK_3, va="bottom",
            ha="left", zorder=2)
    if mode == "zscore":
        for r, lab in [(70, "+1"), (90, "+2")]:
            ax.text(2, r, lab, fontsize=7, color=C.INK_3, va="bottom", ha="left", zorder=2)

    v, labels = radius(mets, mode)
    xs, ys = v * np.cos(ang), v * np.sin(ang)
    ax.fill(np.r_[xs, xs[0]], np.r_[ys, ys[0]], color=colour, alpha=0.2, zorder=3, lw=0)
    ax.plot(np.r_[xs, xs[0]], np.r_[ys, ys[0]], color=colour, lw=2, zorder=4)
    ax.scatter(xs, ys, s=34, color=colour, edgecolors=C.SURFACE, linewidths=1.5, zorder=5)
    names = {c: lab for lab, c, _ in D.RADAR_METRICS}
    for c, a, lab in zip(mets.metric, ang, labels):
        lx, ly = 121 * np.cos(a), 121 * np.sin(a)
        ha = "center" if abs(np.cos(a)) < 0.2 else ("left" if np.cos(a) > 0 else "right")
        va = "center" if abs(np.sin(a)) < 0.2 else ("bottom" if np.sin(a) > 0 else "top")
        ax.text(lx, ly, names[c], ha=ha, va=va, fontsize=8.5, color=C.INK, linespacing=1.1, zorder=6)
        ax.text(108 * np.cos(a), 108 * np.sin(a), lab, ha="center", va="center", fontsize=8.5, weight="semibold",
                color=colour, zorder=6, bbox=dict(boxstyle="round,pad=0.18", fc=C.SURFACE, ec="none"))
    ax.set_xlim(-175, 175); ax.set_ylim(-160, 165)


def _impact_row(fig, x0, top_in, H, w, imp):
    """Three tiles: L1 attack / defence / net impact vs the position group (top_in = inches from bottom)."""
    fig.text(x0, top_in / H, f"Model impact (L1, last 365 days, vs average {D.GROUP_NAMES[imp['group']][:-1]})",
             fontsize=9, color=C.INK, weight="semibold", va="top")
    tiles = [("Attack", imp["att_vs_group"], imp["att_pct"]), ("Defence", imp["def_vs_group"], imp["def_pct"]),
             ("Net", imp["net_vs_group"], imp["net_pct"])]
    for j, (k, v, p) in enumerate(tiles):
        xx = x0 + j * w / 3
        fig.text(xx, (top_in - 0.3) / H, f"{v:+.3f}", fontsize=15, weight="semibold", color=C.INK, va="top")
        fig.text(xx, (top_in - 0.62) / H, f"{k}: npxG per 90\n{p:.0f}th percentile", fontsize=8,
                 color=C.INK_2, va="top", linespacing=1.2)


def radar_figure(season: str, players, mode: str = "percentile", impact: bool = True, min_minutes: int = 900,
                 out=None):
    if mode not in ("percentile", "zscore"):
        raise ValueError("mode must be 'percentile' or 'zscore'")
    k = len(players)
    H = 10.2 if impact else 8.8
    fig = plt.figure(figsize=(7.4 * k, H), facecolor=C.SURFACE)
    scale = ("Percentile against same-position players" if mode == "percentile"
             else "Standard deviations from the same-position average")
    fig.text(0.06 / k, 1 - 0.45 / H, "Player profiles", fontsize=18, weight="semibold", color=C.INK)
    fig.text(0.06 / k, 1 - 0.75 / H, f"{scale}, Premier League {season.replace('/20', '/')} "
             f"(min. {min_minutes} minutes)", fontsize=10.5, color=C.INK_2)
    data = []
    for i, pl in enumerate(players):
        r = D.player_radar(season, pl, min_minutes)
        col = C.color(r["team"])
        x0 = i / k
        bottom_in = 3.2 if impact else 1.95
        ax = fig.add_axes([x0 + 0.01 / k, bottom_in / H, 0.98 / k, 5.3 / H])
        draw_radar(ax, r["metrics"], col, mode)
        fig.text(x0 + 0.06 / k, 1 - 1.3 / H, r["name"], fontsize=14, weight="semibold", color=C.INK)
        fig.text(x0 + 0.06 / k, 1 - 1.5 / H, f"{r['team']}  ·  vs {r['n_peers']} {r['group_name']}  ·  "
                 f"{r['minutes']:,} minutes", fontsize=9.5, color=C.INK_2, va="top")
        imp = D.player_impact(r["player_id"]) if impact else None
        if impact:
            if imp is not None:
                _impact_row(fig, x0 + 0.12 / k, 2.95, H, 0.76 / k, imp)
            else:
                fig.text(x0 + 0.12 / k, 2.8 / H, "Model impact: no L1 rating (under 900 minutes in the "
                         "last 365 days)", fontsize=8.5, color=C.INK_3)
        tax = fig.add_axes([x0 + 0.12 / k, 0.55 / H, 0.76 / k, 1.15 / H]); tax.axis("off")
        mets = r["metrics"]
        half = (len(mets) + 1) // 2
        for j, row in enumerate(mets.itertuples()):
            cc, rr = divmod(j, half)
            xx, yy = cc * 0.52, 0.95 - rr * 0.17
            lab = next(lb for lb, c, _ in D.RADAR_METRICS if c == row.metric).replace("\n", " ")
            tax.text(xx, yy, lab, fontsize=8, color=C.INK_2, ha="left", va="center")
            tax.text(xx + 0.45, yy, fmt_value(row.metric, row.value), fontsize=8, color=C.INK, ha="right",
                     va="center", weight="semibold")
        data.append({**{kk: vv for kk, vv in r.items() if kk != "metrics"}, "metrics": mets, "impact": imp})
    note = ("Numbers on the radar are percentiles (50 = group median; dashed ring)" if mode == "percentile" else
            f"Numbers on the radar are z-scores (0 = group mean; dashed ring; edge = +{Z_LIM})")
    fig.text(0.06 / k, 0.1 / H, f"{note}. Values below each radar are per 90 minutes. Non-penalty xG. Shaded "
             "sector = passing.\n*Adjusted to 50% opponent possession. Group = most frequent starting slot. "
             "Model impact: Step 3 player effects relative to the group. Source: FotMob player stats (data v2).",
             fontsize=8, color=C.INK_3)
    slug = "_".join(str(p).split()[-1].lower() for p in players)
    out = Path(out or FIG / f"radar_{mode}_{season.replace('/', '-')}_{slug}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=C.SURFACE)
    plt.close(fig)
    return out, data


if __name__ == "__main__":
    args = sys.argv[1:]
    season = args[0] if args else "2025/2026"
    modes = [args[1]] if len(args) > 1 else ["percentile", "zscore"]
    players = args[2:] or ["Declan Rice", "Bukayo Saka"]
    for md in modes:
        print(radar_figure(season, players, md)[0])
