"""Pitch figures in club colours: shot map (with situation filter), passing network, team shape (convex hull).

Data come from viz_data.py; colours from viz_colors.py. Each function writes a PNG and returns the
figure's data so a dashboard can reproduce it. Pitches are vertical, attacking upward; the team's left
(y = 68) is on the left of the image (mplsoccer's VerticalPitch flips the horizontal axis).

    python src/viz_plots.py                      # Arsenal v Chelsea 2026/27 R3 examples -> reports/figures/viz/
    python src/viz_plots.py 5795435 Chelsea
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from mplsoccer import VerticalPitch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import viz_colors as C  # noqa: E402
import viz_data as D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures" / "viz"

plt.rcParams["font.family"] = ["Inter", "DejaVu Sans"]
plt.rcParams["font.size"] = 10
HALO = [pe.withStroke(linewidth=3, foreground=C.SURFACE)]
STATS_X = 5.75 / 8


def _save(fig, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=C.SURFACE)
    plt.close(fig)
    return out


def _pitch(half=False, pad_bottom=1.0):
    return VerticalPitch(pitch_type="custom", pitch_length=105, pitch_width=68, half=half,
                         pad_top=1 if not half else 2, pad_bottom=pad_bottom, pad_left=1, pad_right=1,
                         line_color=C.LINE, linewidth=1.1, pitch_color=C.SURFACE, goal_type="box",
                         corner_arcs=True, line_zorder=1)


def _header(fig, H, title, sub):
    fig.text(0.06, 1 - 0.5 / H, title, fontsize=18, weight="semibold", color=C.INK)
    fig.text(0.06, 1 - 0.78 / H, sub, fontsize=10.5, color=C.INK_2)


def _stat_column(fig, H, top_in, stats, gap_in=0.95):
    for i, (k, v) in enumerate(stats):
        y = 1 - (top_in + i * gap_in) / H
        fig.text(STATS_X, y, v, fontsize=19, weight="semibold", color=C.INK, va="top")
        fig.text(STATS_X, y - 0.34 / H, k, fontsize=9.5, color=C.INK_2, va="top")


def _full_pitch_figure(H=10.4, pitch_bottom=0.95, pitch_h=7.7):
    fig = plt.figure(figsize=(8, H), facecolor=C.SURFACE)
    pitch = _pitch()
    w = pitch_h * 70 / 107
    ax = fig.add_axes([0.3 / 8, pitch_bottom / H, w / 8, pitch_h / H])
    pitch.draw(ax=ax)
    return fig, ax, pitch, H


def _chips(fig, x_in, y_in, H, labels, active, colour, ink):
    """A row of filter 'buttons' (preview of the dashboard control): active ones filled."""
    x = x_in
    for lab in labels:
        on = lab in active
        t = fig.text(x / 8, 1 - y_in / H, lab, fontsize=9, va="center", ha="left",
                     color=ink if on else C.INK_2,
                     bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.8",
                               fc=colour if on else C.SURFACE, ec=colour if on else C.LINE, lw=1.1))
        fig.canvas.draw()
        bb = t.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        x += bb.width + 0.22


# ---------------------------------------------------------------- shot map
OUTCOME_STYLE = {   # all circles; only goals are filled
    "Goal":       dict(fill=True,  edge=None,        lw=1.5),
    "Saved":      dict(fill=False, edge=None,        lw=1.9),   # edge = club colour
    "Blocked":    dict(fill=False, edge=C.INK_2,     lw=1.4),
    "Off target": dict(fill=False, edge=C.INK_3,     lw=1.1),
}


def xg_size(xg):
    return 70 + 1400 * np.asarray(xg)   # marker area in pt^2; 0.02 xG ~ 10 pt across


def shot_map(match_id: int, team: str, situations=None, out=None):
    """Every shot of `team` in the match as a circle sized by xG; goals filled, others hollow.

    situations: subset of viz_data.SITUATIONS keys (the dashboard's filter); None = all.
    """
    col = C.club(team)
    active = list(situations) if situations is not None else list(D.SITUATIONS)
    s = D.shots(match_id, team, active)
    allsit = D.shots(match_id, team)
    present = [k for k in D.SITUATIONS if k in set(allsit.situation_label)]
    opp = D.shots(match_id, D.opponent(match_id, team))
    m = D.match_info(match_id)
    stats = D.shot_summary(s)

    H = 8.0
    fig = plt.figure(figsize=(8, H), facecolor=C.SURFACE)
    pitch = VerticalPitch(pitch_type="custom", pitch_length=105, pitch_width=68, half=True,
                          pad_bottom=-19.5, pad_top=2, pad_left=1, pad_right=1, line_color=C.LINE,
                          linewidth=1.2, pitch_color=C.SURFACE, goal_type="box", corner_arcs=True)
    ax = fig.add_axes([0.05, 1.45 / H, 0.90, 3.6 / H])
    pitch.draw(ax=ax)

    for res in ["Off target", "Blocked", "Saved", "Goal"]:   # goals on top
        g = s[s.outcome_label == res].sort_values("xg", ascending=False)
        if g.empty:
            continue
        st = OUTCOME_STYLE[res]
        edge = st["edge"] or col["chart"]
        pitch.scatter(g.x_m, g.y_m, s=xg_size(g.xg), marker="o",
                      facecolors=col["chart"] if st["fill"] else "none",
                      edgecolors=C.SURFACE if st["fill"] else edge, linewidths=st["lw"],
                      zorder={"Goal": 6, "Saved": 5, "Blocked": 4, "Off target": 3}[res], ax=ax)

    for r in s[s.outcome_label == "Goal"].itertuples():
        left = r.y_m >= 34
        gap = 1.2 + 0.035 * np.sqrt(xg_size(r.xg))
        ax.text(r.y_m + gap if left else r.y_m - gap, r.x_m, f"{r.player_short} {r.minute_label}\n{r.xg:.2f} xG",
                ha="right" if left else "left", va="center", fontsize=9, color=C.INK, zorder=9,
                path_effects=HALO, linespacing=1.15)

    fig.text(0.06, 1 - 0.5 / H, f"{team} shot map", fontsize=18, weight="semibold", color=C.INK)
    fig.text(0.06, 1 - 0.78 / H, m["label"], fontsize=10.5, color=C.INK_2)
    _chips(fig, 0.48, 1.18, H, present, active, col["chart"], col["ink"])
    tiles = [("Goals", f"{stats['goals']}"), ("xG", f"{stats['xg']:.2f}"), ("Shots", f"{stats['shots']}"),
             ("On target", f"{stats['on_target']}"), ("Blocked", f"{stats['blocked']}"),
             ("xG / shot", f"{stats['xg_per_shot']:.2f}" if stats["shots"] else "–")]
    for i, (k, v) in enumerate(tiles):
        x0 = 0.06 + i * 0.148
        fig.text(x0, 1 - 1.95 / H, v, fontsize=20, weight="semibold", color=C.INK)
        fig.text(x0, 1 - 2.15 / H, k, fontsize=9.5, color=C.INK_2)
    o = D.shot_summary(opp)
    scope = "all situations" if set(active) >= set(present) else ", ".join(active).lower()
    fig.text(0.06, 1 - 2.45 / H, f"Showing {scope}.  {D.opponent(match_id, team)} (all situations): "
             f"{o['shots']} shots, {o['xg']:.2f} xG, {o['goals']} goal{'s' if o['goals'] != 1 else ''}",
             fontsize=9.5, color=C.INK_3)

    def h(fc, ec, ms=9, lw=1.4):
        return Line2D([], [], marker="o", ls="", ms=ms, mfc=fc, mec=ec, mew=lw)

    lax = fig.add_axes([0.05, 0.35 / H, 0.90, 1.0 / H]); lax.axis("off")
    out_h = [h(col["chart"], col["chart"]), h("none", col["chart"], lw=1.9), h("none", C.INK_2),
             h("none", C.INK_3, lw=1.1)]
    sizes = [0.05, 0.2, 0.5]
    size_h = [h("none", C.INK_3, ms=np.sqrt(xg_size(v)), lw=1) for v in sizes]
    kw = dict(frameon=False, labelcolor=C.INK_2, fontsize=9.5, borderaxespad=0, title_fontsize=9.5,
              alignment="left")
    l1 = lax.legend(out_h, D.OUTCOMES, title="Outcome", loc="upper left", bbox_to_anchor=(0.01, 1), ncol=2,
                    handletextpad=0.4, columnspacing=1.6, **kw)
    l2 = lax.legend(size_h, [f"{v:.2f}" for v in sizes], title="Size = xG", loc="upper left",
                    bbox_to_anchor=(0.55, 1), ncol=3, columnspacing=1.0, handlelength=2.6, handletextpad=0.9, **kw)
    for leg in (l1, l2):
        leg.get_title().set_color(C.INK); leg.get_title().set_weight("semibold"); lax.add_artist(leg)
    fig.text(0.06, 0.1 / H, "Source: FotMob shot map (data v2). Own goals excluded. Blocked shots counted "
             "separately from on target.", fontsize=8, color=C.INK_3)
    out = out or FIG / f"shot_map_{match_id}_{team.lower().replace(' ', '_')}.png"
    return _save(fig, out), s


# ---------------------------------------------------------------- passing network
def node_size(passes):
    return 120 + 9 * np.asarray(passes)


def _draw_nodes(pitch, ax, nodes, fill, ink, sized=True):
    s = node_size(nodes.passes_completed) if sized else np.full(len(nodes), 330)
    pitch.scatter(nodes.x_m, nodes.y_m, s=s, c=fill, edgecolors=C.SURFACE, linewidths=2, zorder=5, ax=ax)
    for r, sz in zip(nodes.itertuples(), s):
        ax.text(r.y_m, r.x_m, f"{int(r.shirt_number)}", ha="center", va="center", fontsize=8.5, color=ink,
                weight="semibold", zorder=6)
        off = (np.sqrt(sz) / 2 + 3) / 72 * 105 / 7.7
        ax.text(r.y_m, r.x_m - off, r.name, ha="center", va="top", fontsize=8.5, color=C.INK, zorder=7,
                path_effects=HALO)


def passing_network(match_id: int, team: str, min_link: int = 4, out=None):
    col = C.club(team)
    net = D.network(match_id, team, min_link)
    nodes, links, sm = net["nodes"], net["links"], net["summary"]
    fig, ax, pitch, H = _full_pitch_figure()
    _header(fig, H, f"{team} passing network", D.match_info(match_id)["label"])

    cm = C.cmap(col["chart"])
    shown = links[links.pass_count >= min_link]
    norm = Normalize(vmin=min_link, vmax=max(links.pass_count.max(), min_link + 1))
    for r in shown.itertuples():
        pitch.lines(r.x_m, r.y_m, r.x_m_b, r.y_m_b, lw=0.6 + 0.42 * r.pass_count, color=cm(norm(r.pass_count)),
                    comet=False, zorder=2, capstyle="round", ax=ax)
    _draw_nodes(pitch, ax, nodes, col["chart"], col["ink"])

    _stat_column(fig, H, 1.55, [
        ("Passes completed\nby the starting XI", f"{sm['passes_completed']}"),
        ("Completion", f"{sm['completion']:.0%}"),
        (f"Links drawn\n(≥{min_link} passes)", f"{sm['links_shown']}"),
        (f"Most passes\n{sm['hub']}", f"{sm['hub_passes']}"),
        (f"Most passes from one player\nto another: {sm['top_link']}", f"{sm['top_link_passes']}"),
    ])
    fig.text(STATS_X, 1 - 6.5 / H, "Link = passes between\nthe pair, both ways", fontsize=9.5, color=C.INK,
             weight="semibold", va="top")
    lax = fig.add_axes([STATS_X, 1 - 7.55 / H, 0.22, 0.6 / H]); lax.axis("off")
    for i, v in enumerate([5, 10, 20]):
        lax.plot([0, 0.35], [1 - i * 0.45] * 2, lw=0.6 + 0.42 * v, color=cm(norm(min(v, norm.vmax))),
                 solid_capstyle="round")
        lax.text(0.45, 1 - i * 0.45, f"{v}", va="center", fontsize=9.5, color=C.INK_2)
    lax.set_xlim(-0.1, 1); lax.set_ylim(-0.2, 1.2)
    fig.text(STATS_X, 1 - 7.85 / H, "Circle size = passes\ncompleted", fontsize=9.5, color=C.INK,
             weight="semibold", va="top")
    subs = ", ".join(f"{n} {m}'" for n, m in sm["substituted"]) or "none"
    fig.text(0.06, 0.18 / H, "Starting XI at average positions over each player's time on the pitch; substitutes "
             f"not shown.\nSubstituted: {subs}. Source: FotMob pass network (data v2).",
             fontsize=8, color=C.INK_3, va="bottom")
    out = out or FIG / f"passing_network_{match_id}_{team.lower().replace(' ', '_')}.png"
    return _save(fig, out), net


# ---------------------------------------------------------------- team shape (convex hull)
def convex_hull(match_id: int, team: str, out=None):
    col = C.club(team)
    sh = D.team_shape(match_id, team)
    nodes = sh["nodes"]
    of = nodes[~nodes.is_gk.fillna(False).astype(bool)]
    fig, ax, pitch, H = _full_pitch_figure()
    _header(fig, H, f"{team} team shape", D.match_info(match_id)["label"])

    poly = sh["hull"][["x_m", "y_m"]].to_numpy()
    pitch.polygon([poly], fc=col["chart"], ec="none", alpha=0.12, zorder=2, ax=ax)
    pitch.polygon([poly], fc="none", ec=col["chart"], lw=1.6, zorder=3, ax=ax)

    dl = sh["cb_height_m"]
    ax.plot([0, 68], [dl, dl], color=C.INK_3, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.text(67, dl + 0.8, f"Centre-backs {dl:.0f} m", fontsize=8.5, color=C.INK_2, ha="left", va="bottom",
            path_effects=HALO, zorder=4)
    yb = of.x_m.min() - 6
    ax.annotate("", xy=(of.y_m.max(), yb), xytext=(of.y_m.min(), yb),
                arrowprops=dict(arrowstyle="<->", color=C.INK_2, lw=1), zorder=4)
    ax.text((of.y_m.max() + of.y_m.min()) / 2, yb - 1, f"Width {sh['width_m']:.0f} m", ha="center", va="top",
            fontsize=8.5, color=C.INK_2, path_effects=HALO)
    xb = of.y_m.min() - 4
    ax.annotate("", xy=(xb, of.x_m.max()), xytext=(xb, of.x_m.min()),
                arrowprops=dict(arrowstyle="<->", color=C.INK_2, lw=1), zorder=4)
    ax.text(xb - 1, (of.x_m.max() + of.x_m.min()) / 2, f"Depth {sh['depth_m']:.0f} m", ha="left", va="center",
            rotation=90, fontsize=8.5, color=C.INK_2, path_effects=HALO)
    _draw_nodes(pitch, ax, nodes, col["chart"], col["ink"], sized=False)

    _stat_column(fig, H, 1.55, [
        ("Area of the outfield\nshape (m²)", f"{sh['area_m2']:,.0f}"),
        ("Larger than this share of\nPL team-matches, 2023–27", f"{sh['area_pct']:.0%}"),
        ("Width (m)", f"{sh['width_m']:.0f}"),
        ("Depth (m)", f"{sh['depth_m']:.0f}"),
        ("Centre-back height\n(m from own goal)", f"{dl:.0f}"),
        ("Team centre (m from\nown goal)", f"{sh['centre_x_m']:.0f}"),
    ])
    fig.text(0.06, 0.18 / H, "Convex hull of the 10 outfield starters' average positions (goalkeeper excluded). "
             f"Reference: all {sh['n_reference']:,} team-matches in the data.\nSource: FotMob pass network "
             "(data v2).", fontsize=8, color=C.INK_3, va="bottom")
    out = out or FIG / f"convex_hull_{match_id}_{team.lower().replace(' ', '_')}.png"
    return _save(fig, out), sh


if __name__ == "__main__":
    mid = int(sys.argv[1]) if len(sys.argv) > 1 else 5795435
    team = sys.argv[2] if len(sys.argv) > 2 else "Arsenal"
    print(shot_map(mid, team)[0])
    print(shot_map(mid, team, ["Set piece"], out=FIG / f"shot_map_{mid}_{team.lower()}_set_piece.png")[0])
    print(passing_network(mid, team)[0])
    print(convex_hull(mid, team)[0])
