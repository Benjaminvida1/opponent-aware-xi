"""Figures and checkpoint tables for the Step 2 memo (reports/figures/, reports/). Run after profiles.py and roles.py.

  role_profiles.png     what each role does: role averages as z-scores across player-seasons (latest fit)
  role_vs_slot.png      which formation slots each role's starters fill (test seasons)
  profile_test.png      how much better the profiles predict each player's next match than the
                        position-group average, and than unshrunk rates (test seasons)
  reports/role_spotcheck.csv   latest profile and roles of well-known players, for the checkpoint
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

import profiles as PR  # noqa: E402
import roles as RL  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REP = ROOT / "reports"
FIG = REP / "figures"
INK, MUTED, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
DIVERGING = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#ef9a8f", "#b8322f"])
SEQ = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab"])
plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.edgecolor": "#d0d0cc", "figure.facecolor": SURF,
                     "axes.facecolor": SURF})
SHOW = ["npxg", "box_touches", "headed_shots", "xa", "chances_created", "dribble_attempts", "crosses",
        "passes", "pass_acc", "long_balls", "prog_out", "prog_in", "tackles", "interceptions", "recoveries",
        "clearances", "aerial_duels", "aerial_win", "x_rel", "y_abs", "betweenness"]
SPOTCHECK = ["Bukayo Saka", "Declan Rice", "Virgil van Dijk", "Mohamed Salah", "Erling Haaland", "Cole Palmer",
             "Bruno Fernandes", "Martin Ødegaard", "William Saliba", "Moisés Caicedo", "Pedro Porro",
             "Alexander Isak", "Morgan Rogers", "Bruno Guimarães", "James Tarkowski", "Chris Wood"]


def name(r: int) -> str:
    return f"{r}. {RL.ROLE_NAMES.get(r, '')}"


def role_profiles():
    prof = pd.read_parquet(PROC / "player_profile_asof.parquet")
    pm = pd.read_parquet(PROC / "player_match.parquet")
    ps = RL.player_seasons(prof, pm)
    means = pd.read_csv(REP / "role_means.csv")
    m = means[means.fit_for_season == RL.SEASONS[-1]].set_index("role")
    X = RL.transform(ps)
    z = (RL.transform(m) - X.mean()) / X.std(ddof=0)
    z = z[SHOW]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    im = ax.imshow(z.to_numpy(), cmap=DIVERGING, vmin=-2.5, vmax=2.5, aspect="auto")
    ax.set_yticks(range(len(z)), [name(r) for r in z.index])
    ax.set_xticks(range(len(SHOW)), [PR.FEATURES[f][5].replace(" (network)", "") for f in SHOW],
                  rotation=50, ha="right")
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            v = z.iat[i, j]
            if abs(v) >= 1.0:
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 1.8 else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("SD from the average outfield player-season")
    ax.set_title("What each role does (role averages, fit used for 2026/27)", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "role_profiles.png", dpi=160)
    plt.close(fig)


def role_vs_slot():
    ct = pd.read_csv(REP / "role_vs_slot.csv", index_col=0)
    order = ["CB", "FB", "CM", "AM", "W", "FW"]
    labels = {"CB": "Centre-back", "FB": "Full/wing-back", "CM": "Central/holding mid", "AM": "Attacking mid",
              "W": "Wide mid / winger", "FW": "Striker"}
    ct = ct.reindex(order).fillna(0)
    ct.columns = [int(c) for c in ct.columns]
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    im = ax.imshow(ct.to_numpy(), cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(ct.shape[1]), [name(c) for c in ct.columns], rotation=35, ha="right")
    ax.set_yticks(range(len(order)), [labels[o] for o in order])
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            v = ct.iat[i, j]
            if v >= 0.05:
                ax.text(j, i, f"{100 * v:.0f}%", ha="center", va="center", fontsize=7.5,
                        color="white" if v > 0.6 else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Which slot each role's starters fill (test seasons; each column sums to 100%)", loc="left",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "role_vs_slot.png", dpi=160)
    plt.close(fig)


def profile_test():
    ev = pd.read_csv(REP / "profile_test.csv")
    ev = ev[ev.feature != "y_side"].sort_values("gain_vs_group_mean_pct")
    labels = [PR.FEATURES[f][5].replace(" (network)", "") for f in ev.feature]
    fig, axes = plt.subplots(1, 2, figsize=(10, 8), sharey=True)
    for ax, col, title, colr in [(axes[0], "group_mean", "vs the position-group average", BLUE),
                                 (axes[1], "unshrunk", "vs the player's own unshrunk rate", ORANGE)]:
        y = np.arange(len(ev))
        v = ev[f"gain_vs_{col}_pct"].to_numpy()
        lo, hi = ev[f"gain_vs_{col}_lo"].to_numpy(), ev[f"gain_vs_{col}_hi"].to_numpy()
        ax.barh(y, v, height=0.6, color=colr)
        ax.errorbar(v, y, xerr=[v - lo, hi - v], fmt="none", ecolor=MUTED, elinewidth=1, capsize=0)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("Prediction error reduced, % (95% interval)")
        ax.grid(axis="x", color="#e6e6e2", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_yticks(np.arange(len(ev)), labels)
    fig.suptitle("Profiles predict each player's next match better (2025/26 + 2026/27, 12,164 player-matches)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "profile_test.png", dpi=160)
    plt.close(fig)


def spotcheck():
    roles = pd.read_parquet(PROC / "player_roles_asof.parquet")
    prof = pd.read_parquet(PROC / "player_profile_asof.parquet")
    K = RL.K_CHOSEN
    rows = []
    for nm in SPOTCHECK:
        r = roles[roles.player_name == nm].sort_values("date")
        if r.empty:
            continue
        last = r.iloc[-1]
        p = prof[(prof.match_id == last.match_id) & (prof.player_id == last.player_id)].iloc[0]
        probs = last[[f"role_{k}" for k in range(1, K + 1)]].astype(float).to_numpy()
        top = np.argsort(probs)[::-1][:2]
        # role history: share of his 2025/26 + 2026/27 squad rows in each top role
        hist = r[r.season.isin(PR.TEST)].top_role.value_counts(normalize=True)
        rows.append(dict(player=nm, as_of=str(last.date.date()), team_id=last.team_id, group=last.group,
                         role=name(top[0] + 1), prob=round(probs[top[0]], 2),
                         second=name(top[1] + 1), second_prob=round(probs[top[1]], 2),
                         test_rows_in_top_role=round(float(hist.get(top[0] + 1, 0)), 2),
                         npxg90=round(p.npxg, 2), xa90=round(p.xa, 2), passes90=round(p.passes, 1),
                         dribbles90=round(p.dribble_attempts, 1), tackles90=round(p.tackles, 1),
                         aerials90=round(p.aerial_duels, 1), depth_vs_team_m=round(p.x_rel, 1),
                         from_centre_m=round(p.y_abs, 1), eff_90s=round(p.eff_90s, 1)))
    pd.DataFrame(rows).to_csv(REP / "role_spotcheck.csv", index=False)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    role_profiles()
    role_vs_slot()
    profile_test()
    print(spotcheck().to_string())
