"""Step 5: explanation output for a recommendation (methodology §14).

Only the player terms carry expected points (L1). Everything else is descriptive, from the Step 4 tools, and is
labelled as not adding expected points (Step 4: style and matchup terms do not improve forecasts):
  shape        predicted positions of the usual and the recommended XI (netpred.PositionModel replayed to the
               match date; a player in a slot he has not started uses the slot position + his personal offset,
               §5.6), and the shape measurements they imply: defensive line, attacking height, width
  who faces    our five most advanced outfielders against their five deepest (mirrored, Hungarian on distance),
  whom         with each pair's dribble and aerial z-scores (as-of profiles)
  where they   the opponent's share of npxG conceded by lane relative to the league (matchups.lane_vulnerability)
  concede

Run:  python src/explain.py MATCH_ID TEAM_ID [MATCH_ID TEAM_ID ...]   -> prints a text card for each
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matchups as MU  # noqa: E402
import netpred as NP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
# Lanes in the attacking team's frame (y = 68 on its left): lane 0 is the attacker's right wing.
LANES = ["right wing", "right half-space", "centre", "left half-space", "left wing"]
_CACHE = {}


def _load():
    if "st" not in _CACHE:
        m, st = NP.load()
        _CACHE.update(m=m, st=st, pm=pd.read_parquet(PROC / "player_match.parquet"),
                      prof=pd.read_parquet(PROC / "player_profile_asof.parquet").drop_duplicates(["match_id", "player_id"])
                      .set_index(["match_id", "player_id"]),
                      D=pd.read_parquet(PROC / "step5_decisions.parquet"), S=pd.read_parquet(PROC / "step5_swaps.parquet"),
                      pos=pd.read_parquet(PROC / "pred_positions.parquet"))
        g = pd.read_csv(ROOT / "reports" / "netpred_position_grid.csv").sort_values("tune_mse").iloc[0]
        _CACHE["kw"] = (float(g.k_team), float(g.k_player), float(g.hl_player))
    return _CACHE


def position_model(date):
    """PositionModel state with every starter of matches before `date` added (as of date)."""
    c = _load()
    st = c["st"]
    pm = NP.PositionModel(*c["kw"])
    past = st[st.date < date]
    d_all = NP.days(past.date)
    for (mid, tid), g in past.groupby(["match_id", "team_id"], sort=False):
        pm.add(g, d_all[past.index.get_indexer(g.index)][0])
    return pm


def place(pm, team, F, slots, d):
    """Predicted (x, y) for each outfielder given his slot."""
    out = {}
    for p, s in slots.items():
        target = pm.S(F, s) + pm.shift(team, F, s)
        played = p in pm.own and _played_slot(p, s)
        out[p] = pm.usual(p, d, target) if played else target + pm.delta(p, d)
    return out


def _played_slot(p, s):
    st = _load()["st"]
    return bool(((st.player_id == p) & (st.slot == s)).any())


def lane_vuln(team, date):
    """matchups.lane_vulnerability for one team as of one date: its share of npxG conceded by lane (shooter's frame)
    over the league's share, 365-day half-life, shrunk with 3 pseudo-xG, clipped to [0.5, 2]."""
    c = _load()
    if "shots" not in c:
        sh = pd.read_parquet(PROC / "shots.parquet")
        sh = sh[(sh.situation_group != "Penalty") & ~sh.is_own_goal.astype(bool)].drop(columns=["date"])
        sh = sh.merge(c["m"][["match_id", "date", "home_team_id", "away_team_id"]], on="match_id")
        sh["conceding"] = np.where(sh.team_id == sh.home_team_id, sh.away_team_id, sh.home_team_id)
        sh["lane"] = np.clip((sh.y_m // 13.6).astype(int), 0, 4)
        c["shots"] = sh
    sh = c["shots"]
    past = sh[sh.date < date]
    if len(past) < 200:
        return None
    w = past.xg * 0.5 ** ((date - past.date).dt.total_seconds() / 86400 / 365.0)
    lg = w.groupby(past.lane).sum().reindex(range(5), fill_value=0)
    lg = lg / lg.sum()
    t = w[past.conceding == team].groupby(past.lane[past.conceding == team]).sum().reindex(range(5), fill_value=0)
    share = (t.to_numpy() + MU.VULN_K * lg.to_numpy()) / (t.sum() + MU.VULN_K)
    return list(np.clip(share / lg.to_numpy(), 0.5, 2.0))


def shape(xy):
    a = np.array(list(xy.values()))
    x, y = a[:, 0], a[:, 1]
    return dict(def_line=float(np.sort(x)[:4].mean()), att_height=float(np.sort(x)[-3:].mean()),
                width=float(2 * np.sqrt(((y - y.mean()) ** 2).mean())), centroid=float(x.mean()))


def name(p):
    pm = _load()["pm"]
    r = pm.loc[pm.player_id == p, "player_name"]
    return r.iloc[0] if len(r) else str(p)


def card(match_id, team_id, xi_name="recommended"):
    c = _load()
    D, S = c["D"], c["S"]
    r = D[(D.match_id == match_id) & (D.team_id == team_id)].iloc[0]
    m = c["m"].set_index("match_id").loc[match_id]
    opp_name = m.away_team if r.home else m.home_team
    team_name = m.home_team if r.home else m.away_team
    lines = [f"## {m.home_team} v {m.away_team} ({m.season}, round {m['round']}, {str(m.date)[:10]}) — for {team_name}",
             f"Formation {r.formation}. Pre-match (usual XI): win {r.usual_p_win:.0%}, draw {r.usual_p_draw:.0%}; "
             f"EP {r.usual_ep:.2f}. Gradients g_A {r.gA:.2f} (attack) vs g_B {r.gB:.2f} (defence)."]
    sw = S[(S.match_id == match_id) & (S.team_id == team_id) & (S.xi == "best")]
    rec_changes = r.recommended_changes > 0
    pb = "n/a" if np.isnan(r.p_better) else f"{r.p_better:.2f}"
    if len(sw) == 0:
        lines.append("**Recommendation: the usual XI** (the model's best XI is the usual XI).")
    else:
        head = "Recommended change" if rec_changes else "Option (below the 70% bar; usual XI recommended)"
        lines.append(f"**{head}: {r.best_ep - r.usual_ep:+.3f} EP, P(better) = {pb}.**")
        for s in sw.itertuples():
            lines.append(f"- {s.in_name} for {s.out_name} ({s.slot}): {s.dEP:+.3f} EP on its own "
                         f"(attack {s.dEP_attack:+.3f}, defence {s.dEP_defence:+.3f})")
    b, n = set(json.loads(r.best_xi)), set(json.loads(r.neutral_xi))
    if b != n:
        lines.append(f"Against a league-average opponent the best XI would have {', '.join(name(p) for p in n - b)} "
                     f"instead of {', '.join(name(p) for p in b - n)}; the difference for this match is "
                     f"{r.best_ep - r.neutral_ep:+.4f} EP.")
    else:
        lines.append("The best XI is the same against a league-average opponent: the opponent does not change the pick.")
    miss = json.loads(r.regulars_missing)
    if miss:
        lines.append(f"Regulars missing from the squad: {', '.join(name(p) for p in miss)}. Predicted cost of the "
                     f"manager's XI against the full-strength usual XI: npxG for {r.actual_raw_npxg_for - r.full_usual_raw_npxg_for:+.2f}, "
                     f"against {r.actual_raw_npxg_against - r.full_usual_raw_npxg_against:+.2f} (EP {r.actual_ep_raw - r.full_usual_ep_raw:+.3f}). "
                     f"Realized npxG {r.npxg_for_actual:.2f}–{r.npxg_against_actual:.2f}.")
    lines.append(f"Manager's XI: {r.recommended_overlap}/11 in common with the recommendation, "
                 f"{r.usual_overlap}/11 with the usual XI; model EP of the manager's XI {r.actual_ep:.2f}.")
    if len(sw) == 0:
        return "\n".join(lines)
    # ---- descriptive layer
    lines.append("*Descriptive only (does not add expected points):*")
    d = NP.days(pd.Series([m.date]))[0]
    pmod = position_model(m.date)
    su = json.loads(r.usual_slots)
    sb = json.loads(r.best_slots)
    su, sb = {int(k): v for k, v in su.items()}, {int(k): v for k, v in sb.items()}
    xu, xb = place(pmod, team_id, r.formation, su, d), place(pmod, team_id, r.formation, sb, d)
    a, b = shape(xu), shape(xb)
    lines.append("- Shape (predicted positions): defensive line {:.1f} → {:.1f} m, attacking height {:.1f} → {:.1f} m, "
                 "width {:.1f} → {:.1f} m.".format(a["def_line"], b["def_line"], a["att_height"], b["att_height"],
                                                   a["width"], b["width"]))
    # who faces whom: our five most advanced vs their five deepest (their predicted positions, mirrored)
    pos = c["pos"]
    op = pos[(pos.match_id == match_id) & (pos.team_id != team_id) & ~pos.is_gk]
    bm = np.column_stack([105 - op.x_pred, 68 - op.y_pred])
    ids = list(xb)
    A = np.array([xb[p] for p in ids])
    ia, ib = np.argsort(-A[:, 0])[:5], np.argsort(bm[:, 0])[:5]
    Dm = np.hypot(*(A[ia, None, :] - bm[None, ib, :]).transpose(2, 0, 1))
    rr, cc = linear_sum_assignment(Dm)
    prof = c["prof"]
    pairs = []
    for i, j in zip(ia[rr], ib[cc]):
        pi, pj = ids[i], int(op.player_id.iloc[j])
        if pi not in set(sw.inn):
            continue
        fa, fd = prof.loc[(match_id, pi)], prof.loc[(match_id, pj)]
        pairs.append(f"{name(pi)} v {name(pj)} (dribbles {fa.dribbles:.1f}/90 v dribbled past {fd.dribbled_past:.1f}/90; "
                     f"aerial win {fa.aerial_win:.0%} v {fd.aerial_win:.0%})")
    if pairs:
        lines.append("- Who faces whom: " + "; ".join(pairs) + ".")
    opp = r.opponent_id
    vals = lane_vuln(opp, m.date)
    if vals is not None:
        # lanes are in the shooting team's frame, i.e. ours when we attack them
        k = int(np.argmax(vals))
        lines.append(f"- Where {opp_name} concede: most from attacks down the {LANES[k]} "
                     f"({vals[k]:.2f}× the league's share of npxG from that lane).")
    return "\n".join(lines)


if __name__ == "__main__":
    a = sys.argv[1:]
    for mid, tid in zip(a[::2], a[1::2]):
        print(card(int(mid), int(tid)), "\n")
