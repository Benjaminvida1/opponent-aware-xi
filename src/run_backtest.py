"""Step 5: lineup picker back-test on 2025/26 + 2026/27, with L1 (methodology §13, §14, §15.4).

Run:  python src/run_backtest.py            (tunes the usual-XI memory on 2024/25, then the test back-test)
      python src/run_backtest.py --report   (tables and checks from the saved decisions only)

For every test team-match, as of the match date and against the opponent's actual XI:
  usual XI        the manager's usual personnel in the teamsheet formation, ignoring the opponent (§13.4 baseline)
  neutral best    the XI with the highest EP against a league-average opponent (quality, not opponent)
  best            the XI with the highest EP against this opponent (strength and XI)
  recommended     best if P(better than usual) > 0.7, else the usual XI (§13.4)
  actual          the manager's XI
The usual-XI memory (half-life of the slot-start weights) is chosen on 2024/25 by agreement with managers' XIs;
nothing is chosen on the test seasons.

Outputs: data/processed/step5_decisions.parquet, step5_swaps.parquet; reports/step5_*.csv / .json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import network as N  # noqa: E402
from lineup import (L1_CFG, THRESHOLD, DayModel, History, days, decide, laplace_draws, swap_breakdown,  # noqa: E402
                    usual_xi)
from outcome import PROC, REPORTS, TEST, TUNE, StintData, fit, player_effects, predict_matches, sub_averages  # noqa: E402
from team_strength import paired_bootstrap  # noqa: E402

USUAL_HL = [15.0, 30.0, 60.0, 120.0, 240.0, 365.0, np.inf]


def full_pool(sq_all, team, date, season, back_days=60):
    """Players the team could pick at full strength: in any of its squads in the previous back_days, and still at
    the club (in one of its squads on or after the date, same season). The second condition uses later squads only
    to define the evaluation sample (§15.4 forced changes); no prediction uses it."""
    t = sq_all[(sq_all.team_id == team) & (sq_all.season == season)]
    recent = set(t[(t.mdate < date) & (t.mdate >= date - pd.Timedelta(days=back_days))].player_id)
    still = set(t[t.mdate >= date].player_id)
    keep = recent & still
    gks = set(t[t.is_gk].player_id)
    return sorted(keep - gks), sorted(keep & gks)


def group_means(data, th, info, prev):
    """As-of mean o and d of outfield starters by position group, over the previous 120 days' matches."""
    xi = data.pt[data.pt.started & ~data.pt.is_gk & data.pt.match_id.isin(prev.match_id)]
    o, d, _, _ = player_effects(data, th, info["zs"], xi.match_id.to_numpy(), xi.player_id.to_numpy())
    grp = data.prof.group.reindex(pd.MultiIndex.from_frame(xi[["match_id", "player_id"]])).to_numpy()
    t = pd.DataFrame(dict(g=grp, o=o, d=d))
    out = {g: (float(r.o), float(r.d)) for g, r in t.dropna().groupby("g")[["o", "d"]].mean().iterrows()}
    out["all"] = (float(t.o.mean()), float(t.d.mean()))
    return out


def squads(hist: History):
    pm = hist.pm
    return {k: g[["player_id", "is_gk", "started", "played", "player_name"]].reset_index(drop=True)
            for k, g in pm.groupby(["match_id", "team_id"])}


# --------------------------------------------------------------------------- usual-XI memory, chosen on 2024/25
def tune_usual(data, hist, sq):
    m = data.matches[data.matches.season.isin(TUNE)]
    rows = []
    for hl in USUAL_HL:
        ov = []
        for mt in m.itertuples():
            d = days(pd.Series([mt.date]))[0]
            for team, home in ((mt.home_team_id, 1), (mt.away_team_id, 0)):
                s = sq[(mt.match_id, team)]
                out = s.loc[~s.is_gk, "player_id"].tolist()
                F = N.sheet_template(mt.home_formation if home else mt.away_formation) or "4-2-3-1"
                xi = usual_xi(hist, mt.match_id, team, out, F, d, 1e9 if np.isinf(hl) else hl)[0]
                act = set(s.loc[s.started & ~s.is_gk, "player_id"])
                ov.append(len(act & set(xi)))
        rows.append(dict(half_life=hl, mean_outfield_overlap=float(np.mean(ov)), n=len(ov)))
        print(rows[-1], flush=True)
    g = pd.DataFrame(rows)
    g.to_csv(REPORTS / "step5_usual_grid.csv", index=False)
    best = g.mean_outfield_overlap.max()
    # ties within 0.02 players go to the longer memory (more "usual")
    return float(g[g.mean_outfield_overlap >= best - 0.02].half_life.max())


# --------------------------------------------------------------------------- back-test
def backtest(usual_hl, seasons=TEST, limit_days=None):
    data = StintData()
    hist = History(data)
    sq = squads(hist)
    l0 = pd.read_parquet(PROC / "l0_predictions_all.parquet").set_index("match_id")
    m = data.matches[data.matches.season.isin(seasons)]
    dec_rows, swap_rows = [], []
    theta = None
    pnames = hist.pm.drop_duplicates("player_id").set_index("player_id").player_name
    sq_all = hist.pm[["match_id", "team_id", "player_id", "is_gk", "mdate", "season"]]
    act_np = data.st.groupby(["match_id", "team_id"]).npxg.sum()
    for k, (day, g) in enumerate(m.groupby(m.date.dt.floor("D"))):
        if limit_days and k >= limit_days:
            break
        t0 = time.time()
        as_of = g.date.min()
        th, info = fit(data, as_of, L1_CFG, theta0=theta)
        theta = th
        prev = data.matches[(data.matches.date < as_of) & (data.matches.date >= as_of - pd.Timedelta(days=120))]
        p = predict_matches(data, th, dict(info, scale=1.0), prev)
        act = data.st.groupby(["match_id", "team_id"]).npxg.sum()
        a = act.reindex(pd.MultiIndex.from_frame(p[["match_id", "team_id"]])).to_numpy()
        info["scale"] = float(a.sum() / p.npxg_pred.sum())
        draws, phi = laplace_draws(data, th, info, L1_CFG, seed=k)
        dm = DayModel(data, th, info, draws, sub_averages(data, th, info["zs"], as_of), (p.O.mean(), p.D_opp.mean()), hist)
        dm.group_means = group_means(data, th, info, prev)
        for mt in g.itertuples():
            r0 = l0.loc[mt.match_id]
            conv = dict(c=r0.calib_c, rho=r0.dc_rho, pen_home=r0.pen_home, pen_away=r0.pen_away)
            for team in (mt.home_team_id, mt.away_team_id):
                opp = mt.away_team_id if team == mt.home_team_id else mt.home_team_id
                so = sq[(mt.match_id, opp)]
                fp = full_pool(sq_all, team, mt.date, mt.season)
                cur = sq[(mt.match_id, team)]
                fp = (sorted(set(fp[0]) | set(cur.loc[~cur.is_gk, "player_id"])),
                      sorted(set(fp[1]) | set(cur.loc[cur.is_gk, "player_id"])))
                dec, ctx = decide(dm, mt, team, cur, so[so.started], conv, usual_hl, full_pool=fp)
                if ctx is None:
                    continue
                row = dict(match_id=mt.match_id, date=mt.date, season=mt.season, round=mt.round, team_id=team,
                           opponent_id=opp, home=int(team == mt.home_team_id), formation=dec.formation,
                           keeper=dec.keeper, keeper_actual=dec.keeper_actual, p_better=dec.p_better, gA=dec.gA,
                           gB=dec.gB, phi=phi, notes="; ".join(dec.notes),
                           npxg_for_actual=float(act_np.get((mt.match_id, team), np.nan)),
                           npxg_against_actual=float(act_np.get((mt.match_id, opp), np.nan)),
                           l0_npxg_for=float(r0.npxg_home_pred if team == mt.home_team_id else r0.npxg_away_pred),
                           l0_npxg_against=float(r0.npxg_away_pred if team == mt.home_team_id else r0.npxg_home_pred),
                           points=float(3 * (mt.home_score > mt.away_score) + (mt.home_score == mt.away_score)
                                        if team == mt.home_team_id else
                                        3 * (mt.away_score > mt.home_score) + (mt.home_score == mt.away_score)))
                in_squad = set(cur.player_id)
                fx = dec.xis.get("full_usual", [])
                missing = [x for x in fx if x not in in_squad]
                if getattr(dec, "keeper_full", None) is not None and dec.keeper_full not in in_squad:
                    missing.append(dec.keeper_full)
                row["regulars_missing"] = json.dumps([int(x) for x in missing])
                for name, sc in dec.scores.items():
                    for kk, v in sc.items():
                        row[f"{name}_{kk}"] = v
                for name, xi in dec.xis.items():
                    row[f"{name}_xi"] = json.dumps([int(x) for x in xi])
                    if name in dec.slots:
                        row[f"{name}_slots"] = json.dumps({int(p_): s for p_, s in dec.slots[name].items()})
                act_set = set(dec.xis["actual"])
                for name in ("usual", "neutral", "best", "recommended"):
                    row[f"{name}_overlap"] = len(act_set & set(dec.xis[name])) + int(dec.keeper == dec.keeper_actual)
                    row[f"{name}_changes"] = 10 - len(set(dec.xis["usual"]) & set(dec.xis[name]))
                dec_rows.append(row)
                for name in ("best", "neutral"):
                    for s in swap_breakdown(dec, ctx, "usual", name):
                        swap_rows.append(dict(match_id=mt.match_id, team_id=team, xi=name, **s,
                                              out_name=pnames.get(s["out"]), in_name=pnames.get(s["inn"])))
        print(f"{str(day)[:10]} {len(g)} matches, phi {phi:.2f}, {time.time() - t0:.0f}s", flush=True)
    D = pd.DataFrame(dec_rows)
    S = pd.DataFrame(swap_rows)
    D.to_parquet(PROC / "step5_decisions.parquet", index=False)
    S.to_parquet(PROC / "step5_swaps.parquet", index=False)
    return D, S


# --------------------------------------------------------------------------- report
def boot_slope(x, y, groups, n_boot=2000, seed=0):
    """OLS slope of y on x with a bootstrap over matches (both sides of a match resampled together)."""
    x, y, groups = np.asarray(x, float), np.asarray(y, float), np.asarray(groups)
    if len(x) < 5:
        return (np.nan,) * 4
    def slope(xx, yy):
        xc = xx - xx.mean()
        return float((xc * (yy - yy.mean())).sum() / (xc ** 2).sum())
    b = slope(x, y)
    ug = np.unique(groups)
    pos = {g: np.flatnonzero(groups == g) for g in ug}
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        idx = np.concatenate([pos[g] for g in rng.choice(ug, len(ug))])
        bs.append(slope(x[idx], y[idx]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return b, float(lo), float(hi), float(np.corrcoef(x, y)[0, 1])


def report():
    D = pd.read_parquet(PROC / "step5_decisions.parquet")
    out = {}
    names = ["actual", "usual", "neutral", "best", "recommended", "full_usual"]
    # 1. model-scored expected points (circular: a sanity check, §15.4.3)
    rows = []
    pm = D.groupby("match_id")
    for n in names:
        rows.append(dict(xi=n, ep=D[f"{n}_ep"].mean(), npxg_for=D[f"{n}_npxg_for"].mean(),
                         npxg_against=D[f"{n}_npxg_against"].mean(), n=int(D[f"{n}_ep"].notna().sum())))
    ep = pd.DataFrame(rows)
    comps = []
    for a, b in [("recommended", "usual"), ("best", "usual"), ("neutral", "usual"), ("best", "neutral"),
                 ("actual", "usual"), ("recommended", "actual"), ("usual", "full_usual")]:
        dd = (D[f"{a}_ep"] - D[f"{b}_ep"]).groupby(D.match_id).mean().dropna().sort_index()
        m_, lo, hi = paired_bootstrap(dd.to_numpy(), np.zeros(len(dd)))
        comps.append(dict(comparison=f"{a} minus {b}", ep_per_match=m_, ci_low=lo, ci_high=hi,
                          per_38_matches=38 * m_))
    comps = pd.DataFrame(comps)
    ep.to_csv(REPORTS / "step5_ep_by_xi.csv", index=False)
    comps.to_csv(REPORTS / "step5_ep_comparisons.csv", index=False)
    # 2. how often a change is recommended
    ch = D.best_changes > 0
    rec = D.recommended_changes > 0
    out["decisions"] = dict(team_matches=len(D), best_differs_from_usual=float(ch.mean()),
                            recommended_change=float(rec.mean()),
                            mean_changes_best=float(D.best_changes.mean()),
                            mean_changes_when_recommended=float(D.loc[rec, "recommended_changes"].mean()) if rec.any() else 0,
                            p_better_quantiles=D.p_better.dropna().quantile([.1, .25, .5, .75, .9]).round(3).to_dict(),
                            neutral_differs_from_usual=float((D.neutral_changes > 0).mean()),
                            best_differs_from_neutral=float((D.best_xi != D.neutral_xi).mean()),
                            mean_gain_when_recommended=float((D.recommended_ep - D.usual_ep)[rec].mean()) if rec.any() else 0)
    # 3. agreement with managers (§15.4.1), out of 11
    ov = pd.DataFrame([dict(xi=n, mean_overlap=D[f"{n}_overlap"].mean(), share_9_plus=(D[f"{n}_overlap"] >= 9).mean(),
                            share_11=(D[f"{n}_overlap"] == 11).mean()) for n in ["usual", "neutral", "best", "recommended"]])
    ov.to_csv(REPORTS / "step5_overlap.csv", index=False)
    # 4. underdog / favourite: how the opponent changes the best XI (best vs neutral best)
    D["p_win_usual"] = D.usual_p_win
    D["tier"] = pd.cut(D.usual_p_win, [0, .25, .45, .65, 1], labels=["big underdog", "underdog", "even/slight fav", "favourite"])
    tier = D.groupby("tier", observed=True).apply(lambda g: pd.Series(dict(
        n=len(g), gA_over_gB=(g.gA / g.gB).mean(), best_differs_from_neutral=(g.best_xi != g.neutral_xi).mean(),
        dO_best_vs_neutral=(g.best_OA - g.neutral_OA).mean(), dD_best_vs_neutral=(g.best_DA - g.neutral_DA).mean(),
        recommended_change=(g.recommended_changes > 0).mean())))
    tier.to_csv(REPORTS / "step5_by_tier.csv")
    # 5. is L1's sensitivity to lineup changes right? (out of sample; not circular)
    cal = []
    dif = D[D.actual_xi.apply(json.loads).apply(set) != D.usual_xi.apply(json.loads).apply(set)]
    for side in ("for", "against"):
        x = dif[f"actual_raw_npxg_{side}"] - dif[f"usual_raw_npxg_{side}"]
        y = dif[f"npxg_{side}_actual"] - dif[f"usual_raw_npxg_{side}"]
        b, lo, hi, r = boot_slope(x, y, dif.match_id)
        cal.append(dict(check="manager's XI vs usual XI", side=side, n=len(dif), slope=b, ci_low=lo, ci_high=hi, corr=r,
                        mean_pred_change=float(x.mean()), mean_realized=float(y.mean())))
    # 6. forced changes (§15.4.2): a regular missing from the squad
    D["n_missing"] = D.regulars_missing.apply(lambda s: len(json.loads(s)))
    fc = D[(D.n_missing > 0) & D.full_usual_ep.notna()]
    for side in ("for", "against"):
        x = fc[f"actual_raw_npxg_{side}"] - fc[f"full_usual_raw_npxg_{side}"]
        for ref, yv in (("full-strength L1", fc[f"npxg_{side}_actual"] - fc[f"full_usual_raw_npxg_{side}"]),
                        ("L0", fc[f"npxg_{side}_actual"] - fc[f"l0_npxg_{side}"])):
            b, lo, hi, r = boot_slope(x, yv, fc.match_id)
            sign = float((np.sign(x) == np.sign(yv))[x.abs() > 0.02].mean())
            cal.append(dict(check=f"forced changes, realized vs {ref}", side=side, n=len(fc), slope=b, ci_low=lo,
                            ci_high=hi, corr=r, mean_pred_change=float(x.mean()), mean_realized=float(yv.mean()),
                            sign_agreement=sign))
    cal = pd.DataFrame(cal)
    cal.to_csv(REPORTS / "step5_calibration.csv", index=False)
    out["forced"] = dict(team_matches_with_missing_regular=int(len(fc)), share=float(len(fc) / len(D)),
                         mean_missing=float(fc.n_missing.mean()),
                         ep_full_minus_actual=float((fc.full_usual_ep - fc.actual_ep).mean()))
    # 7. sanity: actual-XI npxG here equals L1's test predictions
    l1 = pd.read_parquet(PROC / "l1_predictions.parquet")
    s_ = pd.concat([l1[["match_id", "npxg_home_pred"]].assign(home=1).rename(columns={"npxg_home_pred": "p"}),
                    l1[["match_id", "npxg_away_pred"]].assign(home=0).rename(columns={"npxg_away_pred": "p"})])
    j = D.merge(s_, on=["match_id", "home"])
    out["check_vs_l1_predictions"] = dict(n=len(j), max_abs_diff=float((j.actual_raw_npxg_for - j.p).abs().max()),
                                          mean_abs_diff=float((j.actual_raw_npxg_for - j.p).abs().mean()))
    (REPORTS / "step5_summary.json").write_text(json.dumps(out, indent=2, default=float))
    pd.set_option("display.width", 220)
    for t in (ep, comps, ov, tier, cal):
        print(t.round(4).to_string(), "\n")
    print(json.dumps(out, indent=2, default=float))
    return D


if __name__ == "__main__":
    if "--report" not in sys.argv:
        data = StintData()
        hist = History(data)
        gpath = REPORTS / "step5_usual_grid.csv"
        if gpath.exists():
            g = pd.read_csv(gpath)
            best = g.mean_outfield_overlap.max()
            hl = float(g[g.mean_outfield_overlap >= best - 0.02].half_life.max())
        else:
            hl = tune_usual(data, hist, squads(hist))
        print("usual-XI half-life", hl, flush=True)
        lim = int(sys.argv[sys.argv.index("--days") + 1]) if "--days" in sys.argv else None
        backtest(1e9 if np.isinf(hl) else hl, limit_days=lim)
    report()
