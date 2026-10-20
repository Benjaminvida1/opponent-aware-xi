"""Step 4: opponent-aware layers L2 (style clash) and L3 (matchups) against L1 (methodology §9.1, §15.2-15.3).

Run:  python src/run_l23.py --grid   (tuning grid on 2024/25, then the test report)
      python src/run_l23.py          (test report with the settings chosen from the saved grid)

L1's settings are kept (team 4, personal 300, box-score 10,000). The new terms get their own ridge penalties,
chosen on 2024/25 with Step 3's rule: lowest npxG deviance, settings within 0.5% tie, lowest RPS among them.
Each layer is compared with the previous one on the test seasons by paired bootstrap over matches (§15.2).
The tuning grid refits weekly (every candidate alike, to keep it short); the test report refits every match day, as L1.

L2 = L1 + style clash: every product of the attacking and defending teams' four style scores (16 terms, the
     full-rank form of s_A' U V' s_B, held toward 0 by one ridge penalty); optionally with the style main effects.
L3 = L2 + the six matchup features (zone mismatch wide / half-spaces / centre, dribble, aerial, press vs build-up);
     kernel bandwidth and the lane-vulnerability option tuned on 2024/25.
Network ablation (§15.3), all with the chosen penalties: L3 with features built (a) without the passing network,
     (b) with its geometry only, (c) in full (matchups.py).

Outputs (reports/): l23_grid.csv, l23_chosen.json, l23_test_scores.csv, l23_comparisons.csv, l23_by_season.csv,
  l23_coefficients.csv, l23_summary.json; data/processed/l2_predictions.parquet, l3_predictions.parquet
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from outcome import PROC, REPORTS, TEST, TUNE, Config, StintData, fit, rolling, unpack  # noqa: E402
from run_l1 import l0_predictions, per_match, scores, to_match  # noqa: E402
from team_strength import paired_bootstrap  # noqa: E402

TIE_TOL = 0.005
GAP_TUNE = 7                  # tuning grid: refit weekly (the same for every candidate), to keep the grid short
GAP_TEST = 0                  # test report: refit every match day, as L1
L1_KW = dict(lam_tau=4.0, lam_u=300.0, lam_b=10000.0)
STYLE_LAMS = [10.0, 100.0, 1000.0, 10000.0]
MATCH_LAMS = [30.0, 300.0, 3000.0]


def zones(b, v):
    return [f"zone_wide_b{b:g}{v}", f"zone_half_b{b:g}{v}", f"zone_centre_b{b:g}{v}"]


def matchup_spec(b=10.0, v="v", variant="full"):
    if variant == "none":
        return "cols:" + ",".join(["dribble_nopos", "aerial_nopos", "press_none"])
    press = "press" if variant == "full" else "press_geo"
    return "cols:" + ",".join(zones(b, v) + ["dribble", "aerial", press])


_DATA = None


def data():
    global _DATA
    if _DATA is None:
        _DATA = StintData()
    return _DATA


def _run(args):
    name, cfg, seasons, gap = args
    return name, cfg, rolling(data(), cfg, seasons, min_gap_days=gap)


def run_many(jobs, gap=GAP_TUNE):
    with Pool(2) as pool:
        return list(pool.imap_unordered(_run, [j + (gap,) for j in jobs]))


def tune_rows(res, l0):
    tl0 = l0[l0.season.isin(TUNE)]
    rows = []
    for name, cfg, p in res:
        s = scores(to_match(p, tl0))
        rows.append(dict(name=name, key=cfg.key(), extra=json.dumps(cfg.extra), tune_npxg_dev=s["npxg_deviance"],
                         tune_rps=s["rps"], tune_ll=s["log_loss"]))
        print(rows[-1], flush=True)
    return rows


def pick(g: pd.DataFrame, prefix: str) -> pd.Series:
    c = g[g.name.str.startswith(prefix)]
    best = c.tune_npxg_dev.min()
    tied = c[c.tune_npxg_dev <= best * (1 + TIE_TOL)]
    return tied.sort_values("tune_rps").iloc[0]


def grid(l0):
    rows = []
    base = Config(**L1_KW)
    # stage 1: L1 and the style-clash penalty
    jobs = [("L1", base, TUNE)] + [(f"L2 int {lam:g}", Config(**L1_KW, extra=((f"styleint:full", lam),)), TUNE)
                                   for lam in STYLE_LAMS]
    rows += tune_rows(run_many(jobs), l0)
    g = pd.DataFrame(rows)
    lam_s = json.loads(pick(g, "L2").extra)[0][1]
    # stage 2: style main effects as well; matchup penalty at bandwidth 10 m with vulnerability
    l2 = ((f"styleint:full", lam_s),)
    jobs = [("L2 int+main", Config(**L1_KW, extra=l2 + (("stylemain:full", lam_s),)), TUNE)]
    jobs += [(f"L3 b10v {lam:g}", Config(**L1_KW, extra=l2 + ((matchup_spec(10, "v"), lam),)), TUNE) for lam in MATCH_LAMS]
    rows += tune_rows(run_many(jobs), l0)
    g = pd.DataFrame(rows)
    l2_best = json.loads(pick(g, "L2").extra)
    l2_best = tuple(tuple(x) for x in l2_best)
    lam_m = json.loads(pick(g, "L3").extra)[-1][1]
    # stage 3: bandwidth and vulnerability at the chosen penalty
    jobs = [(f"L3 b{b:g}{v} {lam_m:g}", Config(**L1_KW, extra=l2_best + ((matchup_spec(b, v), lam_m),)), TUNE)
            for b, v in [(6, "v"), (15, "v"), (10, "n")]]
    rows += tune_rows(run_many(jobs), l0)
    g = pd.DataFrame(rows)
    g.to_csv(REPORTS / "l23_grid.csv", index=False)
    return g


def chosen_configs(g):
    l2 = tuple(tuple(x) for x in json.loads(pick(g, "L2").extra))
    l3 = tuple(tuple(x) for x in json.loads(pick(g, "L3").extra))
    return l2, l3


def coefficients(cfg, name):
    """Extra-term coefficients of a fit on all data (as of the day after the last match)."""
    d = data()
    th, info = fit(d, d.matches.date.max() + pd.Timedelta(days=1), cfg)
    ex = unpack(d, th)["extra"]
    from outcome import extra_columns
    mf = d.matchup_table()
    names = [c for sp, _ in cfg.extra for c in extra_columns(mf.head(1), sp).columns]
    return pd.DataFrame(dict(model=name, term=names, coef=ex))


def main(run_grid=False):
    l0 = l0_predictions()
    gpath = REPORTS / "l23_grid.csv"
    g = grid(l0) if (run_grid or not gpath.exists()) else pd.read_csv(gpath)
    l2x, l3x = chosen_configs(g)
    l3_spec, l3_lam = l3x[-1]
    b = float(l3_spec.split("zone_wide_b")[1].split(",")[0][:-1]) if "zone_wide_b" in l3_spec else 10.0
    v = l3_spec.split("zone_wide_b")[1].split(",")[0][-1] if "zone_wide_b" in l3_spec else "v"
    lam_s = l2x[0][1]
    variants = {
        "L1": Config(**L1_KW),
        "L2": Config(**L1_KW, extra=l2x),
        "L3": Config(**L1_KW, extra=l3x),
        "L1 + matchups only": Config(**L1_KW, extra=(l3x[-1],)),
        "L3 without network": Config(**L1_KW, extra=tuple((sp.replace(":full", ":none"), lam) for sp, lam in l2x)
                                     + ((matchup_spec(b, v, "none"), l3_lam),)),
        "L3 network geometry only": Config(**L1_KW, extra=tuple((sp.replace(":full", ":geometry"), lam) for sp, lam in l2x)
                                           + ((matchup_spec(b, v, "geometry"), l3_lam),)),
    }
    test_l0 = l0[l0.season.isin(TEST)]
    tune_l0 = l0[l0.season.isin(TUNE)]
    res = run_many([(n, c, TEST) for n, c in variants.items()], gap=GAP_TEST)
    res += run_many([(n + " [tune]", c, TUNE) for n, c in variants.items() if n not in ("L1", "L2", "L3")])
    preds, tune_preds = {"L0": test_l0}, {}
    for n, c, p in res:
        if n.endswith(" [tune]"):
            tune_preds[n[:-7]] = to_match(p, tune_l0)
        else:
            preds[n] = to_match(p, test_l0)
    pm_ = {n: per_match(q) for n, q in preds.items()}
    table = pd.DataFrame([dict(model=n, **scores(q)) for n, q in preds.items()])
    gi = g.set_index("key")
    table["tune_rps"] = [gi.tune_rps.get(variants[n].key(), np.nan) if n in variants else np.nan for n in table.model]
    table["tune_npxg_dev"] = [gi.tune_npxg_dev.get(variants[n].key(), np.nan) if n in variants else np.nan for n in table.model]
    for n, q in tune_preds.items():
        s = scores(q)
        table.loc[table.model == n, ["tune_rps", "tune_npxg_dev"]] = [s["rps"], s["npxg_deviance"]]
    comps = []
    for a, bb in [("L2", "L1"), ("L3", "L2"), ("L3", "L1"), ("L1 + matchups only", "L1"), ("L3", "L3 without network"),
                  ("L3", "L3 network geometry only"), ("L3 network geometry only", "L3 without network"),
                  ("L3 without network", "L1")]:
        for metric in ["rps", "log_loss", "npxg_deviance"]:
            dlt, lo, hi = paired_bootstrap(pm_[a][metric], pm_[bb][metric])
            comps.append(dict(comparison=f"{a} minus {bb}", metric=metric, diff=dlt, ci_low=lo, ci_high=hi))
    comps = pd.DataFrame(comps)
    by_season = pd.DataFrame([dict(model=n, season=s_, **scores(q[q.season == s_])) for n, q in preds.items() for s_ in TEST])
    coefs = pd.concat([coefficients(variants["L2"], "L2"), coefficients(variants["L3"], "L3")])
    preds["L2"].to_parquet(PROC / "l2_predictions.parquet", index=False)
    preds["L3"].to_parquet(PROC / "l3_predictions.parquet", index=False)
    # every variant's test forecasts in one table (Step 6: round-level bootstrap in robustness.py)
    pd.concat([q.assign(model=n) for n, q in preds.items() if n != "L0"], ignore_index=True).to_parquet(
        PROC / "l23_all_predictions.parquet", index=False)
    table.to_csv(REPORTS / "l23_test_scores.csv", index=False)
    comps.to_csv(REPORTS / "l23_comparisons.csv", index=False)
    by_season.to_csv(REPORTS / "l23_by_season.csv", index=False)
    coefs.to_csv(REPORTS / "l23_coefficients.csv", index=False)
    ch = dict(L2=[list(x) for x in l2x], L3=[list(x) for x in l3x], bandwidth=b, vulnerability=(v == "v"),
              variants={n: c.key() for n, c in variants.items()})
    (REPORTS / "l23_chosen.json").write_text(json.dumps(ch, indent=2))
    (REPORTS / "l23_summary.json").write_text(json.dumps(dict(chosen=ch, test=table.set_index("model").to_dict("index")),
                                                         indent=2, default=float))
    return table, comps, by_season, coefs


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    t, c, b, k = main(run_grid="--grid" in sys.argv)
    print(t.round(4).to_string(index=False))
    print(c.round(4).to_string(index=False))
    print(b.round(4).to_string(index=False))
    print(k.round(4).to_string(index=False))
