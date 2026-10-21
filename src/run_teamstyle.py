"""Step 6 follow-up: the style clash on top of team strength alone, with no player terms (methodology §15.2).

Question: does the style clash carry any signal when the model does not know who is playing? L2 tested it on top
of L1 (players); here it is tested on top of the team-only stint model (L1 with the player terms switched off,
which ties L0 on test).

Models (all fitted on stints, as-of, with L1's pre-match path and L0's goals conversion):
  T0   team effects on stints, no players (team ridge 4: the setting chosen for it in Step 3's grid)
  TS   T0 + style clash (all 16 products of the two teams' four style scores, one ridge penalty)
The style penalty is chosen on 2024/25 with the Step 4 rule (weekly refits for every candidate; lowest npxG
deviance, settings within 0.5% tie, lowest RPS among them), over the same grid as L2: 10, 100, 1,000, 10,000.
The test report refits every match day and compares TS with T0, L0, L1 and L2 by paired bootstrap over matches
and over whole rounds.

Run:  python src/run_teamstyle.py        (about 1-1.5 hours on two cores)
Outputs (reports/): teamstyle_grid.csv, teamstyle_test_scores.csv, teamstyle_comparisons.csv,
  teamstyle_summary.json; data/processed/teamstyle_predictions.parquet
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from outcome import PROC, REPORTS, TEST, TUNE, Config, StintData, rolling  # noqa: E402
from robustness import cluster_bootstrap  # noqa: E402
from run_l1 import l0_predictions, per_match, scores, to_match  # noqa: E402
from team_strength import paired_bootstrap  # noqa: E402

TIE_TOL = 0.005
GAP_TUNE = 7
STYLE_LAMS = [10.0, 100.0, 1000.0, 10000.0]
TEAM_KW = dict(lam_tau=4.0, use_players=False)

_DATA = None


def data():
    global _DATA
    if _DATA is None:
        _DATA = StintData()
    return _DATA


def _run(args):
    name, cfg, seasons, gap = args
    return name, cfg, rolling(data(), cfg, seasons, min_gap_days=gap)


def run_many(jobs):
    with Pool(2) as pool:
        return list(pool.imap_unordered(_run, jobs))


def main():
    l0 = l0_predictions()
    tune_l0, test_l0 = l0[l0.season.isin(TUNE)], l0[l0.season.isin(TEST)]
    gpath = REPORTS / "teamstyle_grid.csv"
    if gpath.exists():
        g = pd.read_csv(gpath)
    else:
        jobs = [("T0", Config(**TEAM_KW), TUNE, GAP_TUNE)]
        jobs += [(f"TS {lam:g}", Config(**TEAM_KW, extra=(("styleint:full", lam),)), TUNE, GAP_TUNE) for lam in STYLE_LAMS]
        rows = []
        for name, cfg, p in run_many(jobs):
            s = scores(to_match(p, tune_l0))
            rows.append(dict(name=name, lam=(cfg.extra[0][1] if cfg.extra else np.nan), tune_npxg_dev=s["npxg_deviance"],
                             tune_rps=s["rps"], tune_ll=s["log_loss"]))
            print(rows[-1], flush=True)
        g = pd.DataFrame(rows).sort_values("name")
        g.to_csv(gpath, index=False)
    c = g[g.name.str.startswith("TS")]
    best = c.tune_npxg_dev.min()
    lam = float(c[c.tune_npxg_dev <= best * (1 + TIE_TOL)].sort_values("tune_rps").iloc[0].lam)

    res = run_many([("T0", Config(**TEAM_KW), TEST, 0),
                    ("TS", Config(**TEAM_KW, extra=(("styleint:full", lam),)), TEST, 0)])
    preds = {"L0": test_l0}
    for name, _, p in res:
        preds[name] = to_match(p, test_l0)
    preds["L1"] = pd.read_parquet(PROC / "l1_predictions.parquet")
    preds["L2"] = pd.read_parquet(PROC / "l2_predictions.parquet")
    pd.concat([preds[n].assign(model=n) for n in ("T0", "TS")], ignore_index=True).to_parquet(
        PROC / "teamstyle_predictions.parquet", index=False)

    ids = np.sort(test_l0.match_id.to_numpy())
    for n, q in preds.items():
        assert np.array_equal(np.sort(q.match_id.to_numpy()), ids), n
    rnd = test_l0.set_index("match_id")
    cl = (rnd.season.astype(str) + " R" + rnd["round"].astype(str)).reindex(ids).to_numpy()
    pm_ = {n: per_match(q) for n, q in preds.items()}
    table = pd.DataFrame([dict(model=n, **scores(q)) for n, q in preds.items()])
    rows = []
    for a, b in [("TS", "T0"), ("TS", "L0"), ("T0", "L0"), ("L1", "TS"), ("L2", "L1")]:
        for metric in ["rps", "log_loss", "npxg_deviance"]:
            d = pm_[a][metric] - pm_[b][metric]
            dm, lo, hi = paired_bootstrap(pm_[a][metric], pm_[b][metric])
            _, lo_r, hi_r, _ = cluster_bootstrap(d, cl)
            rows.append(dict(comparison=f"{a} minus {b}", metric=metric, diff=dm, ci_low_match=lo, ci_high_match=hi,
                             ci_low_round=lo_r, ci_high_round=hi_r))
    comps = pd.DataFrame(rows)
    table.to_csv(REPORTS / "teamstyle_test_scores.csv", index=False)
    comps.to_csv(REPORTS / "teamstyle_comparisons.csv", index=False)
    (REPORTS / "teamstyle_summary.json").write_text(json.dumps(dict(
        chosen_style_penalty=lam, team_kw=TEAM_KW, grid=g.to_dict("records"),
        test=table.set_index("model").to_dict("index")), indent=2, default=float))
    return g, table, comps


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    g, t, c = main()
    print(g.round(4).to_string(index=False))
    print(t.round(4).to_string(index=False))
    print(c.round(4).to_string(index=False))
