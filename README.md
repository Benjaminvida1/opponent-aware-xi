# Beyond the Best XI: Opponent-Aware Lineup Selection

Code and processed data for *Selecting the Starting XI Against a Specific Opponent: Evidence from Four Premier League Seasons* (working paper, prepared for the MIT Sloan Sports Analytics Conference Research Paper Competition, soccer track).

The project asks which starting XI maximises a team's expected points against a specific opponent, using public Premier League data: player box scores, shot-level expected goals and full-match passing networks. It builds the full model layer by layer and tests each layer on held-out matches.

**Main finding.** Knowing who plays improves forecasts; knowing how the two teams' styles and players line up against each other does not, at any size these data can detect. The best XI is nearly the same against every opponent.

## Results

All results are on the 430 test matches (2025/26 and 2026/27 rounds 1–5). Every setting was chosen on 2024/25; 2023/24 is warm-up only.

| Result | Value | Reproduced by |
| --- | --- | --- |
| Team-strength baseline (L0), ranked probability score | 0.2101 (base rates 0.2278, closing odds 0.2048) | `src/run_l0.py` → `reports/l0_*.csv` |
| Player model on stints (L1) vs L0 | 0.2071; −0.0029 (95% CI −0.0048 to −0.0011) | `src/run_l1.py` → `reports/l1_*.csv` |
| Style clash (L2) vs L1 | −0.0001 (−0.0012 to +0.0010) | `src/run_l23.py` → `reports/l23_*.csv` |
| Matchups (L3) vs L2 | 0.0000 (−0.0001 to +0.0001) | `reports/l23_comparisons.csv` |
| L3 vs L3 built without the passing network | 0.0000 (−0.0010 to +0.0010) | `reports/l23_comparisons.csv` |
| Lineup picker: players shared with the manager's XI (usual XI / recommended) | 9.0 / 8.2 of 11 | `src/run_backtest.py` → `reports/step5_*.csv` |
| Selections where the opponent changes the best XI | 8.5%, worth 0.0001 expected points | `reports/step5_summary.json` |
| Round-level bootstrap (43 rounds instead of 430 matches) | same verdicts; intervals 0.82–1.17× as wide | `src/robustness.py` → `reports/robustness_round_bootstrap.csv` |
| Player model fitted on whole matches instead of stints, vs L1 | +0.0015 (+0.0006 to +0.0024): about half of L1's gain needs stints | `reports/robustness_match_level.csv` |

## Dashboard

**[Explore the interactive dashboard](https://benjaminvida1.github.io/opponent-aware-xi/)**: every match's shots, passing networks and team shapes in club colours; the model's forecast and lineup call for each test match; player radars (percentile or standardised, with the model's impact ratings); team strength over time and playing style.

The site is static (`docs/`, served by GitHub Pages). Rebuild its data after changing the processed tables:

```bash
python src/build_site.py     # writes docs/data/ (about a minute)
python tests/test_viz.py     # checks for the visual layer
```

The plotting code (`src/viz_*.py`) also draws the same charts as PNGs; `reports/dashboard/README.md` documents the data and design rules.

## Quick start

```bash
git clone https://github.com/Benjaminvida1/opponent-aware-xi
cd opponent-aware-xi
pip install -r requirements.txt

python src/run_l0.py         # L0 baseline and evaluation (seconds)
python src/run_l1.py         # L1 player model vs L0 (about 25 minutes)
python src/run_l23.py        # L2 and L3 vs L1, network ablation (about 1.5 hours)
python src/run_backtest.py   # lineup picker back-test (about 10 minutes)
python src/robustness.py     # round-level bootstrap and match-level L1 (about 45 minutes)
python src/explain.py        # explanation cards for the showcase fixtures
```

Every script after `ingest.py` reads `data/processed/` and writes to `reports/`. The processed tables published here already contain the outputs of the intermediate steps (`network.py`, `style.py`, `profiles.py`, `roles.py`, `stints.py`, `gk.py`, `netpred.py`, `matchups.py`); each of those scripts rebuilds its tables from the earlier ones. Add `--grid` to `run_l0.py`, `run_l1.py` or `run_l23.py` to redo the tuning grid on 2024/25. The tests in `tests/` check the as-of rule (no information from after the prediction date) for each step.

## Repository layout

```
src/ingest.py            raw scraped files -> cleaned tables, with every cleaning rule (needs the raw files, not published)
src/team_strength.py     L0 team ratings, expected points, scoring rules, bootstrap
src/run_l0.py            L0 tuning and test report, vs base rates and closing odds
src/paper_analyses.py    network reliability, team vs opponent variance, npxG vs points
src/network.py           passing-network metrics per team-match, slots, lines in possession
src/style.py             own vs imposed style, style embedding
src/profiles.py          player profiles as of every match (shrinkage, recency, possession and opponent adjustment)
src/roles.py             ten player roles (Gaussian mixture), memberships as of every match
src/stints.py            stints with constant players, score and man count
src/outcome.py           L1 player-effects model on stints; L2/L3 extra terms
src/gk.py                goalkeeper shot-stopping (estimated, not used in forecasts)
src/run_l1.py            L1 tuning and test report, ablations, player ratings
src/netpred.py           predicted positions and passing network for any XI
src/matchups.py          style scores and the six matchup features
src/run_l23.py           L2/L3 tuning and test report, network ablation
src/lineup.py            expected points for any XI; usual, best and recommended XIs; P(better)
src/run_backtest.py      back-test of the lineup picker
src/explain.py           explanation cards
src/robustness.py        round-level bootstrap; match-level L1 without stints
src/step*_figures.py     figures in reports/figures/
tests/                   checks for each step
reports/                 every result file
memos/                   plain-language memo for each build step
```

## Data

Coverage: 1,190 Premier League matches, 2023/24 to 2026/27 round 5; 31,362 shots; 262,126 passer–receiver links.

| Table | One row per | Main contents |
| --- | --- | --- |
| `matches` | match | date (UTC), teams, score, possession, teamsheet formations, managers |
| `player_match` | squad member per match | minutes and timing, box-score, passing and defensive statistics; shot-derived xG |
| `shots` | shot | minute, location (metres), xG, xG on target, situation, outcome |
| `network_nodes`, `network_edges` | appearance; passer–receiver pair | average position, passes completed |
| `stints`, `stint_players` | stint | players on the pitch, score, man count, npxG |
| `player_profile_asof`, `player_roles_asof` | squad member per match | 38 profile measurements and 10 role probabilities, as of the match |
| `team_match_network`, `team_style_asof` | team per match | network metrics; own and imposed style, as of the match |
| `l0_predictions_all`, `l1_predictions`, `l2_predictions`, `l3_predictions` | match | forecasts of each model |
| `step5_decisions`, `step5_swaps` | team selection; swap | back-test of the lineup picker |

Conventions: pitch 105 × 68 m from each team's own perspective (own goal at x = 0, attacking toward x = 105, y = 68 on the team's left). For players who appeared, a blank count in the source means zero. Every prediction uses only matches played before the prediction date.

### Provenance

The raw data were collected from a public match-statistics website. The raw scraped files are not redistributed; this repository publishes the processed tables derived from them, which reproduce every reported result. `src/ingest.py` documents each cleaning step and rebuilds the tables when the raw files are placed in `data/raw/`.

## License

Code: MIT (see `LICENSE`). The data tables are derived from third-party match statistics and are provided for research reproducibility; they are not covered by the MIT license.
