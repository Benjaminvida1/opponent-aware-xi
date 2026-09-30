# Beyond the Best XI: Opponent-Aware Lineup Selection

Code and data for *Beyond the Best XI: Opponent-Aware Lineup Selection with Passing Networks and Expected Points*, submitted to the MIT Sloan Sports Analytics Conference Research Paper Competition (soccer track).

The project asks which starting XI maximises a team's expected points against a specific opponent, using public Premier League data: player box scores, shot-level expected goals and full-match passing networks. This repository holds the data pipeline, the team-strength baseline, its evaluation harness and the preliminary analyses reported in the abstract. Every number in the abstract can be regenerated from the files here.

## Results so far

| Result | Value | Reproduced by |
| --- | --- | --- |
| Team-strength baseline, ranked probability score on 190 held-out matches | 0.2088 (base rates 0.2293; difference −0.020, 95% CI −0.031 to −0.010) | `src/run_l0.py` → `reports/l0_test_scores.csv`, `reports/l0_comparisons.csv` |
| Passing-network metrics with season-level reliability above 0.80 | 12 of 15 | `src/paper_analyses.py` → `reports/network_reliability.csv` |
| Same, above 0.74 after removing possession | 13 of 15 | `reports/network_reliability.csv` |
| Variance in defensive line height explained by opponent vs team | R² 0.18 vs 0.20 | `reports/network_variance.csv` |
| First-half npxG difference vs first-half points as predictors of second-half points | r 0.72 vs 0.60 (20 teams) | `reports/split_half_prediction.csv` |

## Quick start

```bash
git clone <this repository>
cd <repository folder>
pip install -r requirements.txt

python src/run_l0.py            # baseline + evaluation (about 5 s); add --grid to redo the tuning grid (about 45 s)
python src/paper_analyses.py    # network reliability, team vs opponent variance, xG vs points (about 15 s)
```

Both scripts read only `data/processed/` and write to `reports/`. The notebook `notebooks/01_ingest_and_team_strength.ipynb` walks through the baseline with charts.

## Repository layout

```
data/processed/   cleaned tables (Parquet) and baseline predictions
src/ingest.py     raw scraped files -> cleaned tables; documents every cleaning rule (needs the raw files, see below)
src/team_strength.py   team-strength model, expected points, scoring rules, bootstrap
src/run_l0.py     rolling-origin evaluation of the baseline
src/paper_analyses.py  analyses behind the abstract's network and chance-quality results
notebooks/        executed walkthrough
reports/          every result file cited above
```

## Data

Coverage: 430 Premier League matches, all of 2025/26 and rounds 1–5 of 2026/27 (15 August 2025 to 20 September 2026); 662 players, 10,941 shots.

| Table | One row per | Main contents |
| --- | --- | --- |
| `matches` | match | date (UTC), teams and ids, full- and half-time score, possession shares |
| `player_match` | squad member per match | minutes, on/off minute, box-score, passing, defensive and physical statistics; shot-derived xG, npxG, xG on target |
| `shots` | shot | minute and stoppage minute, location (metres), xG, xG on target, body part, situation, outcome, own-goal flag |
| `network_nodes` | player appearance | average touch position (metres), passes completed and failed |
| `network_edges` | passer–receiver pair per match | completed passes over the match |
| `match_events` | substitution or red card | player and minute |
| `team_match_network_metrics` | team per match | the 15 passing-network metrics used in the analyses |
| `l0_predictions` | held-out match | baseline predictions and outcomes |

Conventions: pitch 105 × 68 m from each team's own perspective (own goal at x = 0, attacking toward x = 105, y = 68 on the team's left). A blank count in the source means zero for a player who appeared; physical statistics are genuinely missing in 97 matches. Every prediction uses only matches played before the prediction date.

### Provenance

The raw data were collected from a public match-statistics website. The raw scraped files are not redistributed here; this repository publishes the cleaned tables derived from them, which are sufficient to reproduce every reported result. `src/ingest.py` documents each cleaning step (id crosswalk across three player-id schemes, zero-filling, timing, coordinate conversion) and rebuilds the tables when the raw files are placed in `data/raw/`.

## License

Code: MIT (see `LICENSE`). The data tables are derived from third-party match statistics and are provided for research reproducibility; they are not covered by the MIT license.
