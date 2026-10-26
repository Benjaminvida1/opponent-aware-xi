# Dashboard visuals: data and design rules

Everything a dashboard needs to reproduce the four kept visuals: shot map, passing network, team shape (convex hull) and player radar. The static PNGs in `reports/figures/viz/` are the reference designs.

Rebuild the data from the repo root:

```
python src/ingest.py        # processed tables (about 10 s)
python src/viz_export.py    # dashboard tables + example payloads (about 20 s)
python tests/test_viz.py    # checks (about 15 s)
```

## Files

| File | Rows | What it holds |
|---|---|---|
| `data/processed/dash_shots.parquet` | one per shot (own goals dropped) | `x_m`, `y_m`, `xg`, `xgot`, `situation_label`, `outcome_label`, `minute_label`, `player_short` |
| `data/processed/dash_network_nodes.parquet` | 11 per team-match | starting XI at average positions: `name`, `shirt_number`, `x_m`, `y_m`, `passes_completed`, `minutes_played`, `subbed_off`, `is_gk` |
| `data/processed/dash_network_links.parquet` | one per pair of starters | `pass_count` (both directions added), endpoints `x_m, y_m` / `x_m_b, y_m_b` |
| `data/processed/dash_shapes.parquet` | one per team-match | `area_m2`, `width_m`, `depth_m`, `cb_height_m`, `centre_x_m`, `area_pct` (share of all team-matches with a smaller area) |
| `data/processed/dash_radar.parquet` | player × season × metric | `value` (per 90), `percentile`, `z`, `peer_mean`, `peer_median`, `n_peers`, `group`, `section` |
| `data/processed/dash_impact.parquet` | one per rated player | L1 attack / defence / net impact vs position group, with percentiles |
| `reports/dashboard/team_colors.json` | 27 clubs | club colours and chart tokens (see below) |
| `reports/dashboard/example_match_5795435.json` | one match | everything one match view needs, both teams |
| `reports/dashboard/example_radar.json` | one player | one radar in both scales, plus impact |

Coordinates: metres on a 105 × 68 pitch from the team's own point of view (own goal at x = 0, attacking toward x = 105, y = 68 on the team's left). On a vertical pitch, draw x upward and put y = 68 on the left.

## Colours

- **One colour per club**, from `team_colors.json → clubs[team].chart`. It is the club's kit or crest colour, darkened where needed to reach 3:1 contrast on the light surface `#fcfcfb`. White-shirted clubs (Tottenham, Fulham, Leeds) use their crest colour.
- **Text on a filled club mark** (node numbers, active filter buttons): `clubs[team].ink`.
- **Two clubs on one screen**: keep the home club's colour; if the away club's colour is too close (OKLab distance < 0.15), use its `chart_alt`. `viz_colors.pair_colors(home, away)` does this; `example_match_*.json → teams[team].colour` is already resolved.
- **Neutral marks** (blocked and off-target shots, pitch lines, labels): the `tokens` block. Never use a club colour for text.
- **Kit colours are approximations** of the clubs' brand colours, not official values.

## Shot map

- **Every shot is a circle** sized by xG (area = 70 + 1400 × xG, in pt²).
- **Goals are filled** in the club colour.
- **Other outcomes are hollow rings:**
  - saved: club colour
  - blocked: `ink_2`
  - off target: `ink_3`
- **Situation filter** (the buttons): Open play, Fast break, Set piece, Penalty. Show only the buttons for situations the team actually has in the match. Several can be active at once. The header tiles (goals, xG, shots, on target, blocked, xG per shot) recompute for the active filter; `shot_summary_by_situation` in the example payload has them precomputed.
- **Blocked shots don't count as on target.** The source flags them as saved and on target.
- **Labels:** only goals get one (scorer, minute, xG).

## Passing network

- **Starting XI only, full match.** Positions and pass counts cover each player's whole time on the pitch; the source has no timestamps, so the network can't be cut at the first substitution. List the substituted starters in a footnote (`summary.substituted`).
- **Links** need at least 4 passes between the pair (`summary.min_link`). Width is 0.6 + 0.42 × count px at 200 dpi. Colour runs along a ramp from a light tint of the club colour to the full colour (`clubs[team].ramp`).
- **Nodes:** filled club colour with the shirt number in `ink`; area = 120 + 9 × passes completed.

## Team shape (convex hull)

- **The hull** is drawn around the 10 outfield starters' average positions, filled with the club colour at 12% opacity under a 1.6 px club-colour outline.
- **Measurements:** width, depth and centre-back height (mean depth of the players in LCB / CB / RCB slots, or the two deepest outfield players when slots are missing).
- **Area percentile** is measured against all 2,380 team-matches in the data.

## Player radar

- **Twelve axes in three sections:**
  - shooting & creating
  - passing (shaded sector)
  - defending
- **Peer group:** players with at least 900 minutes in the season, grouped by their most frequent starting slot (centre-backs, full-backs, central midfielders, attacking midfielders, wingers, strikers).
- **Adjustments:** non-penalty xG. Tackles + interceptions and recoveries are adjusted to 50% opponent possession.
- **Two scales (a toggle):**
  - **Percentile** (default): 0–100, with 50 as the dashed ring. Easiest to read.
  - **Standardised (z-score)**: standard deviations from the group mean, drawn from −2.5 at the centre to +2.5 at the edge, with 0 as the dashed ring. It shows how far ahead a player is, not just his rank. Values beyond ±2.5 sit on the edge and are labelled with their true value.
- **Model impact** (optional row under the radar): the L1 player effects from Step 3.
  - **What they measure:** attack is the change in his team's npxG per 90; defence is the opponent npxG per 90 he prevents; net is the sum. All three are relative to the average player in his position group, with percentiles.
  - **Window:** the 365 days before the ratings date, which differs from the radar's season.
  - **Placement:** beside the radar, not on it. They measure on-pitch impact, while the radar describes what the player does.
- **Raw per-90 values** go under the radar so every percentile can be checked.
