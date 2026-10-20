"""Step 3: cut every match into stints (methodology §3).

A stint is a maximal stretch of a match in which the players on the pitch for both sides (and so
the man count) and the score are unchanged. Boundaries: substitutions, red cards, goals, kick-off,
the end of the first half (only when someone comes on or off there) and the final whistle.

Match clock (minutes of play, stoppage included):
    first half   t = minute + added                 (0 .. T1, T1 = 45 + S1)
    second half  t = T1 + (minute - 45) + added     (T1 .. T2, T2 = T1 + 45 + S2)
S1, S2 = the longest stoppage seen in that half of that match (shots, substitutions), and at least
DEFAULT_STOPPAGE. Substitutions at minute 45 with no added time are half-time substitutions (t = T1).
Red cards carry no stoppage minute in the data; one at minute 45 is placed at T1, at 90 at T2.

Each player who appeared is on the pitch from his entry time to his exit time, taken from
player_match (on_minute, off_minute; a red-carded player's exit is his card), with stoppage minutes
attached from match_events where known; where the network records a substitution minute
(2023/24-2025/26) it is used, which also catches substitutes later taken off in stoppage time. Stints use per-player times, not paired
substitutions (findings log: sub-on and sub-off counts differ in some team-matches).

Each shot is credited to the stint that contains it; a goal ends its stint, so the scoring shot
belongs to the stint before the goal. npxG = xG of non-penalty, non-own-goal shots, split into open
play and set pieces (§3 situation mapping); penalty xG kept separately.

Run:  python src/stints.py
Outputs: data/processed/stints.parquet       one row per directed stint (attacking team -> defending)
         data/processed/stint_players.parquet on-pitch players per stint and team
         data/processed/shot_stints.parquet   stint and clock time of every shot
         data/processed/player_times.parquet  entry and exit time of every player who appeared
         reports/stint_checks.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
DEFAULT_STOPPAGE = (3.0, 6.0)   # minimum assumed first- and second-half stoppage (minutes)
MIN_STINT = 1e-6


def _clock(minute, added, half, T1):
    """Minutes since kick-off on the stoppage-inclusive clock."""
    return np.where(half == 1, minute + added, T1 + (minute - 45) + added)


def build(matches: pd.DataFrame, pm: pd.DataFrame, shots: pd.DataFrame, events: pd.DataFrame, nodes: pd.DataFrame):
    p = pm[pm.played].copy()
    ev = events.copy()
    ev["minute_added"] = ev.minute_added.fillna(0)
    # stoppage per match and half
    sh = shots.assign(half=np.where(shots.period.eq("FirstHalf"), 1, 2))
    s_sh = sh.groupby(["match_id", "half"]).minute_added.max()
    evh = ev[ev.event.ne("red_card")]
    evh = evh.assign(half=np.where((evh.minute < 45) | ((evh.minute == 45) & (evh.minute_added > 0)), 1, 2))
    evh = evh[((evh.half == 1) & (evh.minute == 45)) | ((evh.half == 2) & (evh.minute >= 90))]
    s_ev = evh.groupby(["match_id", "half"]).minute_added.max()
    nd = pd.concat([nodes[["match_id", "sub_on_base", "sub_on_added"]].set_axis(["match_id", "b", "a"], axis=1),
                    nodes[["match_id", "sub_off_base", "sub_off_added"]].set_axis(["match_id", "b", "a"], axis=1)]).dropna()
    nd = nd[nd.b.isin([45, 90])].assign(half=lambda d: np.where(d.b == 45, 1, 2))
    s_nd = nd.groupby(["match_id", "half"]).a.max()
    stop = pd.concat([s_sh, s_ev, s_nd]).groupby(level=[0, 1]).max().unstack()
    stop = stop.reindex(matches.match_id).fillna(0)
    S1 = np.maximum(stop.get(1, 0), DEFAULT_STOPPAGE[0])
    S2 = np.maximum(stop.get(2, 0), DEFAULT_STOPPAGE[1])
    clock = pd.DataFrame({"T1": 45 + S1.to_numpy(), "S2": S2.to_numpy()}, index=matches.match_id)
    clock["T2"] = clock.T1 + 45 + clock.S2

    # ---- player entry/exit times
    add_on = ev[ev.event == "sub_on"].set_index(["match_id", "player_id"]).minute_added
    add_off = ev[ev.event == "sub_off"].set_index(["match_id", "player_id"]).minute_added
    k = pd.MultiIndex.from_arrays([p.match_id, p.player_id])
    p["on_add"] = add_on.reindex(k).fillna(0).to_numpy()
    p["off_add"] = add_off.reindex(k).fillna(0).to_numpy()
    # The network's own substitution minutes (2023/24-2025/26) are exact and include players who
    # were taken off in stoppage time (player_match caps them at 90 and match_events omits them).
    nn = nodes.set_index(["match_id", "player_id"])[["sub_on_base", "sub_on_added", "sub_off_base", "sub_off_added"]]
    nn = nn[~nn.index.duplicated()].reindex(k)
    has_on = nn.sub_on_base.notna().to_numpy() & ~p.started.to_numpy()
    has_off = nn.sub_off_base.notna().to_numpy() & p.red_cards.eq(0).to_numpy()
    p.loc[has_on, "on_minute"] = nn.sub_on_base.to_numpy()[has_on]
    p.loc[has_on, "on_add"] = nn.sub_on_added.fillna(0).to_numpy()[has_on]
    p.loc[has_off, "off_minute"] = nn.sub_off_base.to_numpy()[has_off]
    p.loc[has_off, "off_add"] = nn.sub_off_added.fillna(0).to_numpy()[has_off]
    p["network_times"] = has_on | has_off
    p = p.join(clock, on="match_id")
    on_m, off_m = p.on_minute.to_numpy(float), p.off_minute.to_numpy(float)
    red = p.red_cards.gt(0).to_numpy()
    on_half = np.where((on_m < 45) | ((on_m == 45) & (p.on_add > 0)), 1, 2)
    t_on = np.where(on_m <= 0, 0.0, _clock(on_m, p.on_add, on_half, p.T1))
    # exits
    off_half = np.where((off_m < 45) | ((off_m == 45) & (p.off_add > 0) & ~red), 1, 2)
    t_off = _clock(off_m, np.where(red, 0, p.off_add), off_half, p.T1)
    full = (off_m >= 90) & ~red & (p.off_add == 0) & ~has_off
    t_off = np.where(full, p.T2, t_off)
    t_off = np.where(red & (off_m >= 90), p.T2, t_off)
    t_off = np.minimum(t_off, p.T2)
    t_on = np.minimum(t_on, t_off)
    p["t_on"], p["t_off"] = t_on, t_off

    # ---- goals (with own goals) and shots on the clock
    sh = sh.join(clock, on="match_id")
    sh["t"] = np.clip(_clock(sh.minute.clip(upper=np.where(sh.half == 1, 45, 999)), sh.minute_added, sh.half, sh.T1),
                      0, sh.T2)
    home_id = matches.set_index("match_id").home_team_id

    stint_rows, player_rows, shot_rows = [], [], []
    p_by_match = dict(tuple(p.groupby("match_id")))
    sh_by_match = dict(tuple(sh.groupby("match_id")))
    for m in matches.itertuples():
        mid = m.match_id
        pp = p_by_match[mid]
        ss = sh_by_match.get(mid, sh.iloc[:0])
        T2 = clock.at[mid, "T2"]
        goals = ss[ss.is_goal == 1]
        bounds = np.unique(np.r_[0.0, T2, pp.t_on.to_numpy(), pp.t_off.to_numpy(), goals.t.to_numpy()])
        bounds = bounds[(bounds >= 0) & (bounds <= T2)]
        starts, ends = bounds[:-1], bounds[1:]
        keep = ends - starts > MIN_STINT
        starts, ends = starts[keep], ends[keep]
        if not len(starts):
            continue
        # score at stint start: goals strictly before start (a goal at time t ends the stint ending at t)
        gt = goals.t.to_numpy()
        gteam = goals.scoring_team_id.to_numpy()
        teams = [m.home_team_id, m.away_team_id]
        # shot -> stint: first stint whose end >= t
        si = np.searchsorted(ends, ss.t.to_numpy(), side="left").clip(0, len(ends) - 1)
        shot_rows.append(pd.DataFrame({"shot_id": ss.shot_id.to_numpy(), "match_id": mid, "stint": si, "t": ss.t.to_numpy()}))
        for k_, (a, b) in enumerate(zip(starts, ends)):
            mid_t = (a + b) / 2
            onp = pp[(pp.t_on <= mid_t) & (pp.t_off >= mid_t)]
            n_on = {t: int((onp.team_id == t).sum()) for t in teams}
            score = {t: int(((gt <= a) & (gteam == t)).sum()) for t in teams}
            for t in teams:
                for r in onp[onp.team_id == t].itertuples():
                    player_rows.append((mid, k_, t, r.player_id, bool(r.is_gk)))
            for att, dfn in [(teams[0], teams[1]), (teams[1], teams[0])]:
                inn = ss[(si == k_) & (ss.team_id == att) & (ss.is_own_goal == 0)]
                op = inn.loc[inn.situation_group == "open_play", "xg"].sum()
                sp = inn.loc[inn.situation_group == "set_piece", "xg"].sum()
                pen = inn.loc[inn.situation_group == "penalty", "xg"].sum()
                stint_rows.append(dict(
                    match_id=mid, stint=k_, start=a, end=b, duration=b - a,
                    team_id=att, opponent_id=dfn, home=int(att == home_id[mid]),
                    n_on=n_on[att], n_on_opp=n_on[dfn], gd=score[att] - score[dfn],
                    npxg_op=op, npxg_sp=sp, npxg=op + sp, pen_xg=pen, n_shots=len(inn)))
    st = pd.DataFrame(stint_rows)
    st = st.merge(matches[["match_id", "season", "round", "date"]], on="match_id")
    st["state"] = st.gd.clip(-2, 2)
    st["manadv"] = np.sign(st.n_on - st.n_on_opp).astype(int)
    sp = pd.DataFrame(player_rows, columns=["match_id", "stint", "team_id", "player_id", "is_gk"])
    shot_stint = pd.concat(shot_rows, ignore_index=True)
    return st, sp, p, clock, shot_stint


def checks(st, sp, p, clock, matches, shots, pm):
    c = {}
    side = st.groupby(["match_id", "team_id"]).agg(npxg=("npxg", "sum"), pen=("pen_xg", "sum"), dur=("duration", "sum"))
    s = shots[shots.is_own_goal == 0]
    ref = s[s.situation_group != "penalty"].groupby(["match_id", "team_id"]).xg.sum()
    j = side.join(ref.rename("ref")).fillna(0)
    c["npxg_reconciles_team_matches"] = float(np.isclose(j.npxg, j.ref, atol=1e-6).mean())
    c["shots_assigned"] = int(st.n_shots.sum())
    c["shots_total"] = int(len(shots[shots.is_own_goal == 0]))
    dur = st.groupby("match_id").duration.sum() / 2
    c["duration_equals_clock"] = float(np.isclose(dur, clock.T2.reindex(dur.index)).mean())
    c["match_minutes_mean"] = float(clock.T2.mean())
    c["stints"] = int(len(st) / 2)
    c["directed_stints"] = int(len(st))
    c["stints_per_match"] = dict(st.groupby("match_id").stint.nunique().describe()[["mean", "min", "50%", "max"]].round(2))
    c["stints_per_season"] = (st.groupby("season").size() / 2).astype(int).to_dict()
    # final score reconciles: score at end of last stint + goals in it
    g = shots[shots.is_goal == 1].groupby(["match_id", "scoring_team_id"]).size()
    fs = pd.concat([matches.set_index("match_id").home_score.rename("h"), matches.set_index("match_id").away_score.rename("a")], axis=1)
    c["goals_in_shots_match_final_scores"] = float(
        (g.unstack(fill_value=0).pipe(lambda u: pd.Series(
            [u.loc[m.match_id].get(m.home_team_id, 0) == m.home_score and u.loc[m.match_id].get(m.away_team_id, 0) == m.away_score
             if m.match_id in u.index else (m.home_score == 0 and m.away_score == 0)
             for m in matches.itertuples()]))).mean())
    # players on the pitch
    c["players_on_pitch_distribution"] = st.n_on.value_counts().sort_index().astype(int).to_dict()
    c["share_of_minutes_with_11"] = float(st.loc[st.n_on == 11, "duration"].sum() / st.duration.sum())
    c["gk_on_pitch_share_of_minutes"] = float(
        sp[sp.is_gk].groupby(["match_id", "stint", "team_id"]).size().reindex(
            pd.MultiIndex.from_frame(st[["match_id", "stint", "team_id"]])).notna().mul(st.duration.values).sum() / st.duration.sum())
    c["state_share_of_minutes"] = (st.groupby("state").duration.sum() / st.duration.sum()).round(4).to_dict()
    c["manadv_share_of_minutes"] = (st.groupby("manadv").duration.sum() / st.duration.sum()).round(4).to_dict()
    c["stoppage_minutes_mean"] = dict(first=float((clock.T1 - 45).mean()), second=float(clock.S2.mean()))
    played = pm[pm.played]
    c["player_minutes_vs_minutes_played_corr"] = float(np.corrcoef((p.t_off - p.t_on), played.set_index(["match_id", "player_id"]).minutes_played.reindex(
        pd.MultiIndex.from_frame(p[["match_id", "player_id"]])))[0, 1])
    c["players_with_zero_time"] = int(((p.t_off - p.t_on) <= 0).sum())
    return c


def main():
    matches = pd.read_parquet(PROC / "matches.parquet").sort_values("date")
    pm = pd.read_parquet(PROC / "player_match.parquet")
    shots = pd.read_parquet(PROC / "shots.parquet")
    events = pd.read_parquet(PROC / "match_events.parquet")
    nodes = pd.read_parquet(PROC / "network_nodes.parquet")
    st, sp, p, clock, shot_stint = build(matches, pm, shots, events, nodes)
    shot_stint.to_parquet(PROC / "shot_stints.parquet", index=False)
    st.to_parquet(PROC / "stints.parquet", index=False)
    sp.to_parquet(PROC / "stint_players.parquet", index=False)
    p[["match_id", "team_id", "player_id", "is_gk", "started", "t_on", "t_off"]].to_parquet(
        PROC / "player_times.parquet", index=False)
    c = checks(st, sp, p, clock, matches, shots, pm)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "stint_checks.json").write_text(json.dumps(c, indent=2, default=str))
    print(json.dumps(c, indent=2, default=str))


if __name__ == "__main__":
    main()
