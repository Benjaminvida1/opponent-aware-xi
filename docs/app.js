/* Beyond the Best XI — dashboard (no build step, no dependencies). Data: docs/data/*.json (src/build_site.py). */
"use strict";

const $app = document.getElementById("app");
const $tip = document.getElementById("tip");
const cache = {};
const TOK = { ink: "#0b0b0b", ink2: "#52514e", ink3: "#8a8983", line: "#e4e3de", pitch: "#cfcec8", surface: "#fcfcfb", draw: "#b9b8b1" };
let META = null;

/* ------------------------------------------------------------------ utilities */
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (v, d = 0) => (v == null ? "–" : (v * 100).toFixed(d) + "%");
const fx = (v, d = 2) => (v == null ? "–" : Number(v).toFixed(d));
const sgn = (v, d = 3) => (v == null ? "–" : (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d));
const seasonLabel = (s) => s.replace("/20", "/");
const ord = (n) => { const v = Math.round(n), s = ["th", "st", "nd", "rd"], m = v % 100; return v + (s[(m - 20) % 10] || s[m] || s[0]); };
const fmtDay = (d) => new Date(d + "T12:00:00Z").toLocaleDateString("en-GB", { weekday: "short", day: "numeric", month: "short" });
const fmtDate = (d) => new Date(d + "T12:00:00Z").toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
const club = (t) => META.clubs[t] || { chart: "#2a78d6", ink: "#fff", chart_alt: "#eb6834", ink_alt: "#fff", tint: "#e6eefa" };
const surname = (n) => { const p = String(n).split(" "); return p.length > 1 ? p.slice(1).join(" ") : n; };

async function load(path) {
  if (!cache[path]) cache[path] = fetch("data/" + path).then((r) => { if (!r.ok) throw new Error(path); return r.json(); });
  return cache[path];
}

function hexMix(hex, w, base = "#ffffff") { // w=1 -> hex
  const a = hex.match(/\w\w/g).map((x) => parseInt(x, 16)), b = base.match(/\w\w/g).map((x) => parseInt(x, 16));
  return "#" + a.map((v, i) => Math.round(w * v + (1 - w) * b[i]).toString(16).padStart(2, "0")).join("");
}

/* tooltip: any element with data-tip inside a container wired with wireTips() */
function wireTips(root) {
  root.addEventListener("mousemove", (e) => {
    const t = e.target.closest("[data-tip]");
    if (!t) { $tip.hidden = true; return; }
    $tip.innerHTML = t.getAttribute("data-tip");
    $tip.hidden = false;
    const w = $tip.offsetWidth, h = $tip.offsetHeight;
    let x = e.clientX + 14, y = e.clientY + 14;
    if (x + w > innerWidth - 8) x = e.clientX - w - 14;
    if (y + h > innerHeight - 8) y = e.clientY - h - 14;
    $tip.style.left = x + "px"; $tip.style.top = y + "px";
  });
  root.addEventListener("mouseleave", () => { $tip.hidden = true; });
}

/* ------------------------------------------------------------------ pitch (vertical, attacking up) */
// data coords: x_m 0..105 toward the opponent goal, y_m 0..68 with 68 on the team's left.
const PX = (y) => 68 - y, PY = (x) => 105 - x;

function pitchLines() {
  const s = `stroke="${TOK.pitch}" stroke-width="0.35" fill="none"`;
  const arcT = "M 26.69 16.5 A 9.15 9.15 0 0 0 41.31 16.5", arcB = "M 26.69 88.5 A 9.15 9.15 0 0 1 41.31 88.5";
  return `<g ${s}>
    <rect x="0" y="0" width="68" height="105"/>
    <line x1="0" y1="52.5" x2="68" y2="52.5"/><circle cx="34" cy="52.5" r="9.15"/>
    <rect x="13.84" y="0" width="40.32" height="16.5"/><rect x="24.84" y="0" width="18.32" height="5.5"/>
    <rect x="30.34" y="-2" width="7.32" height="2"/><path d="${arcT}"/>
    <rect x="13.84" y="88.5" width="40.32" height="16.5"/><rect x="24.84" y="99.5" width="18.32" height="5.5"/>
    <rect x="30.34" y="105" width="7.32" height="2"/><path d="${arcB}"/>
  </g>
  <g fill="${TOK.pitch}"><circle cx="34" cy="52.5" r="0.4"/><circle cx="34" cy="11" r="0.4"/><circle cx="34" cy="94" r="0.4"/></g>`;
}

/* ------------------------------------------------------------------ shot map */
const OUT_STYLE = (i, col) => [
  { fill: col, stroke: "#fff", sw: 0.35 },          // goal
  { fill: "none", stroke: col, sw: 0.5 },            // saved
  { fill: "none", stroke: TOK.ink2, sw: 0.38 },     // blocked
  { fill: "none", stroke: TOK.ink3, sw: 0.3 },      // off target
][i];
const shotR = (xg) => Math.sqrt((70 + 1400 * xg) / Math.PI) / 7.6;

function shotSummary(shots) {
  const goals = shots.filter((s) => s[3] === 0).length, xg = shots.reduce((a, s) => a + s[2], 0);
  return { n: shots.length, goals, xg, on: shots.filter((s) => s[3] <= 1).length, blocked: shots.filter((s) => s[3] === 2).length };
}

function shotMapSVG(shots, col) {
  const order = [3, 2, 1, 0];
  const marks = order.flatMap((o) => shots.filter((s) => s[3] === o).sort((a, b) => b[2] - a[2]).map((s) => {
    const st = OUT_STYLE(o, col);
    const tip = `<b>${esc(s[5])}</b> ${esc(s[6])}<br>${META.outcomes[o]} · ${META.situations[s[4]]}<br>xG ${s[2].toFixed(2)}`;
    return `<circle cx="${PX(s[1]).toFixed(2)}" cy="${PY(s[0]).toFixed(2)}" r="${shotR(s[2]).toFixed(2)}" fill="${st.fill}" stroke="${st.stroke}" stroke-width="${st.sw}" data-tip="${esc(tip)}" style="cursor:default"/>`;
  }));
  const labels = shots.filter((s) => s[3] === 0).map((s) => {
    const left = s[1] >= 34, r = shotR(s[2]) + 0.8, x = PX(s[1]) + (left ? -r : r);
    return `<text x="${x.toFixed(2)}" y="${(PY(s[0]) + 0.9).toFixed(2)}" font-size="2.3" font-weight="600" text-anchor="${left ? "end" : "start"}" fill="${TOK.ink}" stroke="${TOK.surface}" stroke-width="0.7" paint-order="stroke">${esc(s[5])} ${esc(s[6])}</text>`;
  });
  return `<svg viewBox="-2 -3.5 72 41" role="img" aria-label="Shot map">${pitchLines()}${marks.join("")}${labels.join("")}</svg>`;
}

/* ------------------------------------------------------------------ passing network */
const nodeR = (p) => Math.sqrt((120 + 9 * p) / Math.PI) / 5.6;

function networkSVG(team, col, ink, minLink) {
  const N = team.nodes, links = team.links.filter((l) => l[2] >= minLink);
  const vmax = Math.max(minLink + 1, ...team.links.map((l) => l[2]));
  const ls = links.sort((a, b) => a[2] - b[2]).map((l) => {
    const a = N[l[0]], b = N[l[1]], t = (l[2] - minLink) / (vmax - minLink);
    return `<line x1="${PX(a[4])}" y1="${PY(a[3])}" x2="${PX(b[4])}" y2="${PY(b[3])}" stroke="${hexMix(col, 0.25 + 0.75 * t)}" stroke-width="${((0.5 + 0.32 * l[2]) / 5.2).toFixed(2)}" stroke-linecap="round" data-tip="${esc(`<b>${a[1]} ↔ ${b[1]}</b><br>${l[2]} passes`)}"/>`;
  });
  const ns = N.map((n) => {
    const r = nodeR(n[5]);
    const tip = `<b>${esc(n[1])}</b> #${n[2] ?? ""}<br>${n[5]} passes completed of ${n[5] + n[6]}<br>${n[7] < 89 ? "Off after " + n[7] + "'" : "Played the full match"}`;
    return `<g data-tip="${esc(tip)}"><circle cx="${PX(n[4])}" cy="${PY(n[3])}" r="${r.toFixed(2)}" fill="${col}" stroke="#fff" stroke-width="0.45"/>
      <text x="${PX(n[4])}" y="${(PY(n[3]) + 0.8).toFixed(2)}" font-size="2.2" font-weight="700" text-anchor="middle" fill="${ink}">${n[2] ?? ""}</text>
      <text x="${PX(n[4])}" y="${(PY(n[3]) + r + 2.6).toFixed(2)}" font-size="2.1" text-anchor="middle" fill="${TOK.ink}" stroke="${TOK.surface}" stroke-width="0.6" paint-order="stroke">${esc(n[1])}</text></g>`;
  });
  return `<svg viewBox="-2 -3.5 72 112" role="img" aria-label="Passing network">${pitchLines()}${ls.join("")}${ns.join("")}</svg>`;
}

function networkStats(team) {
  const N = team.nodes, done = N.reduce((a, n) => a + n[5], 0), att = N.reduce((a, n) => a + n[5] + n[6], 0);
  const hub = [...N].sort((a, b) => b[5] - a[5])[0];
  const top = [...team.links].sort((a, b) => b[2] - a[2])[0];
  return { done, comp: att ? done / att : null, hub, top: top ? [N[top[0]][1], N[top[1]][1], top[2]] : null };
}

/* ------------------------------------------------------------------ team shape */
function shapeSVG(team, col, ink) {
  const N = team.nodes, h = team.shape.hull.map((i) => `${PX(N[i][4])},${PY(N[i][3])}`).join(" ");
  const cb = team.shape.cb;
  const ns = N.map((n) => `<g><circle cx="${PX(n[4])}" cy="${PY(n[3])}" r="1.9" fill="${col}" stroke="#fff" stroke-width="0.4"/>
    <text x="${PX(n[4])}" y="${(PY(n[3]) + 0.75).toFixed(2)}" font-size="2" font-weight="700" text-anchor="middle" fill="${ink}">${n[2] ?? ""}</text>
    <text x="${PX(n[4])}" y="${(PY(n[3]) + 4.3).toFixed(2)}" font-size="2" text-anchor="middle" fill="${TOK.ink}" stroke="${TOK.surface}" stroke-width="0.6" paint-order="stroke">${esc(n[1])}</text></g>`);
  return `<svg viewBox="-2 -3.5 72 112" role="img" aria-label="Team shape">${pitchLines()}
    <polygon points="${h}" fill="${col}" fill-opacity="0.12" stroke="${col}" stroke-width="0.45" stroke-linejoin="round"/>
    <line x1="0" x2="68" y1="${PY(cb)}" y2="${PY(cb)}" stroke="${TOK.ink3}" stroke-width="0.3" stroke-dasharray="1.2 0.9"/>
    <text x="67.4" y="${PY(cb) + 2.6}" font-size="2" text-anchor="end" fill="${TOK.ink2}" stroke="${TOK.surface}" stroke-width="0.6" paint-order="stroke">Centre-backs ${Math.round(cb)} m from own goal</text>
    ${ns.join("")}</svg>`;
}

/* ------------------------------------------------------------------ radar */
function radarSVG(entries, mode) {
  // entries: [{p: [...], z: [...], col}]
  const M = META.radar.metrics, n = M.length, ang = M.map((_, i) => Math.PI / 2 - (i * 2 * Math.PI) / n), step = (2 * Math.PI) / n;
  const rad = (e, i) => mode === "pct" ? e.p[i] : (e.z[i] == null ? null : 50 + Math.max(-2.5, Math.min(2.5, e.z[i])) / 2.5 * 50);
  const xy = (r, a) => [r * Math.cos(a), -r * Math.sin(a)];
  let g = "";
  M.forEach((m, i) => {
    if (m.section !== "Passing") return;
    const a0 = ang[i] + step / 2, a1 = ang[i] - step / 2, [x0, y0] = xy(100, a0), [x1, y1] = xy(100, a1);
    g += `<path d="M0 0 L${x0} ${y0} A100 100 0 0 1 ${x1} ${y1} Z" fill="#f0efec"/>`;
  });
  const rings = mode === "pct" ? [25, 50, 75, 100] : [10, 30, 50, 70, 90, 100];
  rings.forEach((r) => (g += `<circle r="${r}" fill="none" stroke="#d6d5cf" stroke-width="${r === 100 ? 1.4 : 0.8}" ${r === 50 ? 'stroke-dasharray="3 3"' : ""}/>`));
  ang.forEach((a) => { const [x, y] = xy(100, a); g += `<line x2="${x}" y2="${y}" stroke="#d6d5cf" stroke-width="0.8"/>`; });
  g += `<text x="3" y="-52" font-size="7" fill="${TOK.ink3}">${mode === "pct" ? "50th" : "average"}</text>`;
  if (mode === "z") g += `<text x="3" y="-72" font-size="7" fill="${TOK.ink3}">+1</text><text x="3" y="-92" font-size="7" fill="${TOK.ink3}">+2</text>`;
  entries.forEach((e, k) => {
    const pts = M.map((_, i) => { const r = rad(e, i); return r == null ? null : xy(r, ang[i]); });
    const ok = pts.filter(Boolean);
    g += `<polygon points="${ok.map((p) => p.join(",")).join(" ")}" fill="${e.col}" fill-opacity="${entries.length > 1 ? 0.14 : 0.2}" stroke="${e.col}" stroke-width="2" stroke-linejoin="round"/>`;
    pts.forEach((p, i) => {
      if (!p) return;
      const v = mode === "pct" ? `${Math.round(e.p[i])}th percentile` : `${e.z[i] >= 0 ? "+" : ""}${e.z[i].toFixed(1)} SD from average`;
      g += `<circle cx="${p[0]}" cy="${p[1]}" r="3.4" fill="${e.col}" stroke="#fff" stroke-width="1.4" data-tip="${esc(`<b>${e.name}</b><br>${M[i].label}: ${v}`)}"/>`;
    });
  });
  M.forEach((m, i) => {
    const a = ang[i], c = Math.cos(a), s = Math.sin(a), [lx, ly] = xy(121, a);
    const anchor = Math.abs(c) < 0.2 ? "middle" : c > 0 ? "start" : "end";
    const dy = Math.abs(s) < 0.2 ? 3 : s > 0 ? -2 : 8;
    const words = m.label.split(" "), cut = m.label.length > 13 && words.length > 1 ? Math.ceil(words.length / 2) : words.length;
    const lines = [words.slice(0, cut).join(" "), words.slice(cut).join(" ")].filter(Boolean);
    const y0 = ly + dy - (lines.length - 1) * (s > 0.2 ? 9.5 : s < -0.2 ? 0 : 4.75);
    g += `<text x="${lx}" y="${y0}" font-size="8.6" text-anchor="${anchor}" fill="${TOK.ink}">${lines.map((t, j) => `<tspan x="${lx}" dy="${j ? 9.5 : 0}">${esc(t)}</tspan>`).join("")}</text>`;
    if (entries.length === 1) {
      const e = entries[0], val = mode === "pct" ? e.p[i] : e.z[i];
      const [vx, vy] = xy(108, a);
      const t = val == null ? "n/a" : mode === "pct" ? Math.round(val) : (val >= 0 ? "+" : "") + val.toFixed(1);
      g += `<text x="${vx}" y="${vy + 3}" font-size="8.4" font-weight="700" text-anchor="middle" fill="${e.col}" stroke="${TOK.surface}" stroke-width="3" paint-order="stroke">${t}</text>`;
    }
  });
  return `<svg class="radar-svg" viewBox="-200 -152 400 304" role="img" aria-label="Player radar">${g}</svg>`;
}

/* ------------------------------------------------------------------ routing */
const routes = { overview: viewOverview, matches: viewMatches, match: viewMatch, players: viewPlayers, teams: viewTeams, method: viewMethod };

function parseHash() {
  const h = location.hash.replace(/^#\/?/, "");
  const [path, q] = h.split("?");
  const parts = path.split("/").filter(Boolean).map(decodeURIComponent);
  return { name: parts[0] || "overview", arg: parts[1], q: new URLSearchParams(q || "") };
}

async function render() {
  const r = parseHash();
  document.querySelectorAll(".nav a").forEach((a) => a.classList.toggle("on", a.dataset.route === (r.name === "match" ? "matches" : r.name)));
  $tip.hidden = true;
  try {
    META = META || (await load("meta.json"));
    await (routes[r.name] || viewOverview)(r);
  } catch (err) {
    console.error(err);
    $app.innerHTML = `<p class="muted">Something went wrong loading this page. <a href="#/">Back to the overview</a></p>`;
  }
}
addEventListener("hashchange", () => { render(); window.scrollTo(0, 0); });
render();

/* ------------------------------------------------------------------ Overview */
async function viewOverview() {
  const f = META.facts, L = META.ladder, hit = META.forecast_hit_rate;
  const l0 = L.find((x) => x.key === "L0"), l1 = L.find((x) => x.key === "L1"), l3 = L.find((x) => x.key === "L3");
  const rung = (x) => `<div class="rung ${x.key === "L1" ? "hi" : ""} ${x.key.startsWith("Market") ? "mkt" : ""}">
      <div class="who"><div class="nm">${esc(x.name)}</div><div class="bl">${esc(x.blurb)}</div></div>
      <div class="bar" aria-hidden="true"><i style="width:${Math.max(0.5, x.share_of_market_gain * 100)}%"></i></div>
      <div class="pc">${Math.round(x.share_of_market_gain * 100)}%</div></div>`;
  $app.innerHTML = `
  <section class="hero">
    <div class="kicker">Premier League · ${seasonLabel(f.seasons[0])} to ${seasonLabel(f.seasons.at(-1))}</div>
    <h1 class="display">Does the opponent change the best XI?</h1>
    <p class="lede">I built a model that picks a starting lineup for every Premier League match, then tested whether knowing the opponent makes those picks better. The short answer: <strong>who plays matters a lot. Who you play against barely changes the best team to send out.</strong></p>
    <div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:20px">
      <a class="cta" href="#/matches">Explore the matches</a><a class="cta ghost" href="#/players">Compare players</a><a class="cta ghost" href="#/method">How it works</a>
    </div>
  </section>
  <section class="section grid g4">
    ${[[f.matches.toLocaleString(), "matches"], [f.shots.toLocaleString(), "shots, each with its xG"], [f.players.toLocaleString(), "players"], [f.seasons.length, "seasons, 2023/24 to the start of 2026/27"]]
      .map(([n, l]) => `<div class="card tile"><div class="num">${n}</div><div class="lab">${l}</div></div>`).join("")}
  </section>
  <section class="section">
    <div class="section-head"><div><h2 class="display">How good are the predictions?</h2>
      <p>Each step adds one idea to the model. The bar shows how much of the gap between guessing and the bookmakers it closes, on ${f.test_matches} matches the model never saw while it was being built (2025/26 and the start of 2026/27).</p></div></div>
    <div class="card"><div class="ladder">${L.map(rung).join("")}</div>
      <p class="note">Bookmakers' closing odds know the team news, injuries and everything else the market knows; they are the benchmark, not a target. Score: ranked probability score (lower is better), from ${L[0].rps} for guessing to ${L.at(-1).rps} for the bookmakers.</p></div>
  </section>
  <section class="section">
    <div class="section-head"><div><h2 class="display">What I found</h2></div></div>
    <div class="grid g3">
      <div class="card find"><div class="kicker">1 · Lineups matter</div><div class="big">${Math.round(l0.share_of_market_gain * 100)}% → ${Math.round(l1.share_of_market_gain * 100)}%</div>
        <h3>Knowing who plays helps</h3><p>Rating each player and adding up the eleven on the pitch closes much more of the gap to the bookmakers than team strength alone.</p></div>
      <div class="card find"><div class="kicker">2 · The opponent doesn't</div><div class="big">${Math.round(l1.share_of_market_gain * 100)}% → ${Math.round(l3.share_of_market_gain * 100)}%</div>
        <h3>Styles and matchups add nothing</h3><p>How the two teams' styles fit together, and who faces whom on the pitch, make no measurable difference to the predictions (the score barely moves, well inside the margin of error). That's the main finding.</p></div>
      <div class="card find"><div class="kicker">3 · The lineup picker</div><div class="big">${pct(f.opponent_changes_best, 1)}</div>
        <h3>of picks change with the opponent</h3><p>The model's best XI agrees with the manager's on ${f.recommended_overlap} of 11 players on average. Facing a different opponent changes its pick in only ${pct(f.opponent_changes_best, 1)} of selections.</p></div>
    </div>
    <div class="card" style="margin-top:16px;display:flex;gap:24px;flex-wrap:wrap;align-items:center">
      <div class="tile"><div class="num">${pct(hit.model, 1)}</div><div class="lab">Matches where the model's favourite won</div></div>
      <div class="tile"><div class="num">${pct(hit.market, 1)}</div><div class="lab">Matches where the bookmakers' favourite won</div></div>
      <p class="muted" style="flex:1 1 280px;margin:0">Picking the most likely result is a blunt test (draws are rarely anyone's favourite), but it shows the model is in the same league as the market. Open any match from 2025/26 or 2026/27 to see both forecasts.</p>
    </div>
  </section>`;
}

/* ------------------------------------------------------------------ Matches list */
async function viewMatches(r) {
  const all = await load("matches.json");
  const seasons = [...new Set(all.map((m) => m.season))].sort().reverse();
  const teams = [...new Set(all.flatMap((m) => [m.home, m.away]))].sort();
  const st = { season: r.q.get("season") || seasons[0], team: r.q.get("team") || "", model: r.q.get("model") === "1" };
  $app.innerHTML = `
    <div class="section-head" style="margin-top:0"><div><div class="kicker">Match explorer</div><h2 class="display">Every match, every shot</h2>
      <p>Pick a match to see both teams' shots, passing networks and shapes. Matches from 2025/26 onward also show the model's forecast and its lineup call for each side.</p></div></div>
    <div class="controls">
      <select class="select" id="fSeason" aria-label="Season">${seasons.map((s) => `<option value="${s}" ${s === st.season ? "selected" : ""}>${seasonLabel(s)}</option>`).join("")}</select>
      <select class="select" id="fTeam" aria-label="Team"><option value="">All teams</option>${teams.map((t) => `<option ${t === st.team ? "selected" : ""}>${esc(t)}</option>`).join("")}</select>
      <label class="inline"><input type="checkbox" id="fModel" ${st.model ? "checked" : ""}> Only matches with a model forecast</label>
    </div>
    <p class="small" style="margin:-6px 0 0">The small bar on the right is the model's pre-match chance of a home win, a draw (grey) and an away win.</p>
    <div id="mlist"></div>`;
  const draw = () => {
    const rows = all.filter((m) => m.season === st.season && (!st.team || m.home === st.team || m.away === st.team) && (!st.model || m.model));
    const byRound = {};
    rows.forEach((m) => (byRound[m.round] = byRound[m.round] || []).push(m));
    const rounds = Object.keys(byRound).map(Number).sort((a, b) => b - a);
    document.getElementById("mlist").innerHTML = rows.length ? rounds.map((rd) => `<div class="round"><h3>Round ${rd}</h3><div class="mlist">${byRound[rd].map(matchRow).join("")}</div></div>`).join("")
      : `<p class="muted">No matches for this filter.</p>`;
  };
  const sync = () => { const q = new URLSearchParams({ season: st.season, ...(st.team ? { team: st.team } : {}), ...(st.model ? { model: "1" } : {}) }); history.replaceState(null, "", "#/matches?" + q); draw(); };
  document.getElementById("fSeason").onchange = (e) => { st.season = e.target.value; sync(); };
  document.getElementById("fTeam").onchange = (e) => { st.team = e.target.value; sync(); };
  document.getElementById("fModel").onchange = (e) => { st.model = e.target.checked; sync(); };
  draw();
}

function matchRow(m) {
  const [hc, ac] = m.col;
  const mini = m.model ? `<div class="mini" title="Model: home ${pct(m.model[0])}, draw ${pct(m.model[1])}, away ${pct(m.model[2])}"><i style="flex:${m.model[0]};background:${hc}"></i><i style="flex:${m.model[1]};background:${TOK.draw}"></i><i style="flex:${m.model[2]};background:${ac}"></i></div>` : "";
  return `<a class="mrow" href="#/match/${m.id}"><span class="dt">${fmtDay(m.date)}</span>
    <span class="tm h"><span>${esc(m.home)}</span><span class="dot" style="background:${hc}"></span></span>
    <span class="sc">${m.hs}–${m.as}</span>
    <span class="tm"><span class="dot" style="background:${ac}"></span><span>${esc(m.away)}</span></span>
    <span class="fc">${mini}</span></a>`;
}

/* ------------------------------------------------------------------ Match page */
async function viewMatch(r) {
  const all = await load("matches.json");
  const m = all.find((x) => String(x.id) === String(r.arg));
  if (!m) { $app.innerHTML = `<p class="muted">Match not found. <a href="#/matches">Back to matches</a></p>`; return; }
  const d = await load(`match/${m.id}.json`);
  const [hc, ac, hi, ai] = m.col;
  const T = [{ ...d.teams.home, col: hc, ink: hi }, { ...d.teams.away, col: ac, ink: ai }];
  const res = m.hs > m.as ? 0 : m.hs === m.as ? 1 : 2;
  const resTxt = [`${m.home} win`, "Draw", `${m.away} win`][res];
  const hasCall = T.some((t) => t.call);
  $app.innerHTML = `
    <a class="back" href="#/matches?season=${encodeURIComponent(m.season)}">← All ${seasonLabel(m.season)} matches</a>
    <div class="board">
      <div class="side"><span class="stripe" style="background:${hc}"></span><a class="club" href="#/teams/${encodeURIComponent(m.home)}" style="text-decoration:none">${esc(m.home)}</a></div>
      <div class="score">${m.hs}–${m.as}</div>
      <div class="side a"><span class="stripe" style="background:${ac}"></span><a class="club" href="#/teams/${encodeURIComponent(m.away)}" style="text-decoration:none">${esc(m.away)}</a></div>
      <div class="meta">Premier League ${seasonLabel(m.season)} · Round ${m.round} · ${fmtDate(m.date)}</div>
    </div>
    <nav class="subnav" aria-label="On this page">${m.model ? '<a href="#fc" data-jump>Forecast</a>' : ""}<a href="#shots" data-jump>Shots</a><a href="#net" data-jump>Passing</a><a href="#shape" data-jump>Shape</a>${hasCall ? '<a href="#calls" data-jump>Lineup calls</a>' : ""}</nav>
    ${m.model ? `<section class="section" id="fc" style="margin-top:20px"><div class="section-head"><div><h2 class="display">Before kick-off</h2><p>Win, draw and loss chances from the model and from the bookmakers' closing odds. Result: <strong>${esc(resTxt)}</strong>.</p></div></div>
      <div class="card fcast">${fbar("Model", m.model, hc, ac, res)}${m.market ? fbar("Bookmakers", m.market, hc, ac, res) : ""}
      <div class="legend"><span><i class="sw" style="background:${hc}"></i>${esc(m.home)}</span><span><i class="sw" style="background:${TOK.draw}"></i>Draw</span><span><i class="sw" style="background:${ac}"></i>${esc(m.away)}</span></div></div></section>`
      : `<p class="note" style="margin-top:14px">Model forecasts and lineup calls cover the test matches (2025/26 and 2026/27); earlier seasons were used to build the model.</p>`}
    <section class="section" id="shots"><div class="section-head"><div><h2 class="display">Shots</h2><p>Every shot, sized by its expected goals (xG): the chance a typical shot from there goes in. Filled circles are goals.</p></div>
      <div class="chips" id="sitChips">${META.situations.map((s) => `<button class="chip on" data-sit="${s}">${s}</button>`).join("")}</div></div>
      <div class="grid g2" id="shotGrid"></div>
      <div class="legend" style="margin-top:12px"><span><i class="ring" style="background:${TOK.ink};border-color:${TOK.ink}"></i>Goal</span><span><i class="ring" style="border-color:${TOK.ink}"></i>Saved (club colour)</span><span><i class="ring" style="border-color:${TOK.ink2}"></i>Blocked</span><span><i class="ring" style="border-color:${TOK.ink3};border-width:1.5px"></i>Off target</span><span>Bigger circle = better chance</span></div>
    </section>
    <section class="section" id="net"><div class="section-head"><div><h2 class="display">Passing networks</h2><p>Each starter at his average position. Lines join players who passed to each other; thicker and darker means more passes. Circle size shows passes completed.</p></div>
      <label class="inline">Show links with at least <select class="select" id="minLink">${[2, 3, 4, 5, 6, 8, 10].map((v) => `<option ${v === 4 ? "selected" : ""}>${v}</option>`).join("")}</select> passes</label></div>
      <div class="grid g2" id="netGrid"></div></section>
    <section class="section" id="shape"><div class="section-head"><div><h2 class="display">Team shape</h2><p>The outline around the ten outfield starters' average positions: how wide, how deep and how high up the pitch each team played.</p></div></div>
      <div class="grid g2">${T.map(shapeCard).join("")}</div></section>
    ${hasCall ? `<section class="section" id="calls"><div class="section-head"><div><h2 class="display">The model's lineup call</h2><p>Before the match, the model compares each team's usual XI with the best XI it can pick from the matchday squad. It only recommends a change when it is at least 70% sure the change is better.</p></div></div>
      <div class="grid g2">${T.map((t) => callCard(t)).join("")}</div>
      <p class="note">Expected points are the model's own estimate, so the gains are not independent proof. The model doesn't know about fitness, rotation or tactics, which explains many of the differences from what managers chose.</p></section>` : ""}`;

  wireTips($app);
  document.querySelectorAll("[data-jump]").forEach((a) => a.addEventListener("click", (e) => { e.preventDefault(); document.querySelector(a.getAttribute("href")).scrollIntoView({ behavior: "smooth", block: "start" }); }));
  const active = new Set(META.situations);
  const present = new Set(T.flatMap((t) => t.shots.map((s) => META.situations[s[4]])));
  document.querySelectorAll("#sitChips .chip").forEach((c) => { if (!present.has(c.dataset.sit)) { c.disabled = true; c.classList.remove("on"); } });
  const drawShots = () => {
    document.getElementById("shotGrid").innerHTML = T.map((t) => {
      const sh = t.shots.filter((s) => active.has(META.situations[s[4]])), sm = shotSummary(sh);
      return `<div class="card pitch-card"><h3><span class="dot" style="background:${t.col}"></span>${esc(t.team)}</h3>
        <div class="tiles-row">${[[sm.goals, "Goals"], [sm.xg.toFixed(2), "xG"], [sm.n, "Shots"], [sm.on, "On target"], [sm.blocked, "Blocked"]].map(([n, l]) => `<div class="t"><div class="num">${n}</div><div class="lab">${l}</div></div>`).join("")}</div>
        <div class="pitch pitch-wrap">${shotMapSVG(sh, t.col)}</div></div>`;
    }).join("");
  };
  document.getElementById("sitChips").addEventListener("click", (e) => {
    const c = e.target.closest(".chip"); if (!c || c.disabled) return;
    const s = c.dataset.sit;
    if (active.has(s) && [...active].filter((x) => present.has(x)).length === 1) return; // keep at least one
    active.has(s) ? active.delete(s) : active.add(s);
    c.classList.toggle("on", active.has(s)); drawShots();
  });
  drawShots();
  const drawNet = () => {
    const ml = +document.getElementById("minLink").value;
    document.getElementById("netGrid").innerHTML = T.map((t) => {
      const s = networkStats(t);
      return `<div class="card pitch-card"><h3><span class="dot" style="background:${t.col}"></span>${esc(t.team)}</h3>
        <div class="tiles-row">${[[s.done, "Passes completed"], [pct(s.comp), "Completion"], [`${esc(s.hub[1])} · ${s.hub[5]}`, "Most passes"], s.top ? [`${esc(s.top[0])} ↔ ${esc(s.top[1])} · ${s.top[2]}`, "Busiest pair"] : null].filter(Boolean).map(([n, l]) => `<div class="t"><div class="num">${n}</div><div class="lab">${l}</div></div>`).join("")}</div>
        <div class="pitch pitch-wrap">${networkSVG(t, t.col, t.ink, ml)}</div>
        <p class="note">Substituted: ${t.nodes.filter((n) => n[7] < 89).map((n) => `${esc(n[1])} ${n[7]}'`).join(", ") || "none"}. Positions and passes cover each player's whole time on the pitch.</p></div>`;
    }).join("");
  };
  document.getElementById("minLink").onchange = drawNet;
  drawNet();
}

function fbar(label, p, hc, ac, res) {
  const seg = (v, c, i) => `<div style="flex:${v};background:${c};color:${i === 1 ? TOK.ink : "#fff"};${i === res ? "box-shadow:inset 0 -4px 0 rgba(0,0,0,.35)" : ""}" title="${pct(v)}">${v >= 0.09 ? pct(v) : ""}</div>`;
  return `<div class="fbar"><div class="lab">${label}</div><div class="bars">${seg(p[0], hc, 0)}${seg(p[1], TOK.draw, 1)}${seg(p[2], ac, 2)}</div></div>`;
}

function shapeCard(t) {
  const s = t.shape;
  return `<div class="card pitch-card"><h3><span class="dot" style="background:${t.col}"></span>${esc(t.team)}</h3>
    <div class="tiles-row">${[[`${Math.round(s.area).toLocaleString()} m²`, `Area · bigger than ${Math.round(s.area_pct * 100)}% of team-matches`], [`${Math.round(s.width)} m`, "Width"], [`${Math.round(s.depth)} m`, "Depth"], [`${Math.round(s.centre)} m`, "Team centre from own goal"]].map(([n, l]) => `<div class="t"><div class="num">${n}</div><div class="lab">${l}</div></div>`).join("")}</div>
    <div class="pitch pitch-wrap">${shapeSVG(t, t.col, t.ink)}</div></div>`;
}

function callCard(t) {
  const c = t.call;
  if (!c) return `<div class="card"><h3>${esc(t.team)}</h3><p class="muted">No lineup call for this match.</p></div>`;
  const actualIds = new Set(c.actual.map((p) => p.id));
  const list = (xi) => `<ul class="xi">${xi.map((p) => `<li class="${actualIds.has(p.id) ? "" : "diff"}">${esc(p.name)}${p.slot ? ` <span class="tag">${esc(p.slot)}</span>` : ""}</li>`).join("")}</ul>`;
  const verdict = c.recommended ? `Make ${c.changes.length} change${c.changes.length === 1 ? "" : "s"}` : "Stick with the usual XI";
  const sure = c.p_better == null ? "" : c.recommended
    ? `${Math.round(c.p_better * 100)}% sure the new XI is better · win chance ${pct(c.usual_win)} → ${pct(c.recommended_win)}`
    : `Its best alternative was only ${Math.round(c.p_better * 100)}% likely to be better (it needs 70%)`;
  return `<div class="card call"><h3><span class="dot" style="background:${t.col}"></span>${esc(t.team)} · ${esc(c.formation)}</h3>
    <div class="verdict" style="color:${t.col}">${verdict}</div><p class="muted" style="margin:0 0 8px">${sure}</p>
    ${c.changes.map((s) => `<div class="swap"><span class="out">${esc(s.out)}</span><span aria-label="replaced by">→</span><span class="in">${esc(s.in)}</span><span class="pos">${esc(s.slot)} · ${sgn(s.dEP, 2)} expected points</span></div>`).join("")}
    <p class="small" style="margin-top:10px">The manager's XI matched the usual XI on ${c.usual_overlap_with_actual} of 11 players${c.recommended ? ` and the model's pick on ${c.rec_overlap_with_actual}` : ""}. Points won: ${c.points}.</p>
    <details><summary>Show the three XIs (outfield; bold = not in the manager's XI)</summary>
      <div class="grid g2" style="margin-top:8px"><div><div class="kicker">Usual XI</div>${list(c.usual)}</div><div><div class="kicker">Model's pick</div>${list(c.recommended_xi)}</div></div>
      <div style="margin-top:8px"><div class="kicker">Manager's XI</div><ul class="xi">${c.actual.map((p) => `<li>${esc(p.name)}</li>`).join("")}</ul></div></details></div>`;
}

/* ------------------------------------------------------------------ Players */
async function viewPlayers(r) {
  const P = await load("players.json");
  const seasons = [...new Set(P.players.map((p) => p.season))].sort().reverse();
  const groups = META.radar.groups, M = META.radar.metrics;
  const st = { season: r.q.get("season") || seasons[0], group: r.q.get("group") || "", team: r.q.get("team") || "", q: "",
    sort: r.q.get("sort") || "minutes", id: +r.q.get("id") || null, cmp: +r.q.get("cmp") || null, mode: r.q.get("mode") || "pct" };
  const teams = () => [...new Set(P.players.filter((p) => p.season === st.season).map((p) => p.team))].sort();
  const find = (id) => P.players.find((p) => p.season === st.season && p.id === id);
  if (!st.id || !find(st.id)) { const def = P.players.find((p) => p.season === st.season && p.name === "Bukayo Saka") || P.players.find((p) => p.season === st.season); st.id = def.id; }

  $app.innerHTML = `
    <div class="section-head" style="margin-top:0"><div><div class="kicker">Player explorer</div><h2 class="display">Player profiles</h2>
      <p>Each player is compared with players in the same position over a season (900 minutes or more). Switch between percentiles (where he ranks) and standardised scores (how far from average he is).</p></div></div>
    <div class="controls">
      <select class="select" id="pSeason" aria-label="Season">${seasons.map((s) => `<option value="${s}" ${s === st.season ? "selected" : ""}>${seasonLabel(s)}</option>`).join("")}</select>
      <select class="select" id="pGroup" aria-label="Position"><option value="">All positions</option>${Object.entries(groups).map(([k, v]) => `<option value="${k}" ${k === st.group ? "selected" : ""}>${v[0].toUpperCase() + v.slice(1)}</option>`).join("")}</select>
      <select class="select" id="pTeam" aria-label="Team"></select>
      <select class="select" id="pSort" aria-label="Sort by"><option value="minutes">Sort: minutes</option><option value="name">Sort: name</option>${M.map((m, i) => `<option value="${i}">Sort: ${esc(m.label)}</option>`).join("")}</select>
      <input class="search" id="pQ" type="search" placeholder="Search a player" aria-label="Search a player">
    </div>
    <div class="players-layout"><div class="plist" id="plist" role="listbox" aria-label="Players"></div><div id="pview"></div></div>`;
  const sel = (id) => document.getElementById(id);
  const fillTeams = () => { sel("pTeam").innerHTML = `<option value="">All teams</option>` + teams().map((t) => `<option ${t === st.team ? "selected" : ""}>${esc(t)}</option>`).join(""); };
  const sync = () => { const q = new URLSearchParams({ season: st.season, id: st.id, mode: st.mode, ...(st.group ? { group: st.group } : {}), ...(st.team ? { team: st.team } : {}), ...(st.cmp ? { cmp: st.cmp } : {}), ...(st.sort !== "minutes" ? { sort: st.sort } : {}) }); history.replaceState(null, "", "#/players?" + q); };
  const drawList = () => {
    let L = P.players.filter((p) => p.season === st.season && (!st.group || p.group === st.group) && (!st.team || p.team === st.team) && (!st.q || p.name.toLowerCase().includes(st.q)));
    if (st.sort === "name") L.sort((a, b) => surname(a.name).localeCompare(surname(b.name)));
    else if (st.sort === "minutes") L.sort((a, b) => b.minutes - a.minutes);
    else { const i = +st.sort; L.sort((a, b) => (b.p[i] ?? -1) - (a.p[i] ?? -1)); }
    const val = (p) => st.sort === "minutes" || st.sort === "name" ? `${p.minutes.toLocaleString()}'` : p.p[+st.sort] == null ? "n/a" : ord(p.p[+st.sort]);
    sel("plist").innerHTML = L.length ? L.slice(0, 400).map((p) => `<button role="option" class="${p.id === st.id ? "on" : ""}" data-id="${p.id}"><span class="dot" style="background:${club(p.team).chart}"></span><span><span class="pn">${esc(p.name)}</span><br><span class="pt">${esc(p.team)} · ${esc(groups[p.group])}</span></span><span class="pv">${val(p)}</span></button>`).join("")
      : `<p class="muted" style="padding:12px">No players match.</p>`;
  };
  const drawView = () => {
    const a = find(st.id); if (!a) return;
    const peers = P.players.filter((p) => p.season === st.season && p.group === a.group && p.id !== a.id).sort((x, y) => surname(x.name).localeCompare(surname(y.name)));
    const b = st.cmp ? peers.find((p) => p.id === st.cmp) : null;
    const ca = club(a.team);
    let cb = b ? club(b.team).chart : null;
    if (b && (b.team === a.team || (META.clash[a.team] || []).includes(b.team))) cb = club(b.team).chart_alt;
    const entries = [{ ...a, col: ca.chart }, ...(b ? [{ ...b, col: cb }] : [])];
    const im = P.impact[a.id];
    sel("pview").innerHTML = `<div class="card">
      <div class="radar-head"><div><h3 style="font-size:22px">${esc(a.name)}${b ? ` <span class="muted" style="font-weight:400">vs</span> <span style="color:${cb}">${esc(b.name)}</span>` : ""}</h3>
        <p class="small" style="margin:2px 0 0">${esc(a.team)} · ${a.minutes.toLocaleString()} minutes · compared with ${a.n} ${esc(groups[a.group])} in ${seasonLabel(a.season)}</p></div>
        <div class="seg" role="group" aria-label="Scale"><button data-mode="pct" class="${st.mode === "pct" ? "on" : ""}">Percentile</button><button data-mode="z" class="${st.mode === "z" ? "on" : ""}">Standardised</button></div></div>
      <div class="controls" style="margin:10px 0 0"><label class="inline">Compare with <select class="select" id="pCmp"><option value="">nobody</option>${peers.map((p) => `<option value="${p.id}" ${p.id === st.cmp ? "selected" : ""}>${esc(p.name)} (${esc(p.team)})</option>`).join("")}</select></label></div>
      ${radarSVG(entries, st.mode)}
      <p class="note">${st.mode === "pct" ? "Numbers are percentiles: 50 is the middle of the group (dashed ring), 90 means better than 90% of them." : "Numbers are standard deviations from the group average (dashed ring). The edge is +2.5; values beyond it sit on the edge."} Shaded sector: passing. Tackles + interceptions and recoveries are adjusted for how much of the ball the opponents had.${a.v.some((v) => v == null) ? " Line-breaking passes weren't recorded before 2025/26." : ""}</p>
      ${im ? `<div style="margin-top:14px;border-top:1px solid var(--line);padding-top:12px"><h3>Model impact <span class="small" style="font-weight:400">· latest, last 365 days</span></h3>
        <p class="small" style="margin:2px 0 0">How much the model thinks ${esc(a.name)} changes his team's chances (expected goals per 90 minutes) compared with an average ${esc(groups[im.group].replace(/s$/, ""))}.</p>
        <div class="impact">${[["Attack", im.att_vs_group, im.att_pct, "more xG for his team"], ["Defence", im.def_vs_group, im.def_pct, "less xG for the opponent"], ["Overall", im.net_vs_group, im.net_pct, "both together"]].map(([k, v, p, h]) => `<div><div class="num">${sgn(v, 3)}</div><div class="lab">${k}: ${h}<br>${ord(p)} percentile</div></div>`).join("")}</div></div>` : ""}
      <table class="raw"><thead><tr><th>Per 90 minutes</th><th>${esc(surname(a.name))}</th>${b ? `<th style="color:${cb}">${esc(surname(b.name))}</th>` : ""}<th>Group median</th></tr></thead>
      <tbody>${M.map((m, i) => `<tr><td>${esc(m.label)}</td><td>${fmtV(m.key, a.v[i])}</td>${b ? `<td>${fmtV(m.key, b.v[i])}</td>` : ""}<td>${fmtV(m.key, a.med[i])}</td></tr>`).join("")}</tbody></table></div>`;
    sel("pCmp").onchange = (e) => { st.cmp = +e.target.value || null; sync(); drawView(); };
    sel("pview").querySelectorAll("[data-mode]").forEach((x) => (x.onclick = () => { st.mode = x.dataset.mode; sync(); drawView(); }));
  };
  fillTeams(); sel("pSort").value = st.sort;
  sel("pSeason").onchange = (e) => { st.season = e.target.value; st.cmp = null; if (!find(st.id)) st.id = (P.players.find((p) => p.season === st.season && p.name === find(st.id)?.name) || P.players.find((p) => p.season === st.season)).id; fillTeams(); sync(); drawList(); drawView(); };
  sel("pGroup").onchange = (e) => { st.group = e.target.value; sync(); drawList(); };
  sel("pTeam").onchange = (e) => { st.team = e.target.value; sync(); drawList(); };
  sel("pSort").onchange = (e) => { st.sort = e.target.value; sync(); drawList(); };
  sel("pQ").oninput = (e) => { st.q = e.target.value.trim().toLowerCase(); drawList(); };
  sel("plist").onclick = (e) => { const bt = e.target.closest("button[data-id]"); if (!bt) return; st.id = +bt.dataset.id; st.cmp = null; sync(); drawList(); drawView(); if (innerWidth < 900) sel("pview").scrollIntoView({ behavior: "smooth" }); };
  wireTips(sel("pview"));
  drawList(); drawView();
}
function fmtV(key, v) { if (v == null) return "n/a"; if (META.radar.rates.includes(key)) return v.toFixed(0) + "%"; return Math.abs(v) < 1 ? v.toFixed(2) : v.toFixed(1); }

/* ------------------------------------------------------------------ Teams */
async function viewTeams(r) {
  const D = await load("teams.json");
  const names = Object.keys(D.teams).filter((t) => D.teams[t].strength.length).sort();
  const team = names.includes(r.arg) ? r.arg : "Arsenal";
  const t = D.teams[team], c = club(team);
  const rankNow = D.current_order.indexOf(team);
  const dims = META.style_dims;
  const styleNow = names.map((n) => D.teams[n].style_now).filter(Boolean);
  const range = dims.map((_, i) => { const v = styleNow.map((s) => s[i]); return [Math.min(...v), Math.max(...v)]; });
  const pos = (v, i) => { const m = Math.max(Math.abs(range[i][0]), Math.abs(range[i][1])) * 1.08; return 50 + (v / m) * 50; };
  const current = D.current_order.map((n) => ({ n, v: D.teams[n].current?.npxg_diff })).filter((x) => x.v != null);
  const vmax = Math.max(...current.map((x) => Math.abs(x.v)));

  $app.innerHTML = `
    <div class="section-head" style="margin-top:0"><div><div class="kicker">Teams</div><h2 class="display">Strength and style</h2></div></div>
    <div class="team-pick" id="tpick">${names.map((n) => `<button class="${n === team ? "on" : ""}" data-t="${esc(n)}"><span class="dot" style="background:${club(n).chart}"></span>${esc(n)}</button>`).join("")}</div>
    <div class="team-band section" style="background:${c.chart};color:${c.ink};margin-top:18px">
      <div><div class="kicker" style="color:inherit;opacity:.8">Premier League</div><div class="club">${esc(team)}</div></div>
      ${t.current ? `<div style="display:flex;gap:26px;flex-wrap:wrap">${[[t.current.npxg_for.toFixed(2), "Expected goals for, per match"], [t.current.npxg_against.toFixed(2), "Expected goals against"], [rankNow >= 0 ? ord(rankNow + 1) : "–", "Strength rank today"]].map(([n, l]) => `<div class="tile"><div class="num">${n}</div><div class="lab">${l}</div></div>`).join("")}</div>` : `<p style="margin:0;opacity:.85">Not in the 2026/27 Premier League.</p>`}
    </div>
    <div class="grid g2 section" style="margin-top:16px">
      <div class="card"><h3>Strength over time</h3><p class="small" style="margin:2px 0 8px">The team-strength model's rating before each match: chances created minus chances conceded, on a log scale. 0 is an average Premier League team. Starts in October 2023, once the model had a few matches to go on.</p><div id="schart"></div></div>
      <div class="card"><h3>How they play${t.style_date ? ` <span class="small" style="font-weight:400">· as of ${fmtDate(t.style_date)}</span>` : ""}</h3><p class="small" style="margin:2px 0 4px">Four style dimensions found in the passing networks. Grey dots are the other teams.</p>
        ${t.style_now ? dims.map((d, i) => `<div class="style-row"><div style="display:flex;justify-content:space-between"><strong style="font-size:14px">${esc(d.name)}</strong><span class="small">${t.style_now[i] >= 0 ? d.high : d.low}</span></div>
          <div class="style-track"><span class="mid"></span>${styleNow.map((s) => `<span class="lg" style="left:calc(${pos(s[i], i)}% - 2px)"></span>`).join("")}<span class="me" style="left:${pos(t.style_now[i], i)}%;background:${c.chart}" data-tip="${esc(`<b>${team}</b><br>${d.name}: ${t.style_now[i].toFixed(2)}`)}"></span></div>
          <div class="ends"><span>${esc(d.low)}</span><span>${esc(d.high)}</span></div></div>`).join("") : `<p class="muted">No style scores (the dimensions start in 2024/25).</p>`}</div>
    </div>
    <div class="grid g2 section" style="margin-top:16px">
      <div class="card"><h3>Season by season</h3><div class="scroll-x"><table class="seasons"><thead><tr><th>Season</th><th>Pos</th><th>Points</th><th>Expected points</th><th>Goals</th><th>xG for</th><th>xG against</th></tr></thead>
        <tbody>${t.seasons.map((s) => `<tr><td>${seasonLabel(s.season)}</td><td>${s.position}</td><td>${s.points}</td><td>${s.xpts.toFixed(1)}</td><td>${s.gf}–${s.ga}</td><td>${s.npxg_for.toFixed(1)}</td><td>${s.npxg_against.toFixed(1)}</td></tr>`).join("") || `<tr><td colspan="7" class="muted">No complete season in the data.</td></tr>`}</tbody></table></div>
        <p class="note">Expected points: the points a team would expect from the chances it created and conceded. xG excludes penalties.</p></div>
      <div class="card"><h3>Strongest teams today</h3><p class="small" style="margin:2px 0 8px">Expected goal difference per match against an average team, after round 5 of 2026/27.</p>
        <div class="rank" id="trank">${current.map((x, i) => `<button class="${x.n === team ? "on" : ""}" data-t="${esc(x.n)}"><span class="small">${i + 1}</span><span class="dot" style="background:${club(x.n).chart}"></span><span>${esc(x.n)}</span>
          <span style="position:relative;height:10px"><span class="rb" style="position:absolute;${x.v >= 0 ? "left:50%" : `right:50%`};width:${(Math.abs(x.v) / vmax) * 50}%;background:${x.n === team ? club(x.n).chart : "#cfcec8"}"></span><span style="position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;background:#d6d5cf"></span></span>
          <span class="rv">${x.v >= 0 ? "+" : "−"}${Math.abs(x.v).toFixed(2)}</span></button>`).join("")}</div></div>
    </div>`;
  const go = (e) => { const b = e.target.closest("[data-t]"); if (b) location.hash = "#/teams/" + encodeURIComponent(b.dataset.t); };
  document.getElementById("tpick").onclick = go; document.getElementById("trank").onclick = go;
  const ser = t.strength.filter((s) => s[0] >= "2023-10-01");
  document.getElementById("schart").innerHTML = ser.length > 1 ? strengthChart(ser, c.chart) : `<p class="muted">Not enough matches to draw.</p>`;
  wireTips($app);
}

function strengthChart(series, col) {
  const W = 640, H = 260, L = 36, R = 10, T = 10, B = 28;
  const t0 = new Date(series[0][0]).getTime(), t1 = new Date(series.at(-1)[0]).getTime();
  const all = series.map((s) => s[1]), lo = Math.min(-0.3, ...all) - 0.05, hi = Math.max(0.3, ...all) + 0.05;
  const X = (d) => L + ((new Date(d).getTime() - t0) / Math.max(1, t1 - t0)) * (W - L - R), Y = (v) => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
  let g = "";
  [-0.5, -0.25, 0, 0.25, 0.5, 0.75].filter((v) => v > lo && v < hi).forEach((v) => {
    g += `<line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}" stroke="${v === 0 ? "#bdbcb5" : "#eceae5"}" stroke-width="1" ${v === 0 ? 'stroke-dasharray="4 3"' : ""}/><text x="${L - 6}" y="${Y(v) + 4}" font-size="11" text-anchor="end" fill="${TOK.ink3}">${v > 0 ? "+" : ""}${v}</text>`;
  });
  ["2023-08-01", "2024-08-01", "2025-08-01", "2026-08-01"].forEach((d, i, arr) => {
    const tt = new Date(d).getTime(); if (tt < t0 - 864e5 * 60 || tt > t1) return;
    const x = Math.max(L, X(d));
    g += `<line x1="${x}" x2="${x}" y1="${T}" y2="${H - B}" stroke="#eceae5"/>${x < W - R - 50 ? `<text x="${x + 4}" y="${H - 9}" font-size="11" fill="${TOK.ink3}">${d.slice(0, 4)}/${String(+d.slice(2, 4) + 1)}</text>` : ""}`;
  });
  const path = series.map((s, i) => `${i ? "L" : "M"}${X(s[0]).toFixed(1)} ${Y(s[1]).toFixed(1)}`).join(" ");
  g += `<path d="${path}" fill="none" stroke="${col}" stroke-width="2.2" stroke-linejoin="round"/>`;
  g += series.map((s) => `<circle cx="${X(s[0]).toFixed(1)}" cy="${Y(s[1]).toFixed(1)}" r="6" fill="transparent" data-tip="${esc(`${fmtDate(s[0])}<br>Rating ${s[1] >= 0 ? "+" : ""}${s[1].toFixed(2)}`)}"/>`).join("");
  const last = series.at(-1);
  g += `<circle cx="${X(last[0])}" cy="${Y(last[1])}" r="4" fill="${col}" stroke="#fff" stroke-width="1.5"/>`;
  return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto" role="img" aria-label="Team strength over time">${g}</svg>`;
}

/* ------------------------------------------------------------------ Method */
async function viewMethod() {
  const L = META.ladder;
  $app.innerHTML = `<article class="prose">
    <div class="kicker">How it works</div><h2 class="display">The model, in plain words</h2>
    <p>The project asks one question: <strong>which starting XI gives a team the most expected points against a specific opponent?</strong> To answer it, I built a forecasting model one layer at a time and only kept a layer if it predicted results better on matches it had never seen.</p>
    <h2 class="display">The data</h2>
    <p>Four Premier League seasons (2023/24 to round 5 of 2026/27, ${META.facts.matches.toLocaleString()} matches) from FotMob: every shot with its expected goals (xG), each team's passing network (average positions and passes between players), and box-score stats for every player. Bookmakers' closing odds serve as a benchmark.</p>
    <h2 class="display">The layers</h2>
    <ol>
      <li><strong>Team strength.</strong> Each team gets an attack and a defence rating from the chances (non-penalty xG) it creates and concedes, with recent matches counting more.</li>
      <li><strong>Who plays.</strong> Each match is cut into stretches where the same 22 players are on the pitch. Every player gets an attacking and a defensive effect, starting from what his stats suggest and adjusted by what happens to his team's chances while he plays. A lineup's strength is the sum of its players.</li>
      <li><strong>Style clash.</strong> Passing networks reveal four style dimensions (control vs direct, central vs wide, forward reach, build-up through the keeper). This layer lets the effect of a style depend on the opponent's.</li>
      <li><strong>Player matchups.</strong> Who faces whom: zone-by-zone quality, dribblers against defenders, aerial duels, pressing against build-up.</li>
    </ol>
    <p>Every setting was chosen on 2024/25; 2025/26 and the start of 2026/27 (430 matches) were kept aside as the test. Each layer had to beat the previous one on the test matches, with confidence intervals from resampling.</p>
    <h2 class="display">What came out</h2>
    <p>Team strength closes ${Math.round(L[1].share_of_market_gain * 100)}% of the gap between guessing and the bookmakers. Adding who plays raises that to ${Math.round(L[2].share_of_market_gain * 100)}%, a clear and statistically solid gain. Style clash and matchups add nothing measurable: their score (${L[4].rps}) is within 0.0001 of the player model's (${L[2].rps}), well inside the margin of error.</p>
    <p>The lineup picker uses the player model. For each team and match it compares the usual XI with the best XI it can build from the matchday squad, within the manager's formation and position by position, and recommends a change only when it is at least 70% sure the change is better. It agrees with managers on 8.2 of 11 players on average, and the opponent changes its pick in only 8.5% of selections.</p>
    <h2 class="display">Limits</h2>
    <ul>
      <li><strong>The model doesn't know about fitness, rotation or tactics.</strong> More than half the players it brings in hadn't started any of the team's previous three matches; managers often know why.</li>
      <li><strong>Defending is under-rated.</strong> Out of sample, differences in defensive quality matter about three times more than the model thinks.</li>
      <li><strong>The passing data are match totals</strong> with no timestamps, so networks can't be split at substitutions or by game state.</li>
      <li><strong>Opponent effects could exist below what four seasons can detect.</strong> The finding is "no measurable effect", not "no effect".</li>
    </ul>
    <p><a class="cta" href="https://github.com/Benjaminvida1/opponent-aware-xi">Code, data and every result on GitHub</a></p>
  </article>`;
}
