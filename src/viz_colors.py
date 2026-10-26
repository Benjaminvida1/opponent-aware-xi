"""Club colours for every team in the data, plus the chart tokens shared by all visuals.

Each club has a `kit` colour (home shirt or crest) and an `alt` colour (crest or change kit). Kit colours
are approximations of the clubs' brand colours, not official values. For drawing on the light chart
surface they are turned into `chart` colours: the same hue, darkened in OKLCH until it reaches 3:1
contrast with the surface (white shirts use the crest colour instead). `ink` is the text colour that
reads on a filled `chart` mark.

When two clubs appear together (a match view), `pair_colors` keeps the home club's colour and switches
the away club to its alternate when the two are too close to tell apart (OKLab distance < CLASH_DE).

Run `python src/viz_colors.py` to print the table and write reports/dashboard/team_colors.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- shared chart tokens (light theme)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8983"
LINE = "#d6d5cf"
NEUTRAL_MID = "#a9a8a1"     # blocked shots, secondary marks
NEUTRAL_LIGHT = "#c9c8c2"   # off-target shots
MUTED_FILL = "#f0efec"

MIN_CONTRAST = 3.0          # chart colour vs surface
CLASH_DE = 0.15             # OKLab distance below which two club colours are treated as the same

# team -> (kit, alt). Names as in data/processed (matches.home_team).
CLUBS: dict[str, tuple[str, str]] = {
    "AFC Bournemouth":          ("#DA291C", "#1A1A1A"),
    "Arsenal":                  ("#EF0107", "#063672"),
    "Aston Villa":              ("#670E36", "#95BFE5"),
    "Brentford":                ("#E30613", "#1A1A1A"),
    "Brighton & Hove Albion":   ("#0057B8", "#FFCD00"),
    "Burnley":                  ("#6C1D45", "#99D6EA"),
    "Chelsea":                  ("#034694", "#DBA111"),
    "Coventry City":            ("#59CBE8", "#1A1A1A"),
    "Crystal Palace":           ("#1B458F", "#C4122E"),
    "Everton":                  ("#003399", "#F2A900"),
    "Fulham":                   ("#1A1A1A", "#CC0000"),   # white shirt: crest black
    "Hull City":                ("#F5A12D", "#1A1A1A"),
    "Ipswich Town":             ("#3A64A3", "#DE2C37"),
    "Leeds United":             ("#1D428A", "#FFCD00"),   # white shirt: crest blue
    "Leicester City":           ("#003090", "#FDBE11"),
    "Liverpool":                ("#C8102E", "#00B2A9"),
    "Luton Town":               ("#F78F1E", "#002D62"),
    "Manchester City":          ("#6CABDD", "#1C2C5B"),
    "Manchester United":        ("#DA291C", "#1A1A1A"),
    "Newcastle United":         ("#241F20", "#41B6E6"),
    "Nottingham Forest":        ("#DD0000", "#1A1A1A"),
    "Sheffield United":         ("#EE2737", "#1A1A1A"),
    "Southampton":              ("#D71920", "#FFC20E"),
    "Sunderland":               ("#EB172B", "#1A1A1A"),
    "Tottenham Hotspur":        ("#132257", "#2BB5A4"),   # white shirt: crest navy
    "West Ham United":          ("#7A263A", "#1BB1E7"),
    "Wolverhampton Wanderers":  ("#FDB913", "#231F20"),
}
ALIASES = {"Bournemouth": "AFC Bournemouth", "Brighton and Hove Albion": "Brighton & Hove Albion",
           "Brighton": "Brighton & Hove Albion", "Wolves": "Wolverhampton Wanderers",
           "Spurs": "Tottenham Hotspur", "Man City": "Manchester City", "Man United": "Manchester United"}
FALLBACK = ("#2a78d6", "#eb6834")


# ---------------------------------------------------------------- colour maths
def hex_to_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)]) / 255.0


def rgb_to_hex(rgb) -> str:
    rgb = np.clip(np.asarray(rgb, float), 0, 1)
    return "#" + "".join(f"{int(round(c * 255)):02x}" for c in rgb)


def _lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _delin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055)


def luminance(h: str) -> float:
    r, g, b = _lin(hex_to_rgb(h))
    return float(0.2126 * r + 0.7152 * g + 0.0722 * b)


def contrast(a: str, b: str) -> float:
    la, lb = sorted([luminance(a), luminance(b)], reverse=True)
    return (la + 0.05) / (lb + 0.05)


def to_oklab(h: str) -> np.ndarray:
    r, g, b = _lin(hex_to_rgb(h))
    l_ = np.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m_ = np.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s_ = np.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return np.array([0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
                     1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
                     0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_])


def from_oklab(lab) -> str:
    L, a, b = lab
    l_ = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m_ = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    rgb = [4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
           -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
           -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_]
    return rgb_to_hex(_delin(np.clip(rgb, 0, 1)))


def delta_e(a: str, b: str) -> float:
    return float(np.linalg.norm(to_oklab(a) - to_oklab(b)))


def darken_to_contrast(h: str, target: float = MIN_CONTRAST, surface: str = SURFACE) -> str:
    """Lower OKLab lightness (keeping hue and chroma direction) until the colour reaches `target` contrast."""
    lab = to_oklab(h)
    out = h
    for _ in range(80):
        if contrast(out, surface) >= target:
            return out
        lab = np.array([lab[0] - 0.01, lab[1] * 0.995, lab[2] * 0.995])
        out = from_oklab(lab)
    return out


def mix(h: str, w: float, base: str = SURFACE) -> str:
    """w = 1 gives h, w = 0 gives base (blend in linear RGB)."""
    a, b = _lin(hex_to_rgb(h)), _lin(hex_to_rgb(base))
    return rgb_to_hex(_delin(w * a + (1 - w) * b))


def ink_on(fill: str) -> str:
    """White or near-black text, whichever reads better on the fill."""
    return "#ffffff" if contrast("#ffffff", fill) >= contrast(INK, fill) else INK


# ---------------------------------------------------------------- club API
def canonical(team: str) -> str:
    return ALIASES.get(team, team)


def club(team: str) -> dict:
    """All colour tokens for one club."""
    kit, alt = CLUBS.get(canonical(team), FALLBACK)
    chart = darken_to_contrast(kit)
    chart_alt = darken_to_contrast(alt)
    return {"team": canonical(team), "kit": kit, "alt": alt, "chart": chart, "chart_alt": chart_alt,
            "ink": ink_on(chart), "ink_alt": ink_on(chart_alt),
            "tint": mix(chart, 0.14), "ramp": ramp(chart)}


def color(team: str) -> str:
    """The club's chart colour (use this for single-team visuals)."""
    return club(team)["chart"]


def ramp(chart: str, n: int = 6) -> list[str]:
    """Sequential ramp for one club: light tint of the club colour -> full colour."""
    return [mix(chart, w) for w in np.linspace(0.28, 1.0, n)]


def cmap(team_or_hex: str, lo: float = 0.22):
    """Matplotlib colormap from a light tint to the club colour."""
    from matplotlib.colors import LinearSegmentedColormap
    h = team_or_hex if team_or_hex.startswith("#") else color(team_or_hex)
    return LinearSegmentedColormap.from_list(f"club_{h}", [mix(h, lo), mix(h, 0.6), h])


def pair_colors(home: str, away: str) -> tuple[str, str]:
    """Chart colours for two clubs shown together; the away club switches to its alternate on a clash."""
    a, b = club(home), club(away)
    if delta_e(a["chart"], b["chart"]) >= CLASH_DE:
        return a["chart"], b["chart"]
    if delta_e(a["chart"], b["chart_alt"]) >= CLASH_DE:
        return a["chart"], b["chart_alt"]
    return a["chart"], darken_to_contrast(FALLBACK[1])


def all_clubs() -> dict[str, dict]:
    return {t: club(t) for t in CLUBS}


def write_json(path: Path | None = None) -> Path:
    path = path or ROOT / "reports" / "dashboard" / "team_colors.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tokens = {"surface": SURFACE, "ink": INK, "ink_2": INK_2, "ink_3": INK_3, "line": LINE,
              "neutral_mid": NEUTRAL_MID, "neutral_light": NEUTRAL_LIGHT, "muted_fill": MUTED_FILL}
    path.write_text(json.dumps({"tokens": tokens, "clubs": all_clubs(),
                                "rules": {"min_contrast": MIN_CONTRAST, "clash_delta_e_oklab": CLASH_DE}},
                               indent=2, ensure_ascii=False))
    return path


def swatch_figure(path: Path | None = None) -> Path:
    """Reference sheet: every club's kit, chart and alternate colours."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Inter", "DejaVu Sans"]
    path = path or ROOT / "reports" / "figures" / "viz" / "club_colours.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    clubs = all_clubs()
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(clubs) + 1.1), facecolor=SURFACE)
    ax.set_facecolor(SURFACE); ax.axis("off")
    n = len(clubs)
    ax.set_xlim(0, 8); ax.set_ylim(-0.6, n + 0.9)
    for j, h in enumerate(["Club", "Kit", "Chart", "Alternate"]):
        ax.text([0, 4.1, 5.4, 6.7][j], n + 0.35, h, fontsize=9, weight="semibold", color=INK, va="center")
    for i, (t, c) in enumerate(clubs.items()):
        y = n - 1 - i
        ax.text(0, y, t, fontsize=9, color=INK_2, va="center")
        for x, key, ink in [(4.1, "kit", None), (5.4, "chart", c["ink"]), (6.7, "chart_alt", c["ink_alt"])]:
            ax.add_patch(plt.Rectangle((x, y - 0.36), 1.1, 0.72, fc=c[key], ec=LINE, lw=0.6))
            ax.text(x + 0.55, y, c[key].upper(), fontsize=6.5, va="center", ha="center",
                    color=ink or ink_on(c[key]))
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    for t, c in all_clubs().items():
        print(f"{t:26s} kit {c['kit']}  chart {c['chart']} ({contrast(c['chart'], SURFACE):.1f}:1)  "
              f"alt {c['chart_alt']}  ink {c['ink']}")
    print("wrote", write_json())
    print("wrote", swatch_figure())
