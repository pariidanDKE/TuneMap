"""
visualize_library.py

Generates an interactive HTML dashboard from library.json / overview.json.
Output: Data/dashboard.html
"""

import argparse
import json
from pathlib import Path

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ── Colour palette ────────────────────────────────────────────────────────────
ACCENT = "#1DB954"          # Spotify-green-ish
BG     = "#0d0d0d"
CARD   = "#1a1a1a"
TEXT   = "#e0e0e0"
MUTED  = "#888888"
SEQ    = px.colors.sequential.Plasma


def load(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Individual chart builders ─────────────────────────────────────────────────

def chart_genre_bar(overview: dict) -> go.Figure:
    genres = overview["top_genres"]
    labels = list(genres.keys())[:20]
    values = [genres[l] for l in labels]

    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker=dict(color=values, colorscale="Plasma", showscale=False),
        hovertemplate="%{y}: %{x} tracks<extra></extra>",
    ))
    fig.update_layout(
        title="Tracks by Genre",
        **_layout(extra_y=dict(autorange="reversed")),
    )
    return fig


def chart_genre_pie(overview: dict) -> go.Figure:
    genres = overview["top_genres"]
    labels = list(genres.keys())[:12]
    values = [genres[l] for l in labels]
    other  = sum(list(genres.values())[12:])
    if other:
        labels.append("Other")
        values.append(other)

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.45,
        marker=dict(colors=px.colors.qualitative.Vivid),
        textinfo="label+percent",
        hovertemplate="%{label}: %{value} tracks (%{percent})<extra></extra>",
    ))
    fig.update_layout(title="Genre Distribution", **_layout())
    return fig


def chart_top_artists(overview: dict) -> go.Figure:
    artists = overview["top_artists"]
    labels  = list(artists.keys())[:25]
    values  = [artists[l] for l in labels]

    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker=dict(color=values, colorscale="Teal", showscale=False),
        hovertemplate="%{y}: %{x} tracks<extra></extra>",
    ))
    fig.update_layout(
        title="Top 25 Artists by Track Count",
        **_layout(height=650, extra_y=dict(autorange="reversed")),
    )
    return fig


def chart_tracks_by_year(overview: dict) -> go.Figure:
    by_year = overview["tracks_by_year"]
    years   = [int(y) for y in by_year.keys()]
    counts  = list(by_year.values())

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=years, y=counts,
        marker=dict(color=counts, colorscale="Plasma", showscale=False),
        hovertemplate="Year %{x}: %{y} tracks<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=years, y=counts,
        mode="lines",
        line=dict(color=ACCENT, width=2),
        hoverinfo="skip",
    ))
    fig.update_layout(
        title="Tracks by Release Year",
        xaxis_title="Year",
        yaxis_title="Tracks",
        showlegend=False,
        **_layout(),
    )
    return fig


def chart_play_counts(tracks: list) -> go.Figure:
    counts = [t["Play Count"] for t in tracks if t.get("Play Count")]
    fig = go.Figure(go.Histogram(
        x=counts,
        nbinsx=50,
        marker_color=ACCENT,
        opacity=0.85,
        hovertemplate="Play count %{x}: %{y} tracks<extra></extra>",
    ))
    fig.update_layout(
        title="Play Count Distribution",
        xaxis_title="Play Count",
        yaxis_title="Number of Tracks",
        **_layout(),
    )
    return fig


def chart_top_played(tracks: list) -> go.Figure:
    played = sorted(
        (t for t in tracks if t.get("Play Count")),
        key=lambda t: t["Play Count"],
        reverse=True,
    )[:20]

    labels = [f"{t['Name']} — {t.get('Artist','?')}" for t in played]
    values = [t["Play Count"] for t in played]

    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker=dict(color=values, colorscale="Magma", showscale=False),
        hovertemplate="%{y}: %{x} plays<extra></extra>",
    ))
    fig.update_layout(
        title="Top 20 Most Played Tracks",
        **_layout(height=620, extra_y=dict(autorange="reversed")),
    )
    return fig


def chart_loved_vs_explicit(overview: dict) -> go.Figure:
    t = overview["totals"]
    total = t["tracks"]

    categories = ["Explicit", "Favorited / Loved", "Apple Music", "Compilation"]
    field_map  = {
        "Explicit":          "explicit_tracks",
        "Favorited / Loved": "loved_tracks",       # added below
        "Apple Music":       "apple_music_tracks",
        "Compilation":       "compilation_tracks",  # added below
    }

    # counts computed separately if not in overview
    values = [
        t.get("explicit_tracks", 0),
        t.get("loved_tracks", 0),
        t.get("apple_music_tracks", 0),
        t.get("compilation_tracks", 0),
    ]

    fig = go.Figure(go.Bar(
        x=categories, y=values,
        marker=dict(color=[ACCENT, "#FF6B6B", "#4ECDC4", "#FFE66D"]),
        text=[f"{v}<br>({v/total*100:.1f}%)" for v in values],
        textposition="outside",
        hovertemplate="%{x}: %{y} tracks<extra></extra>",
    ))
    fig.update_layout(
        title="Track Flags Overview",
        yaxis_title="Track Count",
        **_layout(extra_y=dict(range=[0, max(values) * 1.2])),
    )
    return fig


def chart_date_added(tracks: list) -> go.Figure:
    from collections import Counter
    months = Counter()
    for t in tracks:
        da = t.get("Date Added")
        if da:
            ym = da[:7]  # "YYYY-MM"
            months[ym] += 1

    sorted_months = sorted(months.items())
    x = [m[0] for m in sorted_months]
    y = [m[1] for m in sorted_months]

    fig = go.Figure(go.Bar(
        x=x, y=y,
        marker=dict(color=y, colorscale="Teal", showscale=False),
        hovertemplate="%{x}: %{y} tracks added<extra></extra>",
    ))
    fig.update_layout(
        title="Tracks Added to Library per Month",
        xaxis_title="Month",
        yaxis_title="Tracks Added",
        **_layout(),
    )
    return fig


def chart_top_skipped(tracks: list) -> go.Figure:
    skipped = sorted(
        (t for t in tracks if t.get("Skip Count")),
        key=lambda t: t["Skip Count"],
        reverse=True,
    )[:15]

    labels = [f"{t['Name']} — {t.get('Artist','?')}" for t in skipped]
    values = [t["Skip Count"] for t in skipped]

    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker=dict(color=values, colorscale="Reds", showscale=False),
        hovertemplate="%{y}: %{x} skips<extra></extra>",
    ))
    fig.update_layout(
        title="Top 15 Most Skipped Tracks",
        **_layout(height=520, extra_y=dict(autorange="reversed")),
    )
    return fig


# ── Layout helper ─────────────────────────────────────────────────────────────

def _layout(height=420, extra_x=None, extra_y=None, **kwargs) -> dict:
    base = dict(
        paper_bgcolor=BG,
        plot_bgcolor=CARD,
        font=dict(color=TEXT, family="Inter, Helvetica, Arial, sans-serif"),
        margin=dict(l=20, r=20, t=50, b=20),
        height=height,
        xaxis=dict(gridcolor="#2a2a2a", zerolinecolor="#333", **(extra_x or {})),
        yaxis=dict(gridcolor="#2a2a2a", zerolinecolor="#333", **(extra_y or {})),
    )
    base.update(kwargs)
    return base


# ── KPI card HTML ─────────────────────────────────────────────────────────────

def kpi_html(overview: dict) -> str:
    t = overview["totals"]
    cards = [
        ("🎵", f"{t['tracks']:,}",          "Tracks"),
        ("🎤", f"{t['unique_artists']:,}",   "Artists"),
        ("💿", f"{t['unique_albums']:,}",    "Albums"),
        ("🎭", f"{t['unique_genres']}",      "Genres"),
        ("⏱️", f"{t['total_playtime_hours']:.0f}h", "Total Playtime"),
        ("🔥", f"{t['explicit_tracks']:,}",  "Explicit"),
        ("💚", f"{t.get('loved_tracks',0)}", "Loved"),
        ("📋", f"{t['playlists']}",          "Playlists"),
    ]
    inner = "".join(
        f"""<div class="kpi">
               <span class="kpi-icon">{icon}</span>
               <span class="kpi-value">{val}</span>
               <span class="kpi-label">{label}</span>
            </div>"""
        for icon, val, label in cards
    )
    return f'<div class="kpi-row">{inner}</div>'


# ── Full dashboard ────────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Apple Music Library Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body   {{ background: #0d0d0d; color: #e0e0e0; font-family: Inter, Helvetica, Arial, sans-serif; }}
  header {{ padding: 2rem 2.5rem 1rem; border-bottom: 1px solid #222; }}
  header h1 {{ font-size: 1.8rem; font-weight: 700; }}
  header p  {{ color: #888; font-size: 0.9rem; margin-top: 0.25rem; }}
  .kpi-row  {{ display: flex; flex-wrap: wrap; gap: 1rem; padding: 1.5rem 2.5rem; }}
  .kpi      {{ background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 12px;
               padding: 1.1rem 1.4rem; flex: 1 1 130px; min-width: 110px;
               display: flex; flex-direction: column; align-items: center; gap: 0.3rem; }}
  .kpi-icon  {{ font-size: 1.5rem; }}
  .kpi-value {{ font-size: 1.6rem; font-weight: 700; color: #1DB954; }}
  .kpi-label {{ font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: .05em; }}
  .grid      {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(540px, 1fr));
               gap: 1.2rem; padding: 0 2.5rem 2.5rem; }}
  .card      {{ background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 12px; overflow: hidden; }}
  .card.full {{ grid-column: 1 / -1; }}
</style>
</head>
<body>
<header>
  <h1>Apple Music Library</h1>
  <p>Personal library dashboard — {track_count:,} tracks · generated {date}</p>
</header>

{kpi_block}

<div class="grid">
  <div class="card full">{genre_bar}</div>
  <div class="card">{genre_pie}</div>
  <div class="card">{year_chart}</div>
  <div class="card full">{artist_chart}</div>
  <div class="card full">{top_played}</div>
  <div class="card">{play_dist}</div>
  <div class="card">{date_added}</div>
  <div class="card full">{top_skipped}</div>
  <div class="card">{flags_chart}</div>
</div>
</body>
</html>"""


def fig_to_div(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})


def build_dashboard(library_path: Path, overview_path: Path, output_path: Path):
    print("Loading data...")
    library  = load(library_path)
    overview = load(overview_path)
    tracks   = library["tracks"]

    # Enrich overview totals with fields not in parse script
    from collections import Counter
    overview["totals"]["loved_tracks"]       = sum(1 for t in tracks if t.get("Loved") or t.get("Favorited"))
    overview["totals"]["compilation_tracks"] = sum(1 for t in tracks if t.get("Compilation"))

    print("Building charts...")
    import datetime
    html = HTML_TEMPLATE.format(
        track_count = overview["totals"]["tracks"],
        date        = datetime.date.today().isoformat(),
        kpi_block   = kpi_html(overview),
        genre_bar   = fig_to_div(chart_genre_bar(overview)),
        genre_pie   = fig_to_div(chart_genre_pie(overview)),
        year_chart  = fig_to_div(chart_tracks_by_year(overview)),
        artist_chart= fig_to_div(chart_top_artists(overview)),
        top_played  = fig_to_div(chart_top_played(tracks)),
        play_dist   = fig_to_div(chart_play_counts(tracks)),
        date_added  = fig_to_div(chart_date_added(tracks)),
        top_skipped = fig_to_div(chart_top_skipped(tracks)),
        flags_chart = fig_to_div(chart_loved_vs_explicit(overview)),
    )

    output_path.write_text(html, encoding="utf-8")
    print(f"Dashboard written → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Apple Music library dashboard")
    parser.add_argument("--library",  default="Data/library.json",  help="Path to library.json")
    parser.add_argument("--overview", default="Data/overview.json",  help="Path to overview.json")
    parser.add_argument("--output",   default="Data/dashboard.html", help="Output HTML file")
    args = parser.parse_args()

    build_dashboard(
        Path(args.library),
        Path(args.overview),
        Path(args.output),
    )


if __name__ == "__main__":
    main()
