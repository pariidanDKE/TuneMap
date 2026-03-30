"""
visualize_graph.py

Pulls the top 10 artists (by track count) and their tracks, albums, and genres
from Neo4j and renders an interactive HTML graph using pyvis.

Output: Data/graph.html
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from pyvis.network import Network

load_dotenv()

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
OUTPUT_PATH    = Path("Data/graph.html")

# ── Colours per node type ─────────────────────────────────────────────────────

COLOURS = {
    "Artist":  "#1DB954",   # green
    "Track":   "#4ECDC4",   # teal
    "Album":   "#FFE66D",   # yellow
    "Genre":   "#FF6B6B",   # red
    "Era":     "#A29BFE",   # purple
}

SIZES = {
    "Artist": 40,
    "Album":  25,
    "Genre":  30,
    "Era":    20,
    "Track":  10,
}

# ── Query ─────────────────────────────────────────────────────────────────────

# Get top 10 artists by track count
TOP_ARTISTS_QUERY = """
MATCH (a:Artist)<-[:BY]-(t:Track)
RETURN a.name AS artist, count(t) AS track_count
ORDER BY track_count DESC
LIMIT 10
"""

# Pull full subgraph for those artists
SUBGRAPH_QUERY = """
MATCH (a:Artist)<-[:BY]-(t:Track)
WHERE a.name IN $artists
OPTIONAL MATCH (t)-[:ON]->(al:Album)
OPTIONAL MATCH (t)-[:IN_GENRE]->(g:Genre)
OPTIONAL MATCH (t)-[:FEATURES]->(fa:Artist)
RETURN a, t, al, g, collect(fa) AS featured
"""


def build_graph(driver) -> Network:
    net = Network(
        height="100vh",
        width="100%",
        bgcolor="#0d0d0d",
        font_color="#e0e0e0",
        notebook=False,
    )
    net.barnes_hut(
        gravity=-8000,
        central_gravity=0.3,
        spring_length=150,
        spring_strength=0.05,
        damping=0.09,
    )

    added_nodes = set()

    def add_node(node_id: str, label: str, node_type: str, title: str = ""):
        if node_id not in added_nodes:
            net.add_node(
                node_id,
                label=label,
                color=COLOURS.get(node_type, "#888"),
                size=SIZES.get(node_type, 15),
                title=title,
                font={"size": 12 if node_type == "Track" else 16},
            )
            added_nodes.add(node_id)

    with driver.session() as session:
        # Get top 10 artists
        top_artists = [r["artist"] for r in session.run(TOP_ARTISTS_QUERY)]
        print(f"Top 10 artists: {top_artists}")

        # Pull subgraph
        results = session.run(SUBGRAPH_QUERY, {"artists": top_artists})

        edges = set()

        for record in results:
            artist   = record["a"]
            track    = record["t"]
            album    = record["al"]
            genre    = record["g"]
            featured = record["featured"]

            # Artist node
            a_id = f"artist_{artist['name']}"
            add_node(a_id, artist["name"], "Artist", f"Artist: {artist['name']}")

            # Track node
            t_id = f"track_{track['persistent_id']}"
            play_count = track.get("play_count", 0)
            add_node(
                t_id,
                track["name"],
                "Track",
                f"{track['name']}\nPlays: {play_count}\nYear: {track.get('year', '?')}",
            )

            edge = (t_id, a_id, "BY")
            if edge not in edges:
                net.add_edge(t_id, a_id, color="#555", width=1)
                edges.add(edge)

            # Album node
            if album:
                al_id = f"album_{album['title']}"
                add_node(al_id, album["title"], "Album", f"Album: {album['title']}")
                edge = (t_id, al_id, "ON")
                if edge not in edges:
                    net.add_edge(t_id, al_id, color="#444", width=1)
                    edges.add(edge)

            # Featured artist nodes
            for fa in featured:
                if fa:
                    fa_id = f"artist_{fa['name']}"
                    add_node(fa_id, fa["name"], "Artist", f"Artist: {fa['name']}")
                    edge = (t_id, fa_id, "FEATURES")
                    if edge not in edges:
                        net.add_edge(t_id, fa_id, color="#FF6B6B", width=1, dashes=True, title="features")
                        edges.add(edge)

            # Genre node
            if genre:
                g_id = f"genre_{genre['name']}"
                add_node(g_id, genre["name"], "Genre", f"Genre: {genre['name']}")
                edge = (t_id, g_id, "IN_GENRE")
                if edge not in edges:
                    net.add_edge(t_id, g_id, color="#333", width=1)
                    edges.add(edge)

    return net


def add_legend(html: str) -> str:
    legend = """
<div style="position:fixed;top:20px;right:20px;background:#1a1a1a;border:1px solid #333;
            border-radius:12px;padding:16px 20px;font-family:Inter,sans-serif;z-index:999;">
  <div style="color:#888;font-size:11px;text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px">Node Types</div>
  <div style="display:flex;flex-direction:column;gap:7px;">
    <div style="display:flex;align-items:center;gap:8px"><div style="width:14px;height:14px;border-radius:50%;background:#1DB954"></div><span style="color:#e0e0e0;font-size:13px">Artist</span></div>
    <div style="display:flex;align-items:center;gap:8px"><div style="width:14px;height:14px;border-radius:50%;background:#4ECDC4"></div><span style="color:#e0e0e0;font-size:13px">Track</span></div>
    <div style="display:flex;align-items:center;gap:8px"><div style="width:14px;height:14px;border-radius:50%;background:#FFE66D"></div><span style="color:#e0e0e0;font-size:13px">Album</span></div>
    <div style="display:flex;align-items:center;gap:8px"><div style="width:14px;height:14px;border-radius:50%;background:#FF6B6B"></div><span style="color:#e0e0e0;font-size:13px">Genre</span></div>
  </div>
  <div style="color:#888;font-size:11px;text-transform:uppercase;letter-spacing:.08em;margin:10px 0 6px">Edges</div>
  <div style="display:flex;flex-direction:column;gap:7px;">
    <div style="display:flex;align-items:center;gap:8px"><div style="width:20px;height:2px;background:#555"></div><span style="color:#e0e0e0;font-size:13px">Primary artist</span></div>
    <div style="display:flex;align-items:center;gap:8px"><div style="width:20px;height:2px;background:#FF6B6B;border-top:2px dashed #FF6B6B"></div><span style="color:#e0e0e0;font-size:13px">Features</span></div>
  </div>
</div>
"""
    return html.replace("</body>", legend + "</body>")


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    print("Building graph ...")
    net = build_graph(driver)
    driver.close()

    # Write to temp file then inject legend
    tmp = OUTPUT_PATH.with_suffix(".tmp.html")
    net.write_html(str(tmp))

    html = tmp.read_text(encoding="utf-8")
    html = add_legend(html)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    tmp.unlink()

    print(f"Graph written → {OUTPUT_PATH}")
    print(f"Nodes: {len(net.nodes)}  Edges: {len(net.edges)}")


if __name__ == "__main__":
    main()
