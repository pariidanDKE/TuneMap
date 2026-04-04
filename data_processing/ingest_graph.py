"""
ingest_graph.py

Reads library.json and ingests it into a local Neo4j instance as a Knowledge Graph.

Schema:
  (:Track)-[:BY]----------->(:Artist)        primary artist
  (:Track)-[:FEATURES]------>(:Artist)        featured artists
  (:Track)-[:ON]------------>(:Album)
  (:Track)-[:IN_GENRE]------>(:Genre)
  (:Track)-[:IN_ERA]-------->(:Era)
  (:Track)-[:IN_PLAYLIST]--->(:Playlist)
  (:Album)-[:BY]------------>(:Artist)
  (:Artist)-[:IN_GENRE]----->(:Genre)         derived post-ingestion

Run:
  python data_processing/ingest_graph.py
"""

import json
import os
import re
from collections.abc import Callable
from pathlib import Path

from neo4j import GraphDatabase

# ── Genre normalisation map ───────────────────────────────────────────────────

GENRE_MAP = {
    "Hip-Hop":               "Hip-Hop/Rap",
    "Rap":                   "Hip-Hop/Rap",
    "Old School Rap":        "Hip-Hop/Rap",
    "Hardcore Rap":          "Hip-Hop/Rap",
    "South African Hip-Hop": "Hip-Hop/Rap",
    "Dirty South":           "Hip-Hop/Rap",
    "Hip-Hop in Russian":    "Hip-Hop/Rap",
    "Latin Rap":             "Hip-Hop/Rap",
    "Alternative Rap":       "Hip-Hop/Rap",
}

# ── Album name normalisation ──────────────────────────────────────────────────

ALBUM_SUFFIX_RE = re.compile(
    r"\s*[\(\-]\s*(Deluxe Edition|Deluxe|Remaster(ed)?|Explicit|Special Edition"
    r"|Expanded Edition|Anniversary Edition|Bonus Track Version)\s*\)?$",
    re.IGNORECASE,
)

def normalise_album(name: str) -> str:
    return ALBUM_SUFFIX_RE.sub("", name).strip()

# ── Era derivation ────────────────────────────────────────────────────────────

def year_to_era(year: int) -> str:
    if year < 1990: return "Pre-90s"
    if year < 2000: return "90s"
    if year < 2010: return "2000s"
    if year < 2020: return "2010s"
    return "2020s"

# ── Featured artist parsing ───────────────────────────────────────────────────

def parse_featured(artist_field: str, primary: str) -> list[str]:
    """
    Split the Artist field into individual names, remove the primary artist,
    return the rest as featured artists.
    """
    if not artist_field or artist_field == primary:
        return []

    # Normalise all separator variants to " & "
    normalised = (
        artist_field
        .replace(" featuring ", " & ")
        .replace(" Featuring ", " & ")
        .replace(" FEATURING ", " & ")
        .replace(" feat. ", " & ")
        .replace(" Feat. ", " & ")
        .replace(" ft. ", " & ")
        .replace(" Ft. ", " & ")
        .replace(" X ", " & ")
        .replace(" x ", " & ")
        .replace(" with ", " & ")
        .replace(" With ", " & ")
    )

    # Split on ", " first, then on " & "
    parts = []
    for chunk in normalised.split(", "):
        for name in chunk.split(" & "):
            name = name.strip()
            if name and name != primary:
                parts.append(name)

    return parts

# ── Neo4j setup ───────────────────────────────────────────────────────────────

CONSTRAINTS = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Track)    REQUIRE t.persistent_id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Artist)   REQUIRE a.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (al:Album)   REQUIRE al.title IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (g:Genre)    REQUIRE g.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Era)      REQUIRE e.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Playlist) REQUIRE p.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Single)   REQUIRE s.name IS UNIQUE",
]

def create_constraints(session):
    for c in CONSTRAINTS:
        session.run(c)
    print("Constraints applied.")

# ── Per-track Cypher ──────────────────────────────────────────────────────────

INGEST_TRACK = """
MERGE (t:Track {persistent_id: $persistent_id})
SET   t.name         = $name,
      t.year         = $year,
      t.release_date = $release_date,
      t.duration_ms  = $duration_ms,
      t.play_count   = $play_count,
      t.skip_count   = $skip_count,
      t.loved        = $loved,
      t.explicit     = $explicit,
      t.date_added   = $date_added,
      t.track_number = $track_number

MERGE (ar:Artist {name: $primary_artist})
MERGE (g:Genre {name: $genre})
MERGE (e:Era   {name: $era})

MERGE (t)-[:BY]->(ar)
MERGE (t)-[:IN_GENRE]->(g)
MERGE (t)-[:IN_ERA]->(e)
"""

INGEST_ALBUM_TRACK = """
MATCH (t:Track {persistent_id: $persistent_id})
MATCH (ar:Artist {name: $primary_artist})
MERGE (al:Album {title: $album})
SET   al.year = $year
MERGE (al)-[:BY]->(ar)
MERGE (t)-[:ON]->(al)
"""

INGEST_SINGLE_TRACK = """
MATCH (t:Track {persistent_id: $persistent_id})
MERGE (s:Single {name: $name})
MERGE (t)-[:IS_SINGLE]->(s)
"""

ADD_FEATURED = """
MATCH (t:Track {persistent_id: $persistent_id})
MERGE (fa:Artist {name: $featured})
MERGE (t)-[:FEATURES]->(fa)
"""

ADD_PLAYLIST = """
MATCH (t:Track {persistent_id: $persistent_id})
MERGE (p:Playlist {name: $playlist})
MERGE (t)-[:IN_PLAYLIST]->(p)
"""

ARTIST_GENRE_EDGES = """
MATCH (a:Artist)<-[:BY]-(t:Track)-[:IN_GENRE]->(g:Genre)
WITH a, g, count(t) AS track_count
MERGE (a)-[r:IN_GENRE]->(g)
SET r.track_count = track_count
"""

# ── Playlist lookup ───────────────────────────────────────────────────────────

def build_playlist_lookup(playlists: list) -> dict[int, list[str]]:
    """Returns {track_id: [playlist_name, ...]}"""
    lookup: dict[int, list[str]] = {}
    for pl in playlists:
        name = pl.get("Name")
        if not name or pl.get("Master") or pl.get("Distinguished Kind"):
            continue
        for tid in pl.get("Playlist Items", []):
            lookup.setdefault(tid, []).append(name)
    return lookup

# ── Core ingest function ──────────────────────────────────────────────────────

def ingest(
    library: dict,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    on_progress: Callable[[int, int, str], None] | None = None,
):
    """
    Ingest a parsed library dict into Neo4j.
    on_progress(current, total, track_name) called every track.
    """
    tracks    = library["tracks"]
    playlists = library["playlists"]
    playlist_lookup = build_playlist_lookup(playlists)
    total = len(tracks)

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    with driver.session() as session:
        create_constraints(session)

        for i, t in enumerate(tracks, 1):
            pid        = t.get("Persistent ID")
            primary    = t.get("Album Artist") or t.get("Artist", "Unknown")
            genre      = GENRE_MAP.get(t.get("Genre", "Unknown"), t.get("Genre", "Unknown"))
            album      = normalise_album(t.get("Album", "Unknown"))
            year       = t.get("Year", 0)
            era        = year_to_era(year)
            is_single  = album.endswith("- Single")
            track_name = t.get("Name", "")

            session.run(INGEST_TRACK, {
                "persistent_id":  pid,
                "name":           track_name,
                "year":           year,
                "release_date":   t.get("Release Date", ""),
                "duration_ms":    t.get("Total Time", 0),
                "play_count":     t.get("Play Count", 0),
                "skip_count":     t.get("Skip Count", 0),
                "loved":          t.get("Loved", False) or t.get("Favorited", False),
                "explicit":       t.get("Explicit", False),
                "date_added":     t.get("Date Added", ""),
                "track_number":   t.get("Track Number", 0),
                "primary_artist": primary,
                "genre":          genre,
                "era":            era,
            })

            if is_single:
                session.run(INGEST_SINGLE_TRACK, {"persistent_id": pid, "name": track_name})
            else:
                session.run(INGEST_ALBUM_TRACK, {
                    "persistent_id":  pid,
                    "primary_artist": primary,
                    "album":          album,
                    "year":           year,
                })

            for fa in parse_featured(t.get("Artist", ""), primary):
                session.run(ADD_FEATURED, {"persistent_id": pid, "featured": fa})

            for pl_name in playlist_lookup.get(t.get("Track ID"), []):
                session.run(ADD_PLAYLIST, {"persistent_id": pid, "playlist": pl_name})

            if on_progress:
                on_progress(i, total, track_name)

        session.run(ARTIST_GENRE_EDGES)

    driver.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import os
    from dotenv import load_dotenv

    load_dotenv()

    neo4j_uri      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
    neo4j_user     = os.getenv("NEO4J_USER",     "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "")
    library_path   = Path("Data/library.json")

    print(f"Loading {library_path} ...")
    with open(library_path, encoding="utf-8") as f:
        library = json.load(f)

    def on_progress(current, total, name):
        if current % 200 == 0:
            print(f"  {current}/{total} tracks ingested ...")

    print(f"Ingesting {len(library['tracks'])} tracks ...")
    ingest(library, neo4j_uri, neo4j_user, neo4j_password, on_progress=on_progress)

    print("\nIngestion complete.")
    print("Open http://localhost:7474 to explore your graph.")


if __name__ == "__main__":
    main()
