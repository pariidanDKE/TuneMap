"""
ingest_lyrics.py

Reads track_themes.json and enriches the existing Neo4j graph with:

  New relationships:
    (:Track)-[:HAS_MOOD]->(:Mood)
    (:Track)-[:HAS_TOPIC]->(:Topic)
    (:Track)-[:MENTIONS_PLACE]->(:Place)

  New Track properties:
    language, total_words, unique_words, type_token_ratio, repetition_rate

Run:
  python data_processing/ingest_lyrics.py
"""

import json
import os
from collections.abc import Callable
from pathlib import Path

from neo4j import GraphDatabase

# ── Constraints ───────────────────────────────────────────────────────────────

CONSTRAINTS = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Mood)  REQUIRE m.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (tp:Topic) REQUIRE tp.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (pl:Place) REQUIRE pl.name IS UNIQUE",
]

# ── Cypher ────────────────────────────────────────────────────────────────────

UPDATE_TRACK_PROPS = """
MATCH (t:Track {persistent_id: $persistent_id})
SET t.language          = $language,
    t.total_words       = $total_words,
    t.unique_words      = $unique_words,
    t.type_token_ratio  = $type_token_ratio,
    t.repetition_rate   = $repetition_rate,
    t.lyrics_found      = $lyrics_found
"""

ADD_MOOD = """
MATCH (t:Track {persistent_id: $persistent_id})
MERGE (m:Mood {name: $mood})
MERGE (t)-[:HAS_MOOD]->(m)
"""

ADD_TOPIC = """
MATCH (t:Track {persistent_id: $persistent_id})
MERGE (tp:Topic {name: $topic})
MERGE (t)-[:HAS_TOPIC]->(tp)
"""

ADD_PLACE = """
MATCH (t:Track {persistent_id: $persistent_id})
MERGE (pl:Place {name: $place})
MERGE (t)-[:MENTIONS_PLACE]->(pl)
"""

# ── Core ingest function ──────────────────────────────────────────────────────

def ingest(
    library: dict,
    themes: dict,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    on_progress: Callable[[int, int], None] | None = None,
):
    """
    Enrich existing Neo4j Track nodes with mood/topic/place data from themes dict.
    on_progress(current, total) called every track.
    """
    id_to_pid = {
        str(t["Track ID"]): t["Persistent ID"]
        for t in library["tracks"]
        if "Track ID" in t and "Persistent ID" in t
    }

    total   = len(themes)
    skipped = 0
    driver  = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    with driver.session() as session:
        for c in CONSTRAINTS:
            session.run(c)

        for i, (track_id, entry) in enumerate(themes.items(), 1):
            pid = id_to_pid.get(track_id)
            if not pid:
                skipped += 1
                if on_progress:
                    on_progress(i, total)
                continue

            session.run(UPDATE_TRACK_PROPS, {
                "persistent_id":    pid,
                "language":         entry.get("language"),
                "total_words":      entry.get("total_words"),
                "unique_words":     entry.get("unique_words"),
                "type_token_ratio": entry.get("type_token_ratio"),
                "repetition_rate":  entry.get("repetition_rate"),
                "lyrics_found":     entry.get("lyrics_found", False),
            })

            for mood in entry.get("moods", []):
                session.run(ADD_MOOD, {"persistent_id": pid, "mood": mood})

            for topic in entry.get("topics", []):
                session.run(ADD_TOPIC, {"persistent_id": pid, "topic": topic})

            for place in entry.get("places", []):
                session.run(ADD_PLACE, {"persistent_id": pid, "place": place})

            if on_progress:
                on_progress(i, total)

    driver.close()
    return total - skipped, skipped


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import os
    from dotenv import load_dotenv

    load_dotenv()

    neo4j_uri      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
    neo4j_user     = os.getenv("NEO4J_USER",     "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "")
    library_path   = Path("Data/library.json")
    themes_path    = Path("Data/track_themes.json")

    with open(library_path, encoding="utf-8") as f:
        library = json.load(f)
    with open(themes_path, encoding="utf-8") as f:
        themes = json.load(f)

    def on_progress(current, total):
        if current % 200 == 0:
            print(f"  {current}/{total} tracks processed ...")

    enriched, skipped = ingest(library, themes, neo4j_uri, neo4j_user, neo4j_password, on_progress=on_progress)
    print(f"\nDone. {enriched} tracks enriched, {skipped} skipped (no persistent_id match).")


if __name__ == "__main__":
    main()
