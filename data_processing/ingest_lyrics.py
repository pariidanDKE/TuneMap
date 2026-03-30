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
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

LIBRARY_PATH = Path("Data/library.json")
THEMES_PATH  = Path("Data/track_themes.json")

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

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Build Track ID → persistent_id lookup from library.json
    with open(LIBRARY_PATH, encoding="utf-8") as f:
        library = json.load(f)
    id_to_pid = {
        str(t["Track ID"]): t["Persistent ID"]
        for t in library["tracks"]
        if "Track ID" in t and "Persistent ID" in t
    }

    with open(THEMES_PATH, encoding="utf-8") as f:
        themes = json.load(f)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        for c in CONSTRAINTS:
            session.run(c)
        print("Constraints applied.")

        total   = len(themes)
        skipped = 0

        for i, (track_id, entry) in enumerate(themes.items(), 1):
            pid = id_to_pid.get(track_id)
            if not pid:
                skipped += 1
                continue

            # Update lexical properties on the Track node
            session.run(UPDATE_TRACK_PROPS, {
                "persistent_id":   pid,
                "language":        entry.get("language"),
                "total_words":     entry.get("total_words"),
                "unique_words":    entry.get("unique_words"),
                "type_token_ratio": entry.get("type_token_ratio"),
                "repetition_rate": entry.get("repetition_rate"),
                "lyrics_found":    entry.get("lyrics_found", False),
            })

            # Mood relationships
            for mood in entry.get("moods", []):
                session.run(ADD_MOOD, {"persistent_id": pid, "mood": mood})

            # Topic relationships
            for topic in entry.get("topics", []):
                session.run(ADD_TOPIC, {"persistent_id": pid, "topic": topic})

            # Place relationships
            for place in entry.get("places", []):
                session.run(ADD_PLACE, {"persistent_id": pid, "place": place})

            if i % 200 == 0:
                print(f"  {i}/{total} tracks processed ...")

        print(f"\nDone. {total - skipped} tracks enriched, {skipped} skipped (no persistent_id match).")

    driver.close()


if __name__ == "__main__":
    main()
