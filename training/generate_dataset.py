"""
training/generate_dataset.py

Generates the Cypher fine-tuning dataset for the TuneMap Apple Music KG.

Each example is one (system, user, assistant) chat turn where:
  - system  = full schema + Cypher rules (same prompt used at inference time)
  - user    = natural language question
  - assistant = correct Cypher query (no markdown, no explanation)

Output: training/data/cypher_dataset.jsonl  (one JSON object per line)

Usage:
    python training/generate_dataset.py [--output training/data/cypher_dataset.jsonl]
"""

import argparse
import json
import random
from pathlib import Path

# ── System prompt (mirrors inference-time prompt in query_engine.py) ──────────

SYSTEM_PROMPT = """\
You are a Cypher query expert for a Neo4j Knowledge Graph built from an Apple Music library.

## Graph Schema

### Node types and their properties:
- (:Track)    — name, year, release_date, duration_ms, play_count, skip_count, loved (bool), explicit (bool), date_added, track_number,
                lyrics_found (bool), language (ISO 639-1 code e.g. "en", "ru", "fr"),
                total_words (int), unique_words (int), type_token_ratio (float), repetition_rate (float)
- (:Artist)   — name
- (:Album)    — title, year
- (:Single)   — name
- (:Genre)    — name
- (:Era)      — name  (values: "Pre-90s", "90s", "2000s", "2010s", "2020s")
- (:Playlist) — name
- (:Mood)     — name  (values: "melancholic", "aggressive", "uplifting", "introspective", "boastful", "celebratory", "dark", "nostalgic", "anxious", "romantic", "defiant", "vulnerable", "cynical")
- (:Topic)    — name  (values: "god", "spirituality", "mother", "father", "death", "love", "sex", "desire", "friendship", "loyalty", "betrayal", "money", "fame", "freedom", "struggle", "fun", "heartbreak", "resentment", "redemption", "ego", "nostalgia", "dreams", "jealousy", "isolation", "insecurity", "mental health", "materialism", "pleasure", "moving on", "gang", "drugs", "prison", "police", "hustle", "government", "rebellion", "anger", "war", "identity", "alienation", "toxic relationship", "confidence", "youth", "passion", "technology", "escapism", "hometown", "hardship", "nature", "improvisation")
- (:Place)    — name  (lowercase city/country/region e.g. "new york", "toronto", "los angeles")

### Relationships:
- (:Track)-[:BY]------------->(:Artist)     primary artist of the track
- (:Track)-[:FEATURES]------->(:Artist)     featured/collaborating artist
- (:Track)-[:ON]------------->(:Album)      track belongs to this album
- (:Track)-[:IS_SINGLE]------>(:Single)     track is a standalone single
- (:Track)-[:IN_GENRE]------->(:Genre)      genre of the track
- (:Track)-[:IN_ERA]--------->(:Era)        decade era based on release year
- (:Track)-[:IN_PLAYLIST]---->(:Playlist)   track is in this user playlist
- (:Track)-[:HAS_MOOD]------->(:Mood)       mood(s) expressed in the lyrics
- (:Track)-[:HAS_TOPIC]------>(:Topic)      topic(s) referenced in the lyrics
- (:Track)-[:MENTIONS_PLACE]->(:Place)      real-world place mentioned in the lyrics
- (:Album)-[:BY]------------->(:Artist)     album's primary artist
- (:Artist)-[:IN_GENRE {track_count}]-->(:Genre)  derived genre affiliation (weighted by track count)

### Important notes:
- Artist identity is based on the Album Artist field (canonical). The Artist field may contain featured artists.
- Album titles have been normalised — edition suffixes (Deluxe, Remaster, etc.) are stripped.
- Genre has been normalised — all Hip-Hop variants map to "Hip-Hop/Rap".
- A track either has an [:ON]->(:Album) edge OR an [:IS_SINGLE]->(:Single) edge, never both.
- play_count and skip_count are integers (0 if never played/skipped).
- loved is a boolean — true means the user marked the track as a favourite.
- Mood, Topic and Place nodes only exist for tracks where lyrics_found = true (~65% of library).
- Topic and Mood names are always lowercase in the graph.
- type_token_ratio measures vocabulary richness (0–1): higher = more varied vocabulary.
- repetition_rate measures chorus/hook density (0–1): higher = more repeated lines.

### Cypher syntax rules (important — do not violate):
- This is Neo4j Cypher, NOT SQL. Never use SELECT, FROM, WHERE as top-level clauses. Use MATCH, RETURN, WITH, WHERE.
- String values that contain double quotes MUST use single-quote delimiters.
- Cypher does NOT have GROUP BY. Aggregation is implicit: any non-aggregated variable in RETURN acts as the grouping key automatically. Never write GROUP BY.
- Use WITH to pass variables between clauses and to filter/order before a subsequent MATCH.
- LIMIT always goes at the end after ORDER BY.
- NEVER produce a cartesian product. Every MATCH clause must share at least one variable with another clause.

### Hints for taste/summary questions:
- To understand overall taste, use the WHOLE library (all tracks), not just playlist tracks.
- play_count is the strongest signal for preference — weight by it wherever possible.
- loved = true tracks are explicit favourites — always worth highlighting separately.

Now generate a Cypher query for the following question. Return ONLY the Cypher query, no explanation.\
"""

# ── Hand-curated (question, cypher) pairs ─────────────────────────────────────
# Covers every node type, every relationship, aggregation, filtering,
# multi-hop traversals, and graph-return (viz) patterns.

EXAMPLES: list[tuple[str, str]] = [

    # ── Basic track lookups ───────────────────────────────────────────────────

    (
        "What are my top 10 most played tracks?",
        """\
MATCH (t:Track) WHERE t.play_count > 0
RETURN t.name AS track, t.play_count AS plays
ORDER BY plays DESC LIMIT 10""",
    ),
    (
        "What are my 20 most played songs?",
        """\
MATCH (t:Track) WHERE t.play_count > 0
RETURN t.name AS track, t.play_count AS plays
ORDER BY plays DESC LIMIT 20""",
    ),
    (
        "Which tracks have I listened to the most?",
        """\
MATCH (t:Track)-[:BY]->(a:Artist)
WHERE t.play_count > 0
RETURN t.name AS track, a.name AS artist, t.play_count AS plays
ORDER BY plays DESC LIMIT 15""",
    ),
    (
        "What are my most skipped tracks?",
        """\
MATCH (t:Track)-[:BY]->(a:Artist)
WHERE t.skip_count > 0
RETURN t.name AS track, a.name AS artist, t.skip_count AS skips
ORDER BY skips DESC LIMIT 15""",
    ),
    (
        "Which songs have I never skipped?",
        """\
MATCH (t:Track)-[:BY]->(a:Artist)
WHERE t.skip_count = 0 AND t.play_count > 0
RETURN t.name AS track, a.name AS artist, t.play_count AS plays
ORDER BY plays DESC LIMIT 20""",
    ),
    (
        "Which tracks have I marked as loved?",
        """\
MATCH (t:Track)-[:BY]->(a:Artist)
WHERE t.loved = true
RETURN t.name AS track, a.name AS artist, t.play_count AS plays
ORDER BY plays DESC""",
    ),
    (
        "Show me all my loved songs sorted by play count",
        """\
MATCH (t:Track)-[:BY]->(a:Artist)
WHERE t.loved = true
RETURN t.name AS track, a.name AS artist, t.play_count AS plays
ORDER BY plays DESC""",
    ),
    (
        "What are my most loved explicit tracks?",
        """\
MATCH (t:Track)-[:BY]->(a:Artist)
WHERE t.loved = true AND t.explicit = true
RETURN t.name AS track, a.name AS artist, t.play_count AS plays
ORDER BY plays DESC LIMIT 20""",
    ),
    (
        "How many tracks are in my library?",
        """\
MATCH (t:Track)
RETURN count(t) AS total_tracks""",
    ),
    (
        "How many unique artists are in my library?",
        """\
MATCH (a:Artist)
RETURN count(a) AS total_artists""",
    ),
    (
        "What is the total play time of my library in hours?",
        """\
MATCH (t:Track)
RETURN round(sum(t.duration_ms) / 3600000.0, 2) AS total_hours""",
    ),
    (
        "What are my shortest tracks?",
        """\
MATCH (t:Track)-[:BY]->(a:Artist)
WHERE t.duration_ms > 0
RETURN t.name AS track, a.name AS artist, t.duration_ms / 1000 AS duration_seconds
ORDER BY duration_seconds ASC LIMIT 15""",
    ),
    (
        "What are my longest tracks?",
        """\
MATCH (t:Track)-[:BY]->(a:Artist)
WHERE t.duration_ms > 0
RETURN t.name AS track, a.name AS artist, t.duration_ms / 60000 AS duration_minutes
ORDER BY duration_minutes DESC LIMIT 10""",
    ),
    (
        "Which tracks were added most recently?",
        """\
MATCH (t:Track)-[:BY]->(a:Artist)
WHERE t.date_added IS NOT NULL
RETURN t.name AS track, a.name AS artist, t.date_added
ORDER BY t.date_added DESC LIMIT 20""",
    ),
    (
        "What are my oldest tracks by release year?",
        """\
MATCH (t:Track)-[:BY]->(a:Artist)
WHERE t.year IS NOT NULL
RETURN t.name AS track, a.name AS artist, t.year
ORDER BY t.year ASC LIMIT 15""",
    ),
    (
        "How many explicit tracks do I have?",
        """\
MATCH (t:Track)
WHERE t.explicit = true
RETURN count(t) AS explicit_count""",
    ),
    (
        "What percentage of my library is explicit?",
        """\
MATCH (t:Track)
WITH count(t) AS total, sum(CASE WHEN t.explicit = true THEN 1 ELSE 0 END) AS explicit_count
RETURN explicit_count, total, round(100.0 * explicit_count / total, 1) AS pct_explicit""",
    ),
    (
        "Which tracks have I never played?",
        """\
MATCH (t:Track)-[:BY]->(a:Artist)
WHERE t.play_count = 0
RETURN t.name AS track, a.name AS artist
LIMIT 30""",
    ),

    # ── Artist queries ────────────────────────────────────────────────────────

    (
        "Who are my top 10 artists by play count?",
        """\
MATCH (t:Track)-[:BY]->(a:Artist)
RETURN a.name AS artist, sum(t.play_count) AS total_plays, count(t) AS track_count
ORDER BY total_plays DESC LIMIT 10""",
    ),
    (
        "Who are my top 20 artists?",
        """\
MATCH (t:Track)-[:BY]->(a:Artist)
RETURN a.name AS artist, sum(t.play_count) AS total_plays, count(t) AS track_count
ORDER BY total_plays DESC LIMIT 20""",
    ),
    (
        "Which artists do I have the most tracks from?",
        """\
MATCH (t:Track)-[:BY]->(a:Artist)
RETURN a.name AS artist, count(t) AS track_count
ORDER BY track_count DESC LIMIT 15""",
    ),
    (
        "What genres does Kanye West span?",
        """\
MATCH (a:Artist {name: "Kanye West"})<-[:BY]-(t:Track)-[:IN_GENRE]->(g:Genre)
RETURN g.name AS genre, count(t) AS track_count
ORDER BY track_count DESC""",
    ),
    (
        "What genres does Drake span?",
        """\
MATCH (a:Artist {name: "Drake"})<-[:BY]-(t:Track)-[:IN_GENRE]->(g:Genre)
RETURN g.name AS genre, count(t) AS track_count
ORDER BY track_count DESC""",
    ),
    (
        "Which artists feature most frequently in my library?",
        """\
MATCH (t:Track)-[:FEATURES]->(a:Artist)
RETURN a.name AS featured_artist, count(t) AS appearances
ORDER BY appearances DESC LIMIT 15""",
    ),
    (
        "Who are the most featured artists in my library?",
        """\
MATCH (t:Track)-[:FEATURES]->(a:Artist)
RETURN a.name AS featured_artist, count(t) AS feature_count
ORDER BY feature_count DESC LIMIT 20""",
    ),
    (
        "Which artists appear both as a primary artist and as a featured guest?",
        """\
MATCH (a:Artist)<-[:BY]-(t1:Track)
MATCH (t2:Track)-[:FEATURES]->(a)
RETURN a.name AS artist, count(DISTINCT t1) AS own_tracks, count(DISTINCT t2) AS guest_appearances
ORDER BY guest_appearances DESC LIMIT 15""",
    ),
    (
        "How many tracks do I have from Frank Ocean?",
        """\
MATCH (t:Track)-[:BY]->(a:Artist {name: "Frank Ocean"})
RETURN count(t) AS track_count""",
    ),
    (
        "What are my most played songs by Kendrick Lamar?",
        """\
MATCH (t:Track)-[:BY]->(a:Artist {name: "Kendrick Lamar"})
WHERE t.play_count > 0
RETURN t.name AS track, t.play_count AS plays
ORDER BY plays DESC LIMIT 10""",
    ),
    (
        "Which artists collaborated with Drake on tracks in my library?",
        """\
MATCH (t:Track)-[:BY]->(a:Artist {name: "Drake"})
MATCH (t)-[:FEATURES]->(collab:Artist)
RETURN DISTINCT collab.name AS collaborator, count(t) AS collaborations
ORDER BY collaborations DESC""",
    ),
    (
        "Find artists who collaborated with Kanye West and what eras their own music comes from",
        """\
MATCH (a:Artist {name: "Kanye West"})<-[:BY]-(t1:Track)-[:FEATURES]->(collab:Artist)
MATCH (collab)<-[:BY]-(t2:Track)-[:IN_ERA]->(e:Era)
RETURN collab.name AS collaborator, collect(DISTINCT e.name) AS eras, count(DISTINCT t2) AS track_count
ORDER BY track_count DESC LIMIT 15""",
    ),
    (
        "Which artist has the highest average play count per track?",
        """\
MATCH (t:Track)-[:BY]->(a:Artist)
WITH a, avg(t.play_count) AS avg_plays, count(t) AS track_count
WHERE track_count >= 3
RETURN a.name AS artist, round(avg_plays, 1) AS avg_plays_per_track, track_count
ORDER BY avg_plays_per_track DESC LIMIT 15""",
    ),
    (
        "Which artists have the most loved tracks?",
        """\
MATCH (t:Track)-[:BY]->(a:Artist)
WHERE t.loved = true
RETURN a.name AS artist, count(t) AS loved_tracks
ORDER BY loved_tracks DESC LIMIT 15""",
    ),

    # ── Album queries ─────────────────────────────────────────────────────────

    (
        "What albums do I have from Kendrick Lamar?",
        """\
MATCH (al:Album)-[:BY]->(a:Artist {name: "Kendrick Lamar"})
RETURN al.title AS album, al.year AS year
ORDER BY al.year ASC""",
    ),
    (
        "Which albums have the most tracks in my library?",
        """\
MATCH (t:Track)-[:ON]->(al:Album)-[:BY]->(a:Artist)
RETURN al.title AS album, a.name AS artist, count(t) AS track_count
ORDER BY track_count DESC LIMIT 15""",
    ),
    (
        "What are the most played albums?",
        """\
MATCH (t:Track)-[:ON]->(al:Album)-[:BY]->(a:Artist)
RETURN al.title AS album, a.name AS artist, sum(t.play_count) AS total_plays
ORDER BY total_plays DESC LIMIT 15""",
    ),
    (
        "Which albums came out in the 2010s in my library?",
        """\
MATCH (al:Album)-[:BY]->(a:Artist)
WHERE al.year >= 2010 AND al.year < 2020
RETURN al.title AS album, a.name AS artist, al.year AS year
ORDER BY al.year DESC LIMIT 30""",
    ),
    (
        "How many albums do I have?",
        """\
MATCH (al:Album)
RETURN count(al) AS total_albums""",
    ),
    (
        "Which albums contain songs about both love and death?",
        """\
MATCH (al:Album)<-[:ON]-(t1:Track)-[:HAS_TOPIC]->(tp1:Topic {name: "love"})
MATCH (al)<-[:ON]-(t2:Track)-[:HAS_TOPIC]->(tp2:Topic {name: "death"})
MATCH (al)-[:BY]->(a:Artist)
RETURN al.title AS album, a.name AS artist
ORDER BY al.year DESC LIMIT 15""",
    ),
    (
        "What are the most recent albums added to my library?",
        """\
MATCH (t:Track)-[:ON]->(al:Album)-[:BY]->(a:Artist)
WITH al, a, max(t.date_added) AS last_added
ORDER BY last_added DESC LIMIT 20
RETURN al.title AS album, a.name AS artist, last_added""",
    ),
    (
        "Which album has the highest total play count?",
        """\
MATCH (t:Track)-[:ON]->(al:Album)-[:BY]->(a:Artist)
RETURN al.title AS album, a.name AS artist, sum(t.play_count) AS total_plays, count(t) AS tracks
ORDER BY total_plays DESC LIMIT 10""",
    ),

    # ── Genre queries ─────────────────────────────────────────────────────────

    (
        "What genres are in my library?",
        """\
MATCH (g:Genre)<-[:IN_GENRE]-(t:Track)
RETURN g.name AS genre, count(t) AS track_count
ORDER BY track_count DESC""",
    ),
    (
        "What are my top genres by play count?",
        """\
MATCH (t:Track)-[:IN_GENRE]->(g:Genre)
RETURN g.name AS genre, sum(t.play_count) AS total_plays, count(t) AS track_count
ORDER BY total_plays DESC""",
    ),
    (
        "Which genres dominate my library by track count?",
        """\
MATCH (t:Track)-[:IN_GENRE]->(g:Genre)
RETURN g.name AS genre, count(t) AS track_count
ORDER BY track_count DESC LIMIT 10""",
    ),
    (
        "Who are the top artists in Hip-Hop/Rap?",
        """\
MATCH (t:Track)-[:IN_GENRE]->(g:Genre {name: "Hip-Hop/Rap"})
MATCH (t)-[:BY]->(a:Artist)
RETURN a.name AS artist, sum(t.play_count) AS total_plays, count(t) AS track_count
ORDER BY total_plays DESC LIMIT 15""",
    ),
    (
        "What are my most played R&B tracks?",
        """\
MATCH (t:Track)-[:IN_GENRE]->(g:Genre {name: "R&B/Soul"})
MATCH (t)-[:BY]->(a:Artist)
RETURN t.name AS track, a.name AS artist, t.play_count AS plays
ORDER BY plays DESC LIMIT 20""",
    ),
    (
        "How many tracks do I have per genre?",
        """\
MATCH (t:Track)-[:IN_GENRE]->(g:Genre)
RETURN g.name AS genre, count(t) AS track_count
ORDER BY track_count DESC""",
    ),
    (
        "Which genre has the highest average play count?",
        """\
MATCH (t:Track)-[:IN_GENRE]->(g:Genre)
WITH g, avg(t.play_count) AS avg_plays, count(t) AS track_count
WHERE track_count >= 5
RETURN g.name AS genre, round(avg_plays, 1) AS avg_play_count, track_count
ORDER BY avg_plays DESC LIMIT 10""",
    ),
    (
        "Which artists span the most genres?",
        """\
MATCH (a:Artist)<-[:BY]-(t:Track)-[:IN_GENRE]->(g:Genre)
WITH a, count(DISTINCT g) AS genre_count, collect(DISTINCT g.name) AS genres
WHERE genre_count >= 2
RETURN a.name AS artist, genre_count, genres
ORDER BY genre_count DESC LIMIT 15""",
    ),

    # ── Era queries ───────────────────────────────────────────────────────────

    (
        "How many tracks do I have per era?",
        """\
MATCH (t:Track)-[:IN_ERA]->(e:Era)
RETURN e.name AS era, count(t) AS track_count
ORDER BY track_count DESC""",
    ),
    (
        "What are my most played tracks from the 90s?",
        """\
MATCH (t:Track)-[:IN_ERA]->(e:Era {name: "90s"})
MATCH (t)-[:BY]->(a:Artist)
RETURN t.name AS track, a.name AS artist, t.play_count AS plays
ORDER BY plays DESC LIMIT 15""",
    ),
    (
        "Which era dominates my library?",
        """\
MATCH (t:Track)-[:IN_ERA]->(e:Era)
RETURN e.name AS era, count(t) AS track_count, sum(t.play_count) AS total_plays
ORDER BY total_plays DESC""",
    ),
    (
        "What genres were most popular in the 2000s in my library?",
        """\
MATCH (t:Track)-[:IN_ERA]->(e:Era {name: "2000s"})
MATCH (t)-[:IN_GENRE]->(g:Genre)
RETURN g.name AS genre, count(t) AS track_count, sum(t.play_count) AS total_plays
ORDER BY total_plays DESC LIMIT 10""",
    ),
    (
        "Which artists defined the 2010s in my library?",
        """\
MATCH (t:Track)-[:IN_ERA]->(e:Era {name: "2010s"})
MATCH (t)-[:BY]->(a:Artist)
RETURN a.name AS artist, count(t) AS track_count, sum(t.play_count) AS total_plays
ORDER BY total_plays DESC LIMIT 15""",
    ),
    (
        "Which artists from the Pre-90s era do I listen to most?",
        """\
MATCH (t:Track)-[:IN_ERA]->(e:Era {name: "Pre-90s"})
MATCH (t)-[:BY]->(a:Artist)
RETURN a.name AS artist, sum(t.play_count) AS total_plays, count(t) AS track_count
ORDER BY total_plays DESC LIMIT 10""",
    ),
    (
        "How are my plays distributed across eras?",
        """\
MATCH (t:Track)-[:IN_ERA]->(e:Era)
RETURN e.name AS era, count(t) AS tracks, sum(t.play_count) AS total_plays
ORDER BY total_plays DESC""",
    ),

    # ── Playlist queries ──────────────────────────────────────────────────────

    (
        "What playlists do I have?",
        """\
MATCH (p:Playlist)
RETURN p.name AS playlist
ORDER BY p.name ASC""",
    ),
    (
        "Which playlist has the most tracks?",
        """\
MATCH (t:Track)-[:IN_PLAYLIST]->(p:Playlist)
RETURN p.name AS playlist, count(t) AS track_count
ORDER BY track_count DESC LIMIT 10""",
    ),
    (
        "What are the top artists in my workout playlist?",
        """\
MATCH (t:Track)-[:IN_PLAYLIST]->(p:Playlist)
WHERE toLower(p.name) CONTAINS 'workout'
MATCH (t)-[:BY]->(a:Artist)
RETURN a.name AS artist, count(t) AS track_count, sum(t.play_count) AS total_plays
ORDER BY total_plays DESC LIMIT 10""",
    ),
    (
        "How many tracks are in each playlist?",
        """\
MATCH (t:Track)-[:IN_PLAYLIST]->(p:Playlist)
RETURN p.name AS playlist, count(t) AS track_count
ORDER BY track_count DESC""",
    ),
    (
        "Which playlists contain the most songs about money?",
        """\
MATCH (t:Track)-[:HAS_TOPIC]->(tp:Topic {name: "money"})
MATCH (t)-[:IN_PLAYLIST]->(p:Playlist)
RETURN p.name AS playlist, count(t) AS money_tracks
ORDER BY money_tracks DESC LIMIT 10""",
    ),
    (
        "Which tracks appear in the most playlists?",
        """\
MATCH (t:Track)-[:IN_PLAYLIST]->(p:Playlist)
MATCH (t)-[:BY]->(a:Artist)
WITH t, a, count(DISTINCT p) AS playlist_count
RETURN t.name AS track, a.name AS artist, playlist_count
ORDER BY playlist_count DESC LIMIT 15""",
    ),

    # ── Lyrics / mood / topic queries ─────────────────────────────────────────

    (
        "What moods dominate my library?",
        """\
MATCH (t:Track)-[:HAS_MOOD]->(m:Mood)
RETURN m.name AS mood, count(t) AS track_count
ORDER BY track_count DESC""",
    ),
    (
        "What are the most common topics in my library?",
        """\
MATCH (t:Track)-[:HAS_TOPIC]->(tp:Topic)
RETURN tp.name AS topic, count(t) AS track_count
ORDER BY track_count DESC LIMIT 20""",
    ),
    (
        "Find me dark melancholic songs about death",
        """\
MATCH (t:Track)-[:HAS_MOOD]->(m1:Mood {name: "dark"})
MATCH (t)-[:HAS_MOOD]->(m2:Mood {name: "melancholic"})
MATCH (t)-[:HAS_TOPIC]->(tp:Topic {name: "death"})
MATCH (t)-[:BY]->(a:Artist)
RETURN t.name AS track, a.name AS artist
ORDER BY t.play_count DESC LIMIT 20""",
    ),
    (
        "Show me uplifting songs about friendship or love",
        """\
MATCH (t:Track)-[:HAS_MOOD]->(m:Mood {name: "uplifting"})
MATCH (t)-[:HAS_TOPIC]->(tp:Topic)
WHERE tp.name IN ["friendship", "love"]
MATCH (t)-[:BY]->(a:Artist)
WITH t, a, collect(DISTINCT tp.name) AS topics
RETURN t.name AS track, a.name AS artist, topics
ORDER BY t.play_count DESC LIMIT 20""",
    ),
    (
        "Which tracks are about ego and fame?",
        """\
MATCH (t:Track)-[:HAS_TOPIC]->(tp1:Topic {name: "ego"})
MATCH (t)-[:HAS_TOPIC]->(tp2:Topic {name: "fame"})
MATCH (t)-[:BY]->(a:Artist)
RETURN t.name AS track, a.name AS artist, t.play_count AS plays
ORDER BY plays DESC LIMIT 20""",
    ),
    (
        "Find aggressive songs about loyalty",
        """\
MATCH (t:Track)-[:HAS_MOOD]->(m:Mood {name: "aggressive"})
MATCH (t)-[:HAS_TOPIC]->(tp:Topic {name: "loyalty"})
MATCH (t)-[:BY]->(a:Artist)
RETURN t.name AS track, a.name AS artist, t.play_count AS plays
ORDER BY plays DESC LIMIT 15""",
    ),
    (
        "What moods dominate my Hip-Hop tracks?",
        """\
MATCH (t:Track)-[:IN_GENRE]->(g:Genre {name: "Hip-Hop/Rap"})
MATCH (t)-[:HAS_MOOD]->(m:Mood)
RETURN m.name AS mood, count(t) AS track_count
ORDER BY track_count DESC""",
    ),
    (
        "Which artists mention New York most in their lyrics?",
        """\
MATCH (t:Track)-[:MENTIONS_PLACE]->(pl:Place {name: "new york"})
MATCH (t)-[:BY]->(a:Artist)
RETURN a.name AS artist, count(t) AS track_count
ORDER BY track_count DESC LIMIT 10""",
    ),
    (
        "What places are mentioned most in my library?",
        """\
MATCH (t:Track)-[:MENTIONS_PLACE]->(pl:Place)
RETURN pl.name AS place, count(t) AS track_count
ORDER BY track_count DESC LIMIT 20""",
    ),
    (
        "Which artists mention Los Angeles in their lyrics?",
        """\
MATCH (t:Track)-[:MENTIONS_PLACE]->(pl:Place {name: "los angeles"})
MATCH (t)-[:BY]->(a:Artist)
RETURN a.name AS artist, count(t) AS mentions, collect(DISTINCT t.name)[..5] AS sample_tracks
ORDER BY mentions DESC LIMIT 15""",
    ),
    (
        "What languages are in my library?",
        """\
MATCH (t:Track)
WHERE t.language IS NOT NULL
RETURN t.language AS language, count(t) AS track_count
ORDER BY track_count DESC""",
    ),
    (
        "How many tracks have lyrics data?",
        """\
MATCH (t:Track)
WHERE t.lyrics_found = true
RETURN count(t) AS tracks_with_lyrics""",
    ),
    (
        "Which tracks have the most varied vocabulary?",
        """\
MATCH (t:Track)-[:BY]->(a:Artist)
WHERE t.type_token_ratio IS NOT NULL
RETURN t.name AS track, a.name AS artist, t.type_token_ratio AS ttr
ORDER BY ttr DESC LIMIT 10""",
    ),
    (
        "Which tracks are the most repetitive lyrically?",
        """\
MATCH (t:Track)-[:BY]->(a:Artist)
WHERE t.repetition_rate IS NOT NULL
RETURN t.name AS track, a.name AS artist, t.repetition_rate AS repetition
ORDER BY repetition DESC LIMIT 10""",
    ),
    (
        "Which artists have the most lyric-rich vocabulary on average?",
        """\
MATCH (t:Track)-[:BY]->(a:Artist)
WHERE t.type_token_ratio IS NOT NULL
WITH a, avg(t.type_token_ratio) AS avg_ttr, count(t) AS track_count
WHERE track_count >= 3
RETURN a.name AS artist, round(avg_ttr, 3) AS avg_vocabulary_richness, track_count
ORDER BY avg_ttr DESC LIMIT 15""",
    ),
    (
        "Find introspective tracks about mental health or isolation",
        """\
MATCH (t:Track)-[:HAS_MOOD]->(m:Mood {name: "introspective"})
MATCH (t)-[:HAS_TOPIC]->(tp:Topic)
WHERE tp.name IN ["mental health", "isolation"]
MATCH (t)-[:BY]->(a:Artist)
WITH t, a, collect(DISTINCT tp.name) AS topics
RETURN t.name AS track, a.name AS artist, topics
ORDER BY t.play_count DESC LIMIT 20""",
    ),
    (
        "Show me nostalgic tracks from the 90s era",
        """\
MATCH (t:Track)-[:HAS_MOOD]->(m:Mood {name: "nostalgic"})
MATCH (t)-[:IN_ERA]->(e:Era {name: "90s"})
MATCH (t)-[:BY]->(a:Artist)
RETURN t.name AS track, a.name AS artist, t.play_count AS plays
ORDER BY plays DESC LIMIT 20""",
    ),
    (
        "Which R&B tracks are most romantic?",
        """\
MATCH (t:Track)-[:IN_GENRE]->(g:Genre {name: "R&B/Soul"})
MATCH (t)-[:HAS_MOOD]->(m:Mood {name: "romantic"})
MATCH (t)-[:BY]->(a:Artist)
RETURN t.name AS track, a.name AS artist, t.play_count AS plays
ORDER BY plays DESC LIMIT 15""",
    ),
    (
        "Find boastful tracks about money and ego",
        """\
MATCH (t:Track)-[:HAS_MOOD]->(m:Mood {name: "boastful"})
MATCH (t)-[:HAS_TOPIC]->(tp:Topic)
WHERE tp.name IN ["money", "ego"]
MATCH (t)-[:BY]->(a:Artist)
RETURN t.name AS track, a.name AS artist, collect(DISTINCT tp.name) AS topics, t.play_count AS plays
ORDER BY plays DESC LIMIT 15""",
    ),
    (
        "Which tracks mention both Toronto and New York?",
        """\
MATCH (t:Track)-[:MENTIONS_PLACE]->(p1:Place {name: "toronto"})
MATCH (t)-[:MENTIONS_PLACE]->(p2:Place {name: "new york"})
MATCH (t)-[:BY]->(a:Artist)
RETURN t.name AS track, a.name AS artist, t.play_count AS plays
ORDER BY plays DESC LIMIT 15""",
    ),

    # ── Multi-hop / cross-entity queries ──────────────────────────────────────

    (
        "Find artists connected through shared collaborators",
        """\
MATCH (a1:Artist)<-[:BY]-(t1:Track)-[:FEATURES]->(bridge:Artist)<-[:FEATURES]-(t2:Track)-[:BY]->(a2:Artist)
WHERE a1 <> a2 AND a1.name < a2.name
RETURN a1.name AS artist1, bridge.name AS shared_collaborator, a2.name AS artist2, count(*) AS strength
ORDER BY strength DESC LIMIT 15""",
    ),
    (
        "Which artists are connected through shared topics in their lyrics?",
        """\
MATCH (a1:Artist)<-[:BY]-(t1:Track)-[:HAS_TOPIC]->(tp:Topic)<-[:HAS_TOPIC]-(t2:Track)-[:BY]->(a2:Artist)
WHERE a1 <> a2 AND a1.name < a2.name
RETURN a1.name, tp.name AS shared_topic, a2.name, count(*) AS strength
ORDER BY strength DESC LIMIT 15""",
    ),
    (
        "What topics connect artists across different genres?",
        """\
MATCH (a1:Artist)<-[:BY]-(t1:Track)-[:IN_GENRE]->(g1:Genre)
MATCH (t1)-[:HAS_TOPIC]->(tp:Topic)<-[:HAS_TOPIC]-(t2:Track)-[:IN_GENRE]->(g2:Genre)
MATCH (t2)-[:BY]->(a2:Artist)
WHERE g1 <> g2 AND a1 <> a2
RETURN tp.name AS topic, g1.name AS genre1, a1.name AS artist1, g2.name AS genre2, a2.name AS artist2
ORDER BY t1.play_count DESC LIMIT 20""",
    ),
    (
        "Find English and Russian tracks that share a topic",
        """\
MATCH (t1:Track {language: 'en'})-[:HAS_TOPIC]->(tp:Topic)<-[:HAS_TOPIC]-(t2:Track {language: 'ru'})
MATCH (t1)-[:BY]->(a1:Artist), (t2)-[:BY]->(a2:Artist)
RETURN a1.name AS english_artist, t1.name AS english_track,
       tp.name AS shared_topic,
       t2.name AS russian_track, a2.name AS russian_artist
ORDER BY t1.play_count DESC LIMIT 20""",
    ),
    (
        "Which tracks in my library share both a mood and a topic with a loved track?",
        """\
MATCH (loved:Track)-[:HAS_MOOD]->(m:Mood)
WHERE loved.loved = true
MATCH (loved)-[:HAS_TOPIC]->(tp:Topic)
MATCH (other:Track)-[:HAS_MOOD]->(m)
MATCH (other)-[:HAS_TOPIC]->(tp)
WHERE other <> loved AND other.loved = false
MATCH (other)-[:BY]->(a:Artist)
WITH other, a,
     collect(DISTINCT m.name)[..3] AS shared_moods,
     collect(DISTINCT tp.name)[..3] AS shared_topics
RETURN other.name AS track, a.name AS artist,
       shared_moods, shared_topics
ORDER BY other.play_count DESC LIMIT 20""",
    ),
    (
        "Find artists who collaborated with Drake and what moods their own songs have",
        """\
MATCH (drake:Artist {name: "Drake"})<-[:BY]-(t1:Track)-[:FEATURES]->(collab:Artist)
MATCH (collab)<-[:BY]-(t2:Track)-[:HAS_MOOD]->(m:Mood)
RETURN collab.name AS collaborator, collect(DISTINCT m.name) AS moods, count(DISTINCT t2) AS track_count
ORDER BY track_count DESC LIMIT 15""",
    ),
    (
        "Which genres share the most topics across their tracks?",
        """\
MATCH (g1:Genre)<-[:IN_GENRE]-(t1:Track)-[:HAS_TOPIC]->(tp:Topic)<-[:HAS_TOPIC]-(t2:Track)-[:IN_GENRE]->(g2:Genre)
WHERE g1 <> g2 AND g1.name < g2.name
RETURN g1.name AS genre1, g2.name AS genre2, count(DISTINCT tp) AS shared_topics
ORDER BY shared_topics DESC LIMIT 15""",
    ),
    (
        "What is the mood breakdown for tracks from each era?",
        """\
MATCH (t:Track)-[:IN_ERA]->(e:Era)
MATCH (t)-[:HAS_MOOD]->(m:Mood)
RETURN e.name AS era, m.name AS mood, count(t) AS track_count
ORDER BY era ASC, track_count DESC""",
    ),
    (
        "Which playlists overlap the most in their track selections?",
        """\
MATCH (t:Track)-[:IN_PLAYLIST]->(p1:Playlist)
MATCH (t)-[:IN_PLAYLIST]->(p2:Playlist)
WHERE p1 <> p2 AND p1.name < p2.name
RETURN p1.name AS playlist1, p2.name AS playlist2, count(DISTINCT t) AS shared_tracks
ORDER BY shared_tracks DESC LIMIT 15""",
    ),

    # ── Taste summary queries ─────────────────────────────────────────────────

    (
        "Summarise my music taste",
        """\
MATCH (t:Track)-[:IN_GENRE]->(g:Genre)
WITH g.name AS genre, sum(t.play_count) AS total_plays, count(t) AS track_count
ORDER BY total_plays DESC LIMIT 10
RETURN genre, total_plays, track_count""",
    ),
    (
        "What does my Hip-Hop listening say about me?",
        """\
MATCH (t:Track)-[:IN_GENRE]->(g:Genre {name: "Hip-Hop/Rap"})
WITH count(t) AS hiphop_tracks, sum(t.play_count) AS total_plays
RETURN hiphop_tracks, total_plays""",
    ),
    (
        "What are the top 5 genres and top 3 artists per genre?",
        """\
MATCH (t:Track)-[:IN_GENRE]->(g:Genre)
WITH g, sum(t.play_count) AS genre_plays ORDER BY genre_plays DESC LIMIT 5
WITH collect(g.name) AS top_genres
MATCH (t:Track)-[:IN_GENRE]->(g:Genre)-[:IN_GENRE]-(a:Artist)
WHERE g.name IN top_genres
RETURN g.name AS genre, collect(DISTINCT a.name)[..3] AS top_artists
ORDER BY genre""",
    ),

    # ── Standalone singles ────────────────────────────────────────────────────

    (
        "Which of my tracks are standalone singles?",
        """\
MATCH (t:Track)-[:IS_SINGLE]->(s:Single)
MATCH (t)-[:BY]->(a:Artist)
RETURN t.name AS track, a.name AS artist, t.play_count AS plays
ORDER BY plays DESC LIMIT 30""",
    ),
    (
        "How many standalone singles are in my library?",
        """\
MATCH (t:Track)-[:IS_SINGLE]->(s:Single)
RETURN count(t) AS single_count""",
    ),
    (
        "Who has the most singles in my library?",
        """\
MATCH (t:Track)-[:IS_SINGLE]->(s:Single)
MATCH (t)-[:BY]->(a:Artist)
RETURN a.name AS artist, count(t) AS singles_count
ORDER BY singles_count DESC LIMIT 15""",
    ),

    # ── Artist genre affiliation ──────────────────────────────────────────────

    (
        "What are the primary genres for my top artists?",
        """\
MATCH (t:Track)-[:BY]->(a:Artist)
WITH a, sum(t.play_count) AS total_plays ORDER BY total_plays DESC LIMIT 10
WITH collect(a.name) AS top_artists
MATCH (a:Artist)-[r:IN_GENRE]->(g:Genre)
WHERE a.name IN top_artists
RETURN a.name AS artist, g.name AS genre, r.track_count AS track_count
ORDER BY a.name, r.track_count DESC""",
    ),
    (
        "Which genre has the most genre-affiliated artists?",
        """\
MATCH (a:Artist)-[:IN_GENRE]->(g:Genre)
RETURN g.name AS genre, count(DISTINCT a) AS artist_count
ORDER BY artist_count DESC LIMIT 10""",
    ),

    # ── Skip / engagement ratio queries ──────────────────────────────────────

    (
        "Which tracks have the highest skip rate relative to plays?",
        """\
MATCH (t:Track)-[:BY]->(a:Artist)
WHERE t.play_count > 0
WITH t, a, toFloat(t.skip_count) / t.play_count AS skip_ratio
WHERE skip_ratio > 0
RETURN t.name AS track, a.name AS artist, t.play_count AS plays, t.skip_count AS skips, round(skip_ratio, 2) AS skip_ratio
ORDER BY skip_ratio DESC LIMIT 15""",
    ),
    (
        "Which artists have tracks I mostly skip?",
        """\
MATCH (t:Track)-[:BY]->(a:Artist)
WHERE t.play_count > 0
WITH a, sum(t.skip_count) AS total_skips, sum(t.play_count) AS total_plays
WHERE total_plays > 0
RETURN a.name AS artist, total_plays, total_skips, round(toFloat(total_skips)/total_plays, 2) AS skip_ratio
ORDER BY skip_ratio DESC LIMIT 15""",
    ),

    # ── Year / release date queries ───────────────────────────────────────────

    (
        "What years are best represented in my library?",
        """\
MATCH (t:Track)
WHERE t.year IS NOT NULL
RETURN t.year AS year, count(t) AS track_count
ORDER BY track_count DESC LIMIT 20""",
    ),
    (
        "How many tracks were released each decade?",
        """\
MATCH (t:Track)-[:IN_ERA]->(e:Era)
RETURN e.name AS era, count(t) AS track_count, sum(t.play_count) AS total_plays
ORDER BY e.name ASC""",
    ),
    (
        "Which years have I played music from the most?",
        """\
MATCH (t:Track)
WHERE t.year IS NOT NULL
RETURN t.year AS year, sum(t.play_count) AS total_plays, count(t) AS track_count
ORDER BY total_plays DESC LIMIT 20""",
    ),

    # ── Viz-friendly graph return queries (for render_graph / neovis.js) ──────
    # These RETURN only node/relationship variables — no scalars.

    (
        "Graph my top 10 artists and the genres they belong to",
        """\
MATCH (t:Track)-[:BY]->(a:Artist)
WITH a, sum(t.play_count) AS total_plays ORDER BY total_plays DESC LIMIT 10
WITH collect(a.name) AS top_artists
MATCH (a:Artist)<-[r1:BY]-(t:Track)-[r2:IN_GENRE]->(g:Genre) WHERE a.name IN top_artists
WITH a, r1, t, r2, g ORDER BY t.play_count DESC
WITH a, collect({r1:r1, t:t, r2:r2, g:g})[..3] AS rows
UNWIND rows AS row
WITH a, row.r1 AS r1, row.t AS t, row.r2 AS r2, row.g AS g
RETURN a, r1, t, r2, g""",
    ),
    (
        "Graph my top 10 most played tracks and the artists behind them",
        """\
MATCH (t:Track)-[r:BY]->(a:Artist)
WHERE t.play_count > 0
WITH t, r, a ORDER BY t.play_count DESC LIMIT 10
RETURN t, r, a""",
    ),
    (
        "Graph which artists are featured on my top 5 artists tracks",
        """\
MATCH (t:Track)-[:BY]->(a:Artist)
WITH a, sum(t.play_count) AS total_plays ORDER BY total_plays DESC LIMIT 5
WITH collect(a.name) AS top_artists
MATCH (t:Track)-[rb:BY]->(a:Artist) WHERE a.name IN top_artists
MATCH (t)-[rf:FEATURES]->(f:Artist)
WITH a, rb, t, rf, f ORDER BY t.play_count DESC
WITH a, collect({rb:rb, t:t, rf:rf, f:f})[..4] AS rows
UNWIND rows AS row
RETURN a, row.rb AS rb, row.t AS t, row.rf AS rf, row.f AS f""",
    ),
    (
        "Graph my genres and the top 3 artists in each",
        """\
MATCH (t:Track)-[:IN_GENRE]->(g:Genre)
WITH g, sum(t.play_count) AS gplays ORDER BY gplays DESC LIMIT 8
WITH collect(g.name) AS top_genres
MATCH (t:Track)-[rg:IN_GENRE]->(g:Genre)-[ra:IN_GENRE]-(a:Artist) WHERE g.name IN top_genres
RETURN g, rg, a""",
    ),
    (
        "Graph my top artists and the eras their music comes from",
        """\
MATCH (t:Track)-[:BY]->(a:Artist)
WITH a, sum(t.play_count) AS plays ORDER BY plays DESC LIMIT 10
WITH collect(a.name) AS top_names
MATCH (a:Artist)<-[rb:BY]-(t:Track)-[re:IN_ERA]->(e:Era) WHERE a.name IN top_names
WITH a, rb, t, re, e ORDER BY t.play_count DESC
WITH a, collect({rb:rb, t:t, re:re, e:e})[..3] AS rows
UNWIND rows AS row
RETURN a, row.rb AS rb, row.t AS t, row.re AS re, row.e AS e""",
    ),
    (
        "Graph the mood and topic landscape of my most played tracks",
        """\
MATCH (t:Track)
WHERE t.play_count > 0 AND t.lyrics_found = true
WITH t ORDER BY t.play_count DESC LIMIT 15
WITH collect(t.name) AS top_tracks
MATCH (t:Track) WHERE t.name IN top_tracks
MATCH (t)-[:BY]->(a:Artist)
OPTIONAL MATCH (t)-[rm:HAS_MOOD]->(m:Mood)
OPTIONAL MATCH (t)-[rtp:HAS_TOPIC]->(tp:Topic)
RETURN t, a, rm, m, rtp, tp""",
    ),
    (
        "Graph artist collaboration network for my top artists",
        """\
MATCH (t:Track)-[:BY]->(a:Artist)
WITH a, sum(t.play_count) AS plays ORDER BY plays DESC LIMIT 8
WITH collect(a.name) AS top_names
MATCH (t:Track)-[rb:BY]->(a:Artist) WHERE a.name IN top_names
MATCH (t)-[rf:FEATURES]->(f:Artist)
RETURN a, rb, t, rf, f""",
    ),
    (
        "Graph all moods and their most played tracks",
        """\
MATCH (t:Track)-[rm:HAS_MOOD]->(m:Mood)
WITH m, t, rm ORDER BY t.play_count DESC
WITH m, collect({t:t, rm:rm})[..5] AS rows
UNWIND rows AS row
WITH m, row.rm AS rm, row.t AS t
MATCH (t)-[rb:BY]->(a:Artist)
RETURN m, rm, t, rb, a""",
    ),

    # ── Miscellaneous / tricky queries ────────────────────────────────────────

    (
        "Which tracks have been loved but are also highly skipped?",
        """\
MATCH (t:Track)-[:BY]->(a:Artist)
WHERE t.loved = true AND t.skip_count > 0
RETURN t.name AS track, a.name AS artist, t.play_count AS plays, t.skip_count AS skips
ORDER BY t.skip_count DESC LIMIT 15""",
    ),
    (
        "Which artists in my library released music in the most different years?",
        """\
MATCH (t:Track)-[:BY]->(a:Artist)
WHERE t.year IS NOT NULL
RETURN a.name AS artist, count(DISTINCT t.year) AS year_span, min(t.year) AS earliest, max(t.year) AS latest
ORDER BY year_span DESC LIMIT 15""",
    ),
    (
        "What is my most listened-to song of all time?",
        """\
MATCH (t:Track)-[:BY]->(a:Artist)
RETURN t.name AS track, a.name AS artist, t.play_count AS plays
ORDER BY plays DESC LIMIT 1""",
    ),
    (
        "Find tracks that share a genre, mood, and topic with my most played track",
        """\
MATCH (top:Track)
WITH top ORDER BY top.play_count DESC LIMIT 1
MATCH (top)-[:IN_GENRE]->(g:Genre)
MATCH (top)-[:HAS_MOOD]->(m:Mood)
MATCH (top)-[:HAS_TOPIC]->(tp:Topic)
MATCH (other:Track)-[:IN_GENRE]->(g)
MATCH (other)-[:HAS_MOOD]->(m)
MATCH (other)-[:HAS_TOPIC]->(tp)
WHERE other <> top
MATCH (other)-[:BY]->(a:Artist)
RETURN DISTINCT other.name AS track, a.name AS artist, other.play_count AS plays
ORDER BY plays DESC LIMIT 15""",
    ),
    (
        "How many tracks have lyrics data broken down by genre?",
        """\
MATCH (t:Track)-[:IN_GENRE]->(g:Genre)
WHERE t.lyrics_found = true
RETURN g.name AS genre, count(t) AS tracks_with_lyrics
ORDER BY tracks_with_lyrics DESC""",
    ),
    (
        "Which of my non-English tracks are the most played?",
        """\
MATCH (t:Track)-[:BY]->(a:Artist)
WHERE t.language IS NOT NULL AND t.language <> 'en' AND t.play_count > 0
RETURN t.name AS track, a.name AS artist, t.language AS language, t.play_count AS plays
ORDER BY plays DESC LIMIT 20""",
    ),
    (
        "What are the top 5 most common co-occurring mood pairs in my library?",
        """\
MATCH (t:Track)-[:HAS_MOOD]->(m1:Mood)
MATCH (t)-[:HAS_MOOD]->(m2:Mood)
WHERE m1.name < m2.name
RETURN m1.name AS mood1, m2.name AS mood2, count(t) AS track_count
ORDER BY track_count DESC LIMIT 10""",
    ),
    (
        "Which artists produce the most lyrically repetitive music?",
        """\
MATCH (t:Track)-[:BY]->(a:Artist)
WHERE t.repetition_rate IS NOT NULL
WITH a, avg(t.repetition_rate) AS avg_rep, count(t) AS track_count
WHERE track_count >= 3
RETURN a.name AS artist, round(avg_rep, 3) AS avg_repetition_rate, track_count
ORDER BY avg_rep DESC LIMIT 15""",
    ),
    (
        "Which years saw the most tracks added to my library?",
        """\
MATCH (t:Track)
WHERE t.date_added IS NOT NULL
RETURN substring(t.date_added, 0, 4) AS year_added, count(t) AS tracks_added
ORDER BY year_added DESC LIMIT 10""",
    ),
]


# ── Format helpers ─────────────────────────────────────────────────────────────

def format_example(question: str, cypher: str) -> dict:
    """Format one (question, cypher) pair as a chat-format training example."""
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": question},
            {"role": "assistant", "content": cypher},
        ]
    }


def build_dataset(shuffle: bool = True, seed: int = 42) -> list[dict]:
    examples = [format_example(q, c) for q, c in EXAMPLES]
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(examples)
    return examples


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate Cypher fine-tuning dataset")
    parser.add_argument(
        "--output",
        default="training/data/cypher_dataset.jsonl",
        help="Output JSONL path (default: training/data/cypher_dataset.jsonl)",
    )
    parser.add_argument("--no-shuffle", action="store_true", help="Keep examples in original order")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(shuffle=not args.no_shuffle)

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in dataset:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Written {len(dataset)} examples → {output_path}")

    # Print a sample
    sample = dataset[0]
    print("\n── Sample example ──────────────────────────────────────────────")
    print(f"User:      {sample['messages'][1]['content']}")
    print(f"Assistant: {sample['messages'][2]['content'][:200]}...")


if __name__ == "__main__":
    main()
