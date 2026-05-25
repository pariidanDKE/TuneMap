"""
Build a brand-new validated Cypher dataset for TuneMap.

Outputs:
- training/data/cyper_validated_dataset.jsonl
- training/outputs/cyper_generation_journal.md
- training/outputs/cyper_discarded_pairs.jsonl

This script does NOT reuse training/generate_dataset.py examples.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

# Suppress server-side notification noise.
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

SYSTEM_PROMPT = """Task: Generate Cypher statement to query a graph database.
Instructions: Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided in the schema.
Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement."""

EVAL_PATH = Path("training/data/eval.jsonl")
OUT_PATH = Path("training/data/cyper_validated_dataset.jsonl")
JOURNAL_PATH = Path("training/outputs/cyper_generation_journal.md")
DISCARDED_PATH = Path("training/outputs/cyper_discarded_pairs.jsonl")

TARGET_ROWS = 150
MIN_VALIDATED = 140
RANDOM_SEED = 20260427
LIMITS = [5, 10, 15, 20, 25, 30]
ERAS = ["Pre-90s", "90s", "2000s", "2010s", "2020s"]


@dataclass
class Pair:
    question: str
    cypher: str
    category: str


@dataclass
class ValidationResult:
    pair: Pair
    valid: bool
    reason: str | None = None


def q(s: str) -> str:
    """Escape single quotes for Cypher string literals."""
    return s.replace("\\", "\\\\").replace("'", "\\'")


def norm_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def load_style_samples() -> list[dict]:
    rows = [json.loads(line) for line in EVAL_PATH.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row.get("source") == "external"]

    # Sample repeatedly from a large external pool for style variety.
    seeds = [42, 420, 4242]
    samples: list[dict] = []
    for seed in seeds:
        rng = random.Random(seed)
        picks = rng.sample(rows, min(20, len(rows)))
        samples.extend(picks)
    return samples


def fetch_distinct(session, cypher: str, key: str) -> list[str]:
    return [r[key] for r in session.run(cypher).data() if r.get(key)]


def gather_graph_values(driver) -> dict[str, list[str]]:
    with driver.session() as session:
        values = {
            "artists": fetch_distinct(session, "MATCH (a:Artist) RETURN DISTINCT a.name AS v ORDER BY v", "v"),
            "genres": fetch_distinct(session, "MATCH (g:Genre) RETURN DISTINCT g.name AS v ORDER BY v", "v"),
            "albums": fetch_distinct(session, "MATCH (a:Album) RETURN DISTINCT a.title AS v ORDER BY v", "v"),
            "singles": fetch_distinct(session, "MATCH (s:Single) RETURN DISTINCT s.name AS v ORDER BY v", "v"),
            "playlists": fetch_distinct(session, "MATCH (p:Playlist) RETURN DISTINCT p.name AS v ORDER BY v", "v"),
            "moods": fetch_distinct(session, "MATCH (m:Mood) RETURN DISTINCT m.name AS v ORDER BY v", "v"),
            "topics": fetch_distinct(session, "MATCH (t:Topic) RETURN DISTINCT t.name AS v ORDER BY v", "v"),
            "places": fetch_distinct(session, "MATCH (p:Place) RETURN DISTINCT p.name AS v ORDER BY v", "v"),
            "languages": fetch_distinct(session, "MATCH (t:Track) WHERE t.language IS NOT NULL RETURN DISTINCT t.language AS v ORDER BY v", "v"),
        }
    for key, vals in values.items():
        if not vals:
            raise RuntimeError(f"No values fetched for {key}")
    return values


def build_candidates(values: dict[str, list[str]]) -> list[Pair]:
    rng = random.Random(RANDOM_SEED)

    artists = values["artists"][:40]
    genres = values["genres"][:20]
    albums = values["albums"][:30]
    singles = values["singles"][:30]
    playlists = values["playlists"]
    moods = values["moods"][:20]
    topics = values["topics"][:25]
    places = values["places"][:20]
    languages = values["languages"][:10]

    pairs: list[Pair] = []
    seen_q: set[str] = set()
    seen_c: set[str] = set()

    def add(question: str, cypher: str, category: str) -> None:
        nq = norm_text(question)
        nc = norm_text(cypher)
        if nq in seen_q or nc in seen_c:
            return
        if "group by" in nc or "select " in nc or " from " in nc:
            return
        pairs.append(Pair(question=question, cypher=cypher.strip(), category=category))
        seen_q.add(nq)
        seen_c.add(nc)

    # 1) Era coverage and stats (all eras included multiple times).
    for era in ERAS:
        for lim in LIMITS:
            add(
                f"Show the top {lim} most played tracks from the {era} era.",
                f"MATCH (t:Track)-[:IN_ERA]->(e:Era {{name: '{q(era)}'}}) MATCH (t)-[:BY]->(a:Artist) WHERE t.play_count > 0 RETURN t.name AS track, a.name AS artist, t.play_count AS plays ORDER BY plays DESC LIMIT {lim}",
                "era",
            )
            add(
                f"Which {lim} artists have the most total plays in the {era} era?",
                f"MATCH (t:Track)-[:IN_ERA]->(e:Era {{name: '{q(era)}'}}) MATCH (t)-[:BY]->(a:Artist) RETURN a.name AS artist, sum(t.play_count) AS total_plays, count(t) AS track_count ORDER BY total_plays DESC LIMIT {lim}",
                "era",
            )

    # 2) Playlist analytics with 3+ MATCH and collect.
    for playlist in playlists:
        for lim in [10, 15, 20]:
            add(
                f"For playlist {playlist}, what are the top {lim} genres by track count?",
                f"MATCH (t:Track)-[:IN_PLAYLIST]->(p:Playlist {{name: '{q(playlist)}'}}) MATCH (t)-[:IN_GENRE]->(g:Genre) RETURN g.name AS genre, count(t) AS tracks ORDER BY tracks DESC LIMIT {lim}",
                "playlist",
            )
            add(
                f"Which {lim} artists appear most in playlist {playlist}?",
                f"MATCH (t:Track)-[:IN_PLAYLIST]->(p:Playlist {{name: '{q(playlist)}'}}) MATCH (t)-[:BY]->(a:Artist) RETURN a.name AS artist, count(t) AS tracks, sum(t.play_count) AS total_plays ORDER BY tracks DESC, total_plays DESC LIMIT {lim}",
                "playlist",
            )
            add(
                f"In playlist {playlist}, list up to {lim} tracks with their genre and era.",
                f"MATCH (t:Track)-[:IN_PLAYLIST]->(p:Playlist {{name: '{q(playlist)}'}}) MATCH (t)-[:IN_GENRE]->(g:Genre) MATCH (t)-[:IN_ERA]->(e:Era) RETURN t.name AS track, g.name AS genre, e.name AS era ORDER BY t.play_count DESC LIMIT {lim}",
                "playlist",
            )

        add(
            f"For playlist {playlist}, collect up to five unique genres for each artist.",
            f"MATCH (t:Track)-[:IN_PLAYLIST]->(p:Playlist {{name: '{q(playlist)}'}}) MATCH (t)-[:BY]->(a:Artist) MATCH (t)-[:IN_GENRE]->(g:Genre) RETURN a.name AS artist, collect(DISTINCT g.name)[0..5] AS genres, count(DISTINCT t) AS tracks ORDER BY tracks DESC LIMIT 15",
            "playlist",
        )

    # 3) Artist-focused analytics.
    for artist in artists:
        lim = rng.choice(LIMITS)
        add(
            f"What are the top {lim} most played tracks by {artist}?",
            f"MATCH (t:Track)-[:BY]->(a:Artist {{name: '{q(artist)}'}}) RETURN t.name AS track, t.play_count AS plays, t.skip_count AS skips ORDER BY plays DESC LIMIT {lim}",
            "artist",
        )
        add(
            f"For {artist}, which genres have the most tracks?",
            f"MATCH (t:Track)-[:BY]->(a:Artist {{name: '{q(artist)}'}}) MATCH (t)-[:IN_GENRE]->(g:Genre) RETURN g.name AS genre, count(t) AS tracks, sum(t.play_count) AS total_plays ORDER BY tracks DESC LIMIT 10",
            "artist",
        )
        add(
            f"List up to {lim} collaborators featured on tracks by {artist}.",
            f"MATCH (t:Track)-[:BY]->(a:Artist {{name: '{q(artist)}'}}) OPTIONAL MATCH (t)-[:FEATURES]->(f:Artist) WITH t, f WHERE f IS NOT NULL RETURN f.name AS featured_artist, count(t) AS collaborations ORDER BY collaborations DESC LIMIT {lim}",
            "artist",
        )
        add(
            f"For {artist}, collect up to five distinct featured artists ordered by collaboration frequency.",
            f"MATCH (t:Track)-[:BY]->(a:Artist {{name: '{q(artist)}'}}) OPTIONAL MATCH (t)-[:FEATURES]->(f:Artist) WITH a, f, count(t) AS c WHERE f IS NOT NULL ORDER BY c DESC RETURN a.name AS artist, collect(f.name)[0..5] AS top_featured",
            "artist",
        )

    # 4) Lyrics-driven analytics (mood/topic/place, optional match, with-where).
    for mood in moods:
        lim = rng.choice(LIMITS)
        add(
            f"Show {lim} tracks tagged with mood {mood} and their play counts.",
            f"MATCH (t:Track)-[:HAS_MOOD]->(m:Mood {{name: '{q(mood)}'}}) MATCH (t)-[:BY]->(a:Artist) WHERE t.lyrics_found = true RETURN t.name AS track, a.name AS artist, t.play_count AS plays ORDER BY plays DESC LIMIT {lim}",
            "lyrics",
        )

    for topic in topics:
        lim = rng.choice(LIMITS)
        add(
            f"Which {lim} tracks mention topic {topic} with the highest type-token ratio?",
            f"MATCH (t:Track)-[:HAS_TOPIC]->(tp:Topic {{name: '{q(topic)}'}}) MATCH (t)-[:BY]->(a:Artist) WHERE t.lyrics_found = true AND t.type_token_ratio IS NOT NULL RETURN t.name AS track, a.name AS artist, t.type_token_ratio AS ttr ORDER BY ttr DESC LIMIT {lim}",
            "lyrics",
        )

    for place in places:
        lim = rng.choice(LIMITS)
        add(
            f"Which {lim} most played tracks mention {place} in the lyrics?",
            f"MATCH (t:Track)-[:MENTIONS_PLACE]->(pl:Place {{name: '{q(place)}'}}) MATCH (t)-[:BY]->(a:Artist) WHERE t.lyrics_found = true RETURN t.name AS track, a.name AS artist, t.play_count AS plays ORDER BY plays DESC LIMIT {lim}",
            "lyrics",
        )

    # 5) Language and lexical metrics.
    for lang in languages:
        for lim in [5, 10, 20]:
            add(
                f"In language code {lang}, who are the top {lim} artists by total plays?",
                f"MATCH (t:Track)-[:BY]->(a:Artist) WHERE t.language = '{q(lang)}' RETURN a.name AS artist, sum(t.play_count) AS total_plays, count(t) AS tracks ORDER BY total_plays DESC LIMIT {lim}",
                "language",
            )
        add(
            f"For language code {lang}, show tracks with high repetition rate and at least 50 total words.",
            f"MATCH (t:Track)-[:BY]->(a:Artist) WHERE t.language = '{q(lang)}' AND t.total_words >= 50 AND t.repetition_rate IS NOT NULL RETURN t.name AS track, a.name AS artist, t.repetition_rate AS repetition_rate ORDER BY repetition_rate DESC LIMIT 15",
            "language",
        )

    # 6) Album/single coverage.
    for album in albums:
        lim = rng.choice(LIMITS)
        add(
            f"Show up to {lim} tracks from album {album} sorted by play count.",
            f"MATCH (t:Track)-[:ON]->(al:Album {{title: '{q(album)}'}}) MATCH (t)-[:BY]->(a:Artist) RETURN t.name AS track, a.name AS artist, t.play_count AS plays ORDER BY plays DESC LIMIT {lim}",
            "album",
        )

    for single in singles:
        lim = rng.choice(LIMITS)
        add(
            f"For single release {single}, list up to {lim} related tracks and artists.",
            f"MATCH (t:Track)-[:IS_SINGLE]->(s:Single {{name: '{q(single)}'}}) MATCH (t)-[:BY]->(a:Artist) RETURN t.name AS track, a.name AS artist, t.play_count AS plays ORDER BY plays DESC LIMIT {lim}",
            "single",
        )

    # 7) Additional optional-match heavy templates to satisfy quota.
    add(
        "Find tracks with lyrics where mood tags are missing, then show artist and play count.",
        "MATCH (t:Track) WHERE t.lyrics_found = true OPTIONAL MATCH (t)-[:HAS_MOOD]->(m:Mood) WITH t, count(m) AS mood_count WHERE mood_count = 0 MATCH (t)-[:BY]->(a:Artist) RETURN t.name AS track, a.name AS artist, t.play_count AS plays ORDER BY plays DESC LIMIT 25",
        "optional",
    )
    add(
        "Find tracks with lyrics where topic tags are missing, then show artist and play count.",
        "MATCH (t:Track) WHERE t.lyrics_found = true OPTIONAL MATCH (t)-[:HAS_TOPIC]->(tp:Topic) WITH t, count(tp) AS topic_count WHERE topic_count = 0 MATCH (t)-[:BY]->(a:Artist) RETURN t.name AS track, a.name AS artist, t.play_count AS plays ORDER BY plays DESC LIMIT 25",
        "optional",
    )
    add(
        "Find tracks with lyrics where place mentions are missing, then show artist and play count.",
        "MATCH (t:Track) WHERE t.lyrics_found = true OPTIONAL MATCH (t)-[:MENTIONS_PLACE]->(pl:Place) WITH t, count(pl) AS place_count WHERE place_count = 0 MATCH (t)-[:BY]->(a:Artist) RETURN t.name AS track, a.name AS artist, t.play_count AS plays ORDER BY plays DESC LIMIT 25",
        "optional",
    )
    add(
        "For each genre, collect up to five playlist names where its tracks appear.",
        "MATCH (t:Track)-[:IN_GENRE]->(g:Genre) OPTIONAL MATCH (t)-[:IN_PLAYLIST]->(p:Playlist) WITH g, p WHERE p IS NOT NULL RETURN g.name AS genre, collect(DISTINCT p.name)[0..5] AS playlists, count(DISTINCT t) AS tracks ORDER BY tracks DESC LIMIT 20",
        "optional",
    )
    add(
        "For each era, collect up to five places mentioned by tracks in that era.",
        "MATCH (t:Track)-[:IN_ERA]->(e:Era) OPTIONAL MATCH (t)-[:MENTIONS_PLACE]->(pl:Place) WITH e, pl WHERE pl IS NOT NULL RETURN e.name AS era, collect(DISTINCT pl.name)[0..5] AS places, count(DISTINCT t) AS tracks ORDER BY tracks DESC LIMIT 10",
        "optional",
    )

    # 8) Relationship-driven filters and statistics.
    for genre in genres:
        lim = rng.choice(LIMITS)
        add(
            f"Within genre {genre}, which {lim} tracks have the highest skip-to-play ratio?",
            f"MATCH (t:Track)-[:IN_GENRE]->(g:Genre {{name: '{q(genre)}'}}) MATCH (t)-[:BY]->(a:Artist) WHERE t.play_count > 0 RETURN t.name AS track, a.name AS artist, toFloat(t.skip_count) / t.play_count AS skip_ratio ORDER BY skip_ratio DESC LIMIT {lim}",
            "genre",
        )
        add(
            f"In genre {genre}, which artists have the most loved tracks?",
            f"MATCH (t:Track)-[:IN_GENRE]->(g:Genre {{name: '{q(genre)}'}}) MATCH (t)-[:BY]->(a:Artist) WHERE t.loved = true RETURN a.name AS artist, count(t) AS loved_tracks, sum(t.play_count) AS total_plays ORDER BY loved_tracks DESC, total_plays DESC LIMIT 15",
            "genre",
        )

    rng.shuffle(pairs)
    return pairs


def validate_pair(session, pair: Pair) -> ValidationResult:
    try:
        session.run(f"EXPLAIN {pair.cypher}").consume()
        session.run(pair.cypher).consume()
        return ValidationResult(pair=pair, valid=True)
    except Neo4jError as e:
        # Single retry pass (as requested): apply small textual cleanup then re-run.
        retry = pair.cypher.replace("  ", " ").strip()
        if retry != pair.cypher:
            try:
                session.run(f"EXPLAIN {retry}").consume()
                session.run(retry).consume()
                return ValidationResult(pair=Pair(pair.question, retry, pair.category), valid=True)
            except Neo4jError as e2:
                return ValidationResult(pair=pair, valid=False, reason=f"{e2.code}: {str(e2).splitlines()[0]}")
        return ValidationResult(pair=pair, valid=False, reason=f"{e.code}: {str(e).splitlines()[0]}")


def has_minimum_diversity(pairs: Iterable[Pair]) -> tuple[bool, dict[str, int]]:
    items = list(pairs)
    cy = [p.cypher.lower() for p in items]

    counts = {
        "rows": len(items),
        "collect": sum("collect(" in c for c in cy),
        "optional_match": sum("optional match" in c for c in cy),
        "with_where": sum(("with " in c and " where " in c) for c in cy),
        "match_3_plus": sum(c.count("match ") >= 3 for c in cy),
        "era_pre_90s": sum("'pre-90s'" in c for c in cy),
        "era_90s": sum("'90s'" in c for c in cy),
        "era_2000s": sum("'2000s'" in c for c in cy),
        "era_2010s": sum("'2010s'" in c for c in cy),
        "era_2020s": sum("'2020s'" in c for c in cy),
        "limit_5": sum(" limit 5" in c for c in cy),
        "limit_10": sum(" limit 10" in c for c in cy),
        "limit_15": sum(" limit 15" in c for c in cy),
        "limit_20": sum(" limit 20" in c for c in cy),
        "limit_25": sum(" limit 25" in c for c in cy),
        "limit_30": sum(" limit 30" in c for c in cy),
    }

    ok = (
        counts["rows"] == TARGET_ROWS
        and counts["collect"] >= 5
        and counts["optional_match"] >= 5
        and counts["with_where"] >= 3
        and counts["match_3_plus"] >= 10
        and counts["era_pre_90s"] >= 1
        and counts["era_90s"] >= 1
        and counts["era_2000s"] >= 1
        and counts["era_2010s"] >= 1
        and counts["era_2020s"] >= 1
        and counts["limit_5"] >= 1
        and counts["limit_10"] >= 1
        and counts["limit_15"] >= 1
        and counts["limit_20"] >= 1
        and counts["limit_25"] >= 1
        and counts["limit_30"] >= 1
    )
    return ok, counts


def write_output(pairs: list[Pair]) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for pair in pairs:
            row = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": pair.question},
                    {"role": "assistant", "content": pair.cypher},
                ]
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_journal(
    style_samples: list[dict],
    values: dict[str, list[str]],
    accepted: list[Pair],
    discarded: list[ValidationResult],
    diversity_counts: dict[str, int],
) -> None:
    JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(RANDOM_SEED)
    sample_rows = rng.sample(accepted, min(3, len(accepted)))

    lines: list[str] = []
    lines.append("# Cyper Dataset Generation Journal")
    lines.append("")
    lines.append(f"- target_rows: {TARGET_ROWS}")
    lines.append(f"- accepted_rows: {len(accepted)}")
    lines.append(f"- discarded_rows: {len(discarded)}")
    lines.append(f"- output_file: {OUT_PATH}")
    lines.append("")

    lines.append("## Graph Value Snapshot")
    for key, vals in values.items():
        lines.append(f"- {key}: {len(vals)} values")
    lines.append("")

    lines.append("## Style Samples (External Eval, multi-sample rounds)")
    for i, row in enumerate(style_samples[:9], start=1):
        q_txt = row["conversations"][1]["content"].strip().replace("\n", " ")[:220]
        c_txt = row["conversations"][2]["content"].strip().replace("\n", " ")[:220]
        lines.append(f"{i}. Q: {q_txt}")
        lines.append(f"   C: {c_txt}")
    lines.append("")

    lines.append("## Diversity Counters")
    for k, v in diversity_counts.items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    lines.append("## Category Counts")
    cat_counts = Counter(p.category for p in accepted)
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- {cat}: {cnt}")
    lines.append("")

    lines.append("## Accepted Samples")
    for i, pair in enumerate(sample_rows, start=1):
        lines.append(f"{i}. Question: {pair.question}")
        lines.append(f"   Cypher: {pair.cypher}")
    lines.append("")

    lines.append("## Discarded Pairs")
    if not discarded:
        lines.append("- none")
    else:
        for i, d in enumerate(discarded[:30], start=1):
            lines.append(f"{i}. Question: {d.pair.question}")
            lines.append(f"   Reason: {d.reason}")

    JOURNAL_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    DISCARDED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DISCARDED_PATH.open("w", encoding="utf-8") as f:
        for d in discarded:
            row = {
                "question": d.pair.question,
                "cypher": d.pair.cypher,
                "category": d.pair.category,
                "reason": d.reason,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    load_dotenv(dotenv_path=Path(".env"))
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "54321Dan")

    style_samples = load_style_samples()
    print(f"Loaded external style samples: {len(style_samples)} (3 rounds x up to 20)")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    print(f"Neo4j connected: {uri}")

    try:
        values = gather_graph_values(driver)
        candidates = build_candidates(values)
        print(f"Generated fresh candidates: {len(candidates)}")

        accepted: list[Pair] = []
        discarded: list[ValidationResult] = []

        with driver.session() as session:
            for idx, pair in enumerate(candidates, start=1):
                res = validate_pair(session, pair)
                if res.valid:
                    accepted.append(res.pair)
                else:
                    discarded.append(res)

                if len(accepted) % 10 == 0 and len(accepted) > 0:
                    print(f"Validated {len(accepted)} / {TARGET_ROWS}")

                if len(accepted) >= TARGET_ROWS:
                    break

        if len(accepted) < MIN_VALIDATED:
            raise RuntimeError(
                f"Validated {len(accepted)} rows, below minimum threshold {MIN_VALIDATED}."
            )
        if len(accepted) < TARGET_ROWS:
            raise RuntimeError(
                f"Validated {len(accepted)} rows; need {TARGET_ROWS}. Increase candidate generation."
            )

        accepted = accepted[:TARGET_ROWS]

        diversity_ok, diversity_counts = has_minimum_diversity(accepted)
        if not diversity_ok:
            raise RuntimeError(
                "Diversity constraints not satisfied for accepted set: "
                + json.dumps(diversity_counts, indent=2)
            )

        write_output(accepted)
        write_journal(style_samples, values, accepted, discarded, diversity_counts)

        category_counts = Counter(p.category for p in accepted)
        print("\n=== Final Report ===")
        print(f"Total rows written: {len(accepted)}")
        print("Count per category:")
        for cat, cnt in sorted(category_counts.items(), key=lambda x: (-x[1], x[0])):
            print(f"  - {cat}: {cnt}")
        print(f"Discarded pairs: {len(discarded)}")

        rng = random.Random(RANDOM_SEED)
        sample_rows = rng.sample(accepted, 3)
        print("\n3 random sample rows:")
        for i, row in enumerate(sample_rows, start=1):
            print(f"{i}. Q: {row.question}")
            print(f"   C: {row.cypher}")

        print(f"\nWrote dataset: {OUT_PATH}")
        print(f"Wrote journal: {JOURNAL_PATH}")
        print(f"Wrote discarded log: {DISCARDED_PATH}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
