"""
training/prepare_data.py

Downloads neo4j/text2cypher-2024v1, formats as chatml, optionally merges
TuneMap-specific examples, and writes train.jsonl + eval.jsonl.

Usage:
    python training/prepare_data.py
"""

import json
import logging
import random
from pathlib import Path

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Task: Generate Cypher statement to query a graph database.\n"
    "Instructions: Use only the provided relationship types and properties in the schema.\n"
    "Do not use any other relationship types or properties that are not provided in the schema.\n"
    "Do not include any explanations or apologies in your responses.\n"
    "Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.\n"
    "Do not include any text except the generated Cypher statement."
)

TUNEMAP_VALIDATED = Path("training/data/cypher_dataset_validated.jsonl")
TRAIN_OUT = Path("training/data/train.jsonl")
EVAL_OUT = Path("training/data/eval.jsonl")
SEED = 3407


def format_external_row(row: dict) -> dict:
    user_content = (
        "Generate Cypher statement to query a graph database. "
        "Use only the provided relationship types and properties in the schema.\n"
        f"Schema: {row['schema']}\n"
        f"Question: {row['question']}\n"
        "Cypher output:"
    )
    return {
        "conversations": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": row["cypher"]},
        ],
        "source": "external",
        "database_reference": row.get("database_reference_alias"),
    }


def main():
    log.info("Loading neo4j/text2cypher-2024v1 ...")
    ds = load_dataset("neo4j/text2cypher-2024v1")

    train_rows = [format_external_row(r) for r in ds["train"]]
    eval_rows = [format_external_row(r) for r in ds["test"]]
    log.info(f"External: {len(train_rows)} train, {len(eval_rows)} eval")

    # T004 gate: add TuneMap benchmark rows to eval only (domain benchmark — never in training)
    if TUNEMAP_VALIDATED.exists():
        tunemap_all = [json.loads(l) for l in TUNEMAP_VALIDATED.read_text().splitlines() if l.strip()]

        # generate_dataset.py puts schema in system prompt and bare question in user turn.
        # Reformat to standard 3-turn chatml: user turn includes schema + "Cypher output:".
        # Schema format matches neo4j/text2cypher-2024v1 exactly so the model sees familiar context.
        tunemap_schema = """\
Node properties:
- **Track**
  - `name`: STRING Example: "The Wolf"
  - `year`: INTEGER Min: 1945, Max: 2026
  - `release_date`: STRING Example: "2015-03-01"
  - `duration_ms`: INTEGER Min: 9814, Max: 2069780
  - `play_count`: INTEGER Min: 0, Max: 93
  - `skip_count`: INTEGER Min: 0, Max: 22
  - `loved`: BOOLEAN
  - `explicit`: BOOLEAN
  - `date_added`: STRING Example: "2021-06-15"
  - `track_number`: INTEGER
  - `language`: STRING Available options: ['en', 'ru', 'es', 'fr', 'de', 'ja', 'pt', 'it', 'ko', 'ro', 'uk', 'id', 'tl', 'hr', 'fi', 'af', 'so', 'lt', 'nl', 'unknown']
  - `lyrics_found`: BOOLEAN
  - `unique_words`: INTEGER
  - `total_words`: INTEGER
  - `type_token_ratio`: FLOAT
  - `repetition_rate`: FLOAT
- **Artist**
  - `name`: STRING Example: "Billy Talent"
- **Album**
  - `title`: STRING Example: "The Slim Shady LP"
  - `year`: INTEGER Min: 1945, Max: 2026
- **Single**
  - `name`: STRING Example: "Fresh Outta London"
- **Genre**
  - `name`: STRING Example: "Alternative"
- **Era**
  - `name`: STRING Available options: ['Pre-90s', '90s', '2000s', '2010s', '2020s']
- **Playlist**
  - `name`: STRING Available options: ['Billy Talent Essentials', 'Chill', 'Eminem Essentials', 'Fun music', 'My Shazam Tracks', 'Replay 2019', 'Replay 2020']
- **Mood**
  - `name`: STRING Example: "introspective"
- **Topic**
  - `name`: STRING Example: "alienation"
- **Place**
  - `name`: STRING Example: "new york"
Relationship properties:
- **IN_GENRE**
  - `track_count: INTEGER`
The relationships:
(:Track)-[:BY]->(:Artist)
(:Track)-[:FEATURES]->(:Artist)
(:Track)-[:ON]->(:Album)
(:Track)-[:IS_SINGLE]->(:Single)
(:Track)-[:IN_GENRE]->(:Genre)
(:Track)-[:IN_ERA]->(:Era)
(:Track)-[:IN_PLAYLIST]->(:Playlist)
(:Track)-[:HAS_MOOD]->(:Mood)
(:Track)-[:HAS_TOPIC]->(:Topic)
(:Track)-[:MENTIONS_PLACE]->(:Place)
(:Album)-[:BY]->(:Artist)
(:Artist)-[:IN_GENRE {track_count}]->(:Genre)"""

        def reformat_tunemap(row: dict) -> dict:
            msgs = row["messages"]
            question = msgs[1]["content"]
            user_content = (
                "Generate Cypher statement to query a graph database. "
                "Use only the provided relationship types and properties in the schema.\n"
                f"Schema: {tunemap_schema}\n"
                f"Question: {question}\n"
                "Cypher output:"
            )
            return {
                "conversations": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": msgs[2]["content"]},
                ],
                "source": "tunemap",
                "database_reference": None,
            }

        tunemap_eval = [reformat_tunemap(r) for r in tunemap_all]
        eval_rows.extend(tunemap_eval)
        log.info(f"TuneMap benchmark: +{len(tunemap_eval)} rows → eval only")
    else:
        log.info("TuneMap benchmark: cypher_dataset_validated.jsonl not found, skipping")

    random.seed(SEED)
    random.shuffle(train_rows)

    TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAIN_OUT, "w", encoding="utf-8") as f:
        for row in train_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(EVAL_OUT, "w", encoding="utf-8") as f:
        for row in eval_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    db_ref_count = sum(1 for r in eval_rows if r.get("database_reference"))
    log.info(f"Wrote {len(train_rows)} rows → {TRAIN_OUT}")
    log.info(f"Wrote {len(eval_rows)} rows → {EVAL_OUT}  ({db_ref_count} with database_reference)")


if __name__ == "__main__":
    main()
