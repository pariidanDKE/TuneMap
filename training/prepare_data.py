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

    # T004 gate: merge TuneMap data if validated file exists
    if TUNEMAP_VALIDATED.exists():
        tunemap_all = [json.loads(l) for l in TUNEMAP_VALIDATED.read_text().splitlines() if l.strip()]
        random.seed(SEED)
        random.shuffle(tunemap_all)
        split = max(1, int(len(tunemap_all) * 0.9))
        tunemap_train = tunemap_all[:split]
        tunemap_eval = tunemap_all[split:]

        # Reformat TuneMap rows into standard chatml structure.
        # generate_dataset.py puts schema in system prompt and bare question in user turn.
        # We need the standard 3-turn format: user turn must include schema + "Cypher output:".
        tunemap_schema = (
            "Node types: Track, Artist, Album, Single, Genre, Era, Playlist, Mood, Topic, Place. "
            "Relationships: (Track)-[:BY]->(Artist), (Track)-[:FEATURES]->(Artist), "
            "(Track)-[:ON]->(Album), (Track)-[:IS_SINGLE]->(Single), (Track)-[:IN_GENRE]->(Genre), "
            "(Track)-[:IN_ERA]->(Era), (Track)-[:IN_PLAYLIST]->(Playlist), "
            "(Track)-[:HAS_MOOD]->(Mood), (Track)-[:HAS_TOPIC]->(Topic), "
            "(Track)-[:MENTIONS_PLACE]->(Place), (Album)-[:BY]->(Artist), "
            "(Artist)-[:IN_GENRE {track_count}]->(Genre)."
        )

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

        tunemap_train = [reformat_tunemap(r) for r in tunemap_train]
        tunemap_eval = [reformat_tunemap(r) for r in tunemap_eval]

        train_rows.extend(tunemap_train)
        eval_rows.extend(tunemap_eval)
        log.info(f"TuneMap gate OPEN: +{len(tunemap_train)} train, +{len(tunemap_eval)} eval")
    else:
        log.info("TuneMap gate CLOSED: cypher_dataset_validated.jsonl not found, skipping")

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
