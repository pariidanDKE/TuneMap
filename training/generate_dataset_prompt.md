# TuneMap Cypher Dataset Generation

**Working directory: project root** (all relative paths below are relative to it).

You are tasked with generating a high-quality (question, Cypher) dataset for the TuneMap
Apple Music Knowledge Graph. Execute every step below in order. Do not stop until
`training/data/cypher_dataset_validated.jsonl` has been written.

---

## Step 1 — Read style examples from eval.jsonl

Read `training/data/eval.jsonl`. Each row is a JSON object with a `"conversations"` key
(list of 3 turns: system / user / assistant). Filter to rows where `source == "external"`.
Sample 20 rows at random (seed 42). The Cypher is at `row["conversations"][2]["content"]`.

These are your **style reference** — study the Cypher style (aliases, spacing, clause
ordering, LIMIT placement, how aggregation is written). Your generated Cypher must match
this style, not a different one.

Print the 3 most representative samples so you can confirm the style before generating.

---

## Step 2 — Study the TuneMap schema

The graph you are writing Cypher for has this exact schema:

```
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
(:Artist)-[:IN_GENRE {track_count}]->(:Genre)
```

**Hard constraints — never violate:**
- Cypher only. No SQL. Never use SELECT, FROM, GROUP BY.
- String literals containing double quotes must use single-quote delimiters.
- No GROUP BY — Cypher aggregation is implicit.
- No cartesian products — every MATCH clause must share a variable with another clause.
- LIMIT always comes after ORDER BY.
- A track has EITHER `[:ON]->(:Album)` OR `[:IS_SINGLE]->(:Single)`, never both.
- Mood, Topic, Place nodes only exist for tracks where `lyrics_found = true`.
- Mood and Topic `name` values are always lowercase in the graph.
- `language` values are ISO 639-1 codes: `'en'`, `'ru'`, `'fr'`, etc. — never `'English'`.
- "My library" means the whole graph (all tracks), NOT a specific playlist.
- Use `WITH` to pass variables between pipeline stages, not nested MATCH.

---

## Step 3 — Generate 150 diverse (question, Cypher) pairs

Generate exactly **150** pairs covering the full breadth of the schema. Ensure good
coverage across: Track properties (play_count, skip_count, loved, explicit, duration_ms),
Artist (BY and FEATURES), Album, Single (IS_SINGLE), Genre, Era (all 5 must appear),
Playlist (use only real names from schema), Mood, Topic, Place, language codes,
lyrics analytics (type_token_ratio, repetition_rate, unique_words), date/time fields,
aggregation pipelines, multi-hop queries, and counting/statistics.

**Variety rules:**
- No two questions may be rephrasings of each other. Each must retrieve meaningfully
  different data.
- Vary LIMIT values: use 5, 10, 15, 20, 25, 30 across the dataset.
- Vary whether ORDER BY is ASC or DESC.
- Include both property filters (`WHERE t.loved = true`) and relationship traversal filters.
- At least 10 questions must involve 3 or more MATCH clauses.
- At least 5 questions must use `collect()`.
- At least 5 questions must use `OPTIONAL MATCH`.
- At least 3 questions must use `WITH ... WHERE` for intermediate filtering.
- Do NOT include questions about "my workout playlist" or generic playlist names that
  don't exist in the schema. Use only playlist names from the Available options above.

---

## Step 4 — Validate every query with EXPLAIN

Load `.env` from the project root (`python-dotenv` or manual parse). Credentials:
- `NEO4J_URI` (default: `bolt://localhost:7687`)
- `NEO4J_USER` (default: `neo4j`)
- `NEO4J_PASSWORD` (default: `54321Dan`)

For each generated (question, Cypher) pair, run two checks:

1. **EXPLAIN** — fast syntax check:
   ```python
   session.run(f"EXPLAIN {cypher}").consume()
   ```
2. **Live execution** — catches semantic errors (wrong property names, bad node labels, etc.):
   ```python
   session.run(cypher).consume()
   ```

If either raises an exception → fix the Cypher and retry once. If it still fails, discard
the pair and generate a replacement.

Print a running count: `Validated X / 150` every 10 rows.

Suppress neo4j notification warnings: `logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)`

Target: ≥140 validated pairs. If you end with fewer, generate additional pairs to reach 140.

---

## Step 5 — Format and write output

Each validated pair must be written as one JSON object per line to
`training/data/cypher_dataset_validated.jsonl`.

Format — match exactly what `prepare_data.py` expects:

```json
{
  "messages": [
    {"role": "system", "content": "<the system prompt below>"},
    {"role": "user",   "content": "<the question>"},
    {"role": "assistant", "content": "<the cypher — no markdown fences, no explanation>"}
  ]
}
```

System prompt to use verbatim for every row:
```
Task: Generate Cypher statement to query a graph database.
Instructions: Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided in the schema.
Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.
```

**Cypher field rules:**
- Raw Cypher only — no triple backticks, no language tag, no explanation text.
- Multi-line Cypher is fine (use `\n`).
- No trailing whitespace.

---

## Step 6 — Report

After writing the file, print:
- Total rows written
- Count per category
- Any pairs that were discarded and why
- 3 random sample rows (question + Cypher) from the final file
