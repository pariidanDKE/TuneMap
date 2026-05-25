# Data Model: NL-to-Cypher LoRA Fine-Tuning

**Branch**: `cyper_finetune` | **Date**: 2026-04-15

---

## Core Entity: TrainingExample

The atomic unit throughout the pipeline. One (system, user, assistant) chat turn.

```
TrainingExample
├── system    : str   — full instruction prompt (fixed for all examples, see below)
├── user      : str   — "Generate Cypher...\nSchema: {schema}\nQuestion: {question}\nCypher output:"
└── assistant : str   — Cypher query, no markdown fences, no explanation
```

**Serialised format** (JSONL, one object per line):
```json
{
  "conversations": [
    {"role": "system",    "content": "Task: Generate Cypher statement to query a graph database.\nInstructions: Use only the provided relationship types and properties in the schema.\nDo not use any other relationship types or properties that are not provided in the schema.\nDo not include any explanations or apologies in your responses.\nDo not respond to any questions that might ask anything else than for you to construct a Cypher statement.\nDo not include any text except the generated Cypher statement."},
    {"role": "user",      "content": "Generate Cypher statement to query a graph database. Use only the provided relationship types and properties in the schema.\nSchema: ...\nQuestion: ...\nCypher output:"},
    {"role": "assistant", "content": "MATCH (n:Track) RETURN n.name LIMIT 10"}
  ],
  "source": "external" | "tunemap"
}
```

---

## Data Sources

### Source A — neo4j/text2cypher-2024v1 (primary)

| Field | Maps to |
|---|---|
| `question` | `user` (after schema prefix) |
| `schema` | `user` (schema block) |
| `cypher` | `assistant` |
| `data_source` | retained as `source` metadata |
| `instance_id` | retained for deduplication |
| `database_reference` | retained — routes execution-based eval to the correct Neo4j database (51.12% of test rows have this set) |

- **Train split**: 39,554 rows → `training/data/train.jsonl`
- **Test split**: 4,833 rows → `training/data/eval.jsonl`
- **Status**: Available via HuggingFace `datasets`. No validation required (curated by neo4j).

### Source B — TuneMap-specific (supplementary)

| Field | Maps to |
|---|---|
| `system` | fixed system prompt |
| `user` (NL question) | `user` |
| `assistant` (Cypher) | `assistant` |

- **Generator**: `training/generate_dataset.py`
- **Output**: `training/data/cypher_dataset.jsonl`
- **Expected size**: ~100 rows covering all 10 TuneMap node types and 12 relationship types
- **Status**: ⚠️ NOT YET VALIDATED — script has not been run against the live Neo4j KG. Cypher queries may contain syntax errors or return no results.
- **Gate**: Source B rows are mixed into training ONLY after a validation step confirms Cypher executes without syntax errors on the live graph. Until then, `prepare_data.py` skips this source (a `--skip-tunemap` flag or simple `if validated_path.exists()` guard).

---

## Dataset Splits

```
External (neo4j/text2cypher-2024v1):
  train  →  39,554 examples  →  training/data/train.jsonl      (primary)
  test   →   4,833 examples  →  training/data/eval.jsonl       (translation-based eval: all 4,833; execution-based eval: ~2,471 with database_reference set)

TuneMap-specific (after validation):
  all    →    ~100 examples  →  appended to training/data/train.jsonl (90 rows)
                              →  appended to training/data/eval.jsonl  (10 rows)
```

**Shuffle**: `train.jsonl` shuffled with seed 3407 after merge.

**No re-splitting of external data**: The external dataset's built-in `test` split is used as-is. This matches the neo4j reference paper's evaluation setup.

---

## Evaluation Record

Produced by `eval.py` and written to `training/outputs/eval_report.json`:

```
EvalReport
├── run_id               : str   — timestamp + checkpoint path
├── checkpoint           : str   — path to adapter checkpoint evaluated
├── dataset              : str   — "external" | "tunemap" | "combined"
├── n_translation        : int   — examples in translation pass (all test rows)
├── n_execution          : int   — examples in execution pass (database_reference set)
├── mean_gleu            : float — mean sentence_gleu over all translation examples
├── exec_exact_match_pct : float — % execution examples where result sets match exactly
├── per_example          : list[EvalRecord]
└── failures             : list[EvalRecord]  — 10 worst-GLEU examples

EvalRecord
├── question             : str
├── reference_cypher     : str
├── predicted_cypher     : str
├── gleu                 : float
├── database_reference   : str | null   — database alias; null if no DB access for this row
└── exec_exact_match     : bool | null  — null if database_reference is null
```

---

## State Transitions

```
generate_dataset.py runs
        │
        ▼
cypher_dataset.jsonl  ──(validation gate)──►  SKIP if not validated
        │
        ▼
prepare_data.py
  ├── downloads neo4j/text2cypher-2024v1
  ├── formats all rows as TrainingExample (chatml)
  ├── merges external + TuneMap (if validated)
  └── writes train.jsonl + eval.jsonl
        │
        ▼
train.py
  ├── loads train.jsonl
  ├── fine-tunes Qwen3.5-9B with LoRA
  └── writes checkpoints to training/outputs/
        │
        ▼
eval.py
  ├── loads eval.jsonl
  ├── Pass 1: generates predictions, computes sentence_gleu (all 4,833)
  ├── Pass 2: filters to database_reference rows, executes generated + reference Cypher,
  │           compares result sets as sorted strings, computes exec_exact_match (~2,471)
  └── writes training/outputs/eval_report.json
```

---

## Validation Rules

- **No markdown fences**: assistant turn must not start with ` ``` ` or contain `\`\`\`cypher`
- **No SQL keywords as top-level clauses**: no `SELECT`, `FROM`, `WHERE` as first word
- **No `GROUP BY`**: Cypher does not have it
- **TuneMap schema compliance** (Source B only): queries must reference only node/relationship types defined in `generate_dataset.py` SYSTEM_PROMPT schema
- **Syntax validation gate** (Source B only): each query run as `EXPLAIN <query>` against live Neo4j; any `CypherSyntaxError` excludes the row from training mix (method from paper Section 3.2)
