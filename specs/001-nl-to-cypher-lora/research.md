# Research: NL-to-Cypher LoRA Fine-Tuning

**Branch**: `cyper_finetune` | **Date**: 2026-04-15
**Phase**: 0 — resolves all NEEDS CLARIFICATION items from plan Technical Context

---

## R-001: Text-Only Fine-Tuning API — RESOLVED

**Decision**: Use **`FastLanguageModel`** for both loading and `get_peft_model`. Vision encoder is excluded from LoRA by omitting it from `target_modules` — no explicit `finetune_vision_layers` param needed.

**Source**: Unsloth official Qwen3.5 fine-tuning docs, "text-only fine-tuning" section (https://unsloth.ai/docs/models/qwen3.5/fine-tune).

**Context**: Qwen3.5 is described as a "Causal Language Model with Vision Encoder" (unified VLM). The docs provide two paths:
- **Text-only** (`FastLanguageModel` + `target_modules`) — our path
- **Vision fine-tuning** (`FastVisionModel` + `finetune_language_layers` / `finetune_vision_layers`) — for VLM tasks

For NL-to-Cypher, `FastLanguageModel` is correct. The vision encoder gets no LoRA adapters because it does not appear in `target_modules` — same functional outcome as `finetune_vision_layers=False`, different API.

**Confirmed API**:
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3.5-9B",
    max_seq_length=1600,
    load_in_4bit=False,
    load_in_16bit=True,        # correct unsloth param for native bf16 loading
    full_finetuning=False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=64,                      # docs show r=16; r=64 from neo4j paper — deliberate choice
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # belongs here per docs, not in from_pretrained
    random_state=3407,
    max_seq_length=1600,
)
```

---

## R-002: No-Quantization LoRA (Full bf16)

**Decision**: Load with `load_in_4bit=False, load_in_8bit=False, load_in_16bit=False` (all defaults False except `load_in_4bit=True` which must be explicitly disabled). dtype auto-detected as bfloat16. This is standard LoRA without BnB quantization.

**Rationale**: User specified LoRA not qLoRA due to model sensitivity to quantization. Full bf16 weights + LoRA adapters.

**Confirmed approach**:
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3.5-9B",
    max_seq_length=1600,
    load_in_4bit=False,
    load_in_16bit=True,    # correct unsloth param for native bf16; dtype arg not needed
    full_finetuning=False,
)
# use_gradient_checkpointing goes in get_peft_model, not here
```

---

## R-003: Paged Optimizer

**Decision**: `paged_adamw_8bit` — confirmed from the neo4j/text2cypher-gemma-2-9b-it reference training config.

**Rationale**: This was the exact optimizer used in the inspiration paper with identical LoRA rank, alpha, lr, max_seq_length, and epoch count. No reason to deviate.

**How to apply**: Set `optim="paged_adamw_8bit"` in `SFTConfig`.

---

## R-004: Gradient Accumulation for Effective Batch Size 32

**Decision**:
- `per_device_train_batch_size=1` → `gradient_accumulation_steps=32`
- `per_device_train_batch_size=2` → `gradient_accumulation_steps=16`

Try batch size 2 first; fall back to 1 if OOM during first training run. Effective batch size = 32 in both cases.

---

## R-005: Dataset — neo4j/text2cypher-2024v1

**Schema**:

| Field | Description |
|---|---|
| `question` | Natural language question |
| `schema` | Graph database schema description |
| `cypher` | Target Cypher query |
| `data_source` | Source alias (20 unique values) |
| `instance_id` | Row index string |
| `database_reference_alias` | Database alias |

**Splits**: train=39,554 / test=4,833 (total 44,387)

**Prompt template** (from neo4j reference paper):
```
Generate Cypher statement to query a graph database. Use only the provided relationship types and properties in the schema.
Schema: {schema}
Question: {question}
Cypher output:
```
Assistant turn: `{cypher}` (no markdown fences, no explanation)

**How to apply**: Format as 3-turn chat (system/user/assistant) using `get_chat_template(tokenizer, chat_template="chatml")` and train with `train_on_responses_only`. The `schema` and `question` fields map to user turn; `cypher` maps to assistant turn.

---

## R-006: TuneMap-Specific Dataset (generate_dataset.py)

**Status**: Script exists at `training/generate_dataset.py`. **NOT yet validated** — has not been run against the live Neo4j KG. Cypher queries may contain errors.

**Plan**: Use the external `neo4j/text2cypher-2024v1` as the primary training corpus. TuneMap-specific data supplements as a small "domain-adaptation tail" (~100 rows). Dataset construction (Phase 1 of training pipeline) must include a Cypher validation step before mixing with external data.

**Eval split**: Use external dataset's built-in `test` split for translation-based eval. TuneMap-specific examples (once validated) serve as a second eval set for both translation + execution-based eval.

---

## R-007: Evaluation Metrics — Two Passes

**Source**: Text2Cypher paper Section 4.1, 4.2, 5.1.

**Pass 1 — Translation-based (all 4,833 test examples)**:
- Metric: Google-BLEU (`sentence_gleu` from nltk)
- Compares generated Cypher text against reference Cypher text
- Penalises textual divergence even if semantically equivalent — acceptable as a signal metric

**Pass 2 — Execution-based (~2,471 test examples with `database_reference` set)**:
- Metric: Exact Match on *result sets*, not on query text
- Both generated and reference Cypher are executed against the target Neo4j database
- Results converted to lexicographically-ordered string representations (normalises row ordering)
- Exact Match is applied to those strings — functionally correct queries pass regardless of how they were written
- Paper (Section 5.1) explicitly: "We used Google-BLEU score for translation-based and Exact Match score for execution-based evaluation."

**Implementation**:
```python
from nltk.translate.gleu_score import sentence_gleu

def compute_gleu(reference: str, hypothesis: str) -> float:
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    return sentence_gleu([ref_tokens], hyp_tokens)

def execution_exact_match(ref_results: list, pred_results: list) -> bool:
    # convert each result set to a sorted string representation
    ref_str = str(sorted(str(r) for r in ref_results))
    pred_str = str(sorted(str(r) for r in pred_results))
    return ref_str == pred_str
```

Metrics reported: mean GLEU (all 4,833), execution exact-match % (2,471 DB-accessible subset).

---

## R-008: Transformers + Unsloth Compatibility

**Decision**: Install `unsloth>=2026.4.7` (PyPI, no extra needed when torch+cu128 is already present). System has CUDA Toolkit 12.8; torch must be installed first via `pip install torch --index-url https://download.pytorch.org/whl/cu128`. For fresh env bootstrap without torch, use `unsloth[cu128-ampere-torch290]` instead.

**Version bounds** sourced from unsloth's own `pyproject.toml` (`huggingfacenotorch` extra). Our requirements.txt must stay within these to avoid pip conflicts:
- `transformers>=4.51.3,!=4.52.0,!=4.52.1,!=4.52.2,!=4.52.3,!=4.53.0,!=4.54.0,!=4.55.0,!=4.55.1,!=4.57.0,!=4.57.4,!=4.57.5,!=5.0.0,!=5.1.0,<=5.5.0`
- `trl>=0.18.2,!=0.19.0,<=0.24.0`
- `peft>=0.18.0,!=0.11.0`
- `datasets>=3.4.1,!=4.0.*,!=4.1.0,<4.4.0`
- `accelerate>=0.34.1`

**Installation** (requirements.txt):
```
# Step 1 (once per env): pip install torch --index-url https://download.pytorch.org/whl/cu128
# Step 2: pip install -r requirements.txt
unsloth>=2026.4.7
transformers>=4.51.3,!=4.52.0,!=4.52.1,!=4.52.2,!=4.52.3,!=4.53.0,!=4.54.0,!=4.55.0,!=4.55.1,!=4.57.0,!=4.57.4,!=4.57.5,!=5.0.0,!=5.1.0,<=5.5.0
trl>=0.18.2,!=0.19.0,<=0.24.0
peft>=0.18.0,!=0.11.0
datasets>=3.4.1,!=4.0.*,!=4.1.0,<4.4.0
nltk>=3.9
accelerate>=0.34.1
neo4j>=5.0
```

**Alternatives considered**: pinned release version — rejected because user explicitly requested latest unsloth.

---

## R-009: Reference Training Config (neo4j paper)

Collected for direct comparison. Their config used BnB 4-bit + paged_adamw_8bit on A100. Our config removes BnB (full bf16), adjusts batch size downward to compensate for higher VRAM footprint.

| Param | neo4j paper | This run |
|---|---|---|
| Model | gemma-2-9b-it | Qwen3.5-9B |
| Quantization | 4-bit BnB | None (full bf16) |
| LoRA r / alpha | 64 / 64 | 64 / 64 |
| Batch size | 4 | 1 (only viable — 22GB on 24GB card) |
| Grad accum | 8 | 32 |
| Effective batch | 32 | 32 |
| LR | 2e-5 | 2e-5 |
| Epochs | 1 | 1 |
| max_seq_length | 1600 | 1600 |
| Optimizer | paged_adamw_8bit | paged_adamw_8bit |

---

---

## R-010: Execution-Based Evaluation — Database Access in Test Set

**Source**: Text2Cypher paper Section 3.3 and 4.2.

**Key numbers**:
- Train set: 22,093 of 39,554 instances have `database_reference` set (55.85%)
- Test set: 2,471 of 4,833 instances have `database_reference` set (51.12%)
- The ~2,471 test examples are the execution-based eval subset

**Database access**: Hosted databases are available via the Neo4jLabs-Crowdsourcing Initiative. The `database_reference` field in the dataset rows routes each query to the correct database endpoint.

**Implication for eval.jsonl**: `database_reference` must be retained as a field — do NOT discard it. The field is needed at eval time to connect to the right database per example.

**Implication for eval.py**: Filter `eval.jsonl` to rows where `database_reference` is non-null, execute both generated and reference Cypher against the target DB, compare result sets as described in R-007.

---

## R-011: Full System Prompt (from Paper Table 3)

**Source**: Text2Cypher paper Table 3 "Instructions used".

**Correction**: The abbreviated system prompt used in earlier plan drafts is incomplete. The paper's actual system prompt includes explicit prohibitions on explanatory text — critical for preventing the model from wrapping Cypher in prose.

**Correct system prompt**:
```
Task: Generate Cypher statement to query a graph database.
Instructions: Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided in the schema.
Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.
```

**Correct user turn** (note the `Cypher output:` suffix):
```
Generate Cypher statement to query a graph database. Use only the provided relationship types and properties in the schema.
Schema: {schema}
Question: {question}
Cypher output:
```

**Why it matters**: The `Cypher output:` suffix primes the model to emit only the query with no preamble. The extended system prompt's prohibitions ("no explanations, no apologies") directly reduce the most common failure mode at inference time.

---

## R-012: Syntax Validation via EXPLAIN (for generate_dataset.py)

**Source**: Text2Cypher paper Section 3.2.

**Method**: "Each Cypher query is checked for syntax errors by running 'EXPLAIN' clauses in a local Neo4j database. Queries that trigger syntax errors are identified and removed."

**How to apply for OI-003**: When validating `generate_dataset.py` output, wrap each candidate query as `EXPLAIN <query>` and execute against the live Neo4j KG. Queries that raise a `CypherSyntaxError` are excluded from the training mix. No need for a custom parser — the database engine is the validator.

---

## Open Items (resolve at implementation)

| ID | Item | Where |
|---|---|---|
| ~~OI-001~~ | ~~Verify language_model_only param~~ | RESOLVED — `FastVisionModel.get_peft_model(finetune_language_layers=True)` |
| ~~OI-002~~ | ~~Confirm per_device_batch_size=2 fits on target GPU~~ | RESOLVED — unsloth docs confirm 22GB for Qwen3.5-9B bf16 LoRA; RTX 3090 has 24GB (2GB headroom); batch=1 is the only option, batch=2 is not viable |
| OI-003 | Validate `generate_dataset.py` output via `EXPLAIN` against live Neo4j before mixing | Dataset pipeline |
| OI-004 | Confirm Neo4jLabs-Crowdsourcing database endpoints reachable for execution-based eval | eval.py implementation |
