# Implementation Plan: NL-to-Cypher LoRA Fine-Tuning

**Branch**: `cyper_finetune` | **Date**: 2026-04-15 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-nl-to-cypher-lora/spec.md`

---

## Summary

Fine-tune `Qwen/Qwen3.5-9B` with LoRA (no quantization, full bf16) on the `neo4j/text2cypher-2024v1` dataset supplemented with a small TuneMap-specific dataset, producing a portable LoRA adapter checkpoint in `training/`. Evaluation runs two passes: translation-based (Google-BLEU on all 4,833 test examples) and execution-based (Exact Match on result sets for the ~2,471 test examples that have live database access). All artefacts are scoped exclusively to `training/`.

---

## Technical Context

**Language/Version**: Python 3.12+
**Primary Dependencies**: torch+cu128 (pre-installed), unsloth>=2026.4.7, transformers>=5.0.0, trl>=0.17, peft>=0.15, datasets>=3.0, nltk>=3.9, accelerate>=1.0
**Storage**: `training/` only — no writes outside this directory
**Testing**: Manual eval script (`eval.py`) — two passes: translation-based GLEU (4,833 examples) + execution-based Exact Match on result sets (~2,471 DB-accessible examples); no pytest suite
**Target Platform**: Linux, single GPU (RTX 3090)
**Project Type**: ML training pipeline (script-based, not a library)
**Performance Goals**: Complete 1 epoch over ~39K examples; adapter must load and generate clean Cypher
**Constraints**: Effective batch size = 32; max_seq_length = 1600; no BnB quantization (unsloth docs explicitly state QLoRA is not recommended for Qwen3.5); VRAM: 22GB required (confirmed by unsloth docs), RTX 3090 has 24GB — 2GB headroom, batch=1 is the only viable option
**Scale/Scope**: ~39,554 external training examples + ~100 TuneMap-specific; single-epoch run; one checkpoint per run

---

## Constitution Check

*GATE: Must pass before implementation begins. Re-checked after Phase 1 design.*

### Principle I — Training Module Isolation ✅

All new files reside exclusively in `training/`. No imports from or writes to application code (`app.py`, agents, parsers, `query_engine.py`, Streamlit UI). `generate_dataset.py` (already in `training/`) reads the hardcoded schema but does not import from application modules.

**Violation**: None.

### Principle II — Source-First Documentation ✅

For text-only fine-tuning, `FastLanguageModel` is correct per the unsloth Qwen3.5 docs text-only section. Vision encoder is excluded by omitting it from `target_modules`. `load_in_16bit=True` confirmed. `use_gradient_checkpointing` placement confirmed in `get_peft_model`. No site-packages verification gate required; source is the official unsloth docs.

### Principle III — Simplicity Over Abstraction ✅

No base classes, no plugin architecture, no config files. Three flat scripts: `prepare_data.py`, `train.py`, `eval.py`. Direct calls to unsloth, trl, and nltk APIs.

**Violation**: None.

---

## Project Structure

### Documentation (this feature)

```text
specs/001-nl-to-cypher-lora/
├── plan.md              ← this file
├── research.md          ← Phase 0 output
├── data-model.md        ← Phase 1 output
└── tasks.md             ← Phase 2 output (/speckit.tasks — not yet created)
```

### Source Code

All new files in `training/` only:

```text
training/
├── generate_dataset.py     # existing — TuneMap-specific data generator (NOT yet validated)
├── prepare_data.py         # NEW — download + merge datasets, format chat turns, train/eval split
├── train.py                # NEW — LoRA fine-tuning script (unsloth + trl SFTTrainer)
├── eval.py                 # NEW — GLEU + exact-match evaluation against held-out split
├── requirements.txt        # NEW — pinned deps for training environment
└── data/
    ├── cypher_dataset.jsonl       # TuneMap-specific output of generate_dataset.py
    ├── train.jsonl                # merged + formatted training split
    └── eval.jsonl                 # held-out evaluation split
```

**Checkpoint output** (written by SFTTrainer, not committed to git):
```text
training/outputs/
├── checkpoint-XXXX/     # intermediate checkpoints (save_steps interval)
├── final_adapter/       # LoRA adapter weights only (save_pretrained)
└── merged_16bit/        # merged bf16 model for vLLM (save_pretrained_merged)
```

**vLLM export** (for TuneMap integration, future concern):
```python
model.save_pretrained_merged("training/outputs/merged_16bit", tokenizer, save_method="merged_16bit")
```

**Known limitation — reasoning degradation**: Qwen3.5 is a reasoning model. Fine-tuning on 100% direct-answer Cypher data (no chain-of-thought traces) will degrade the model's reasoning capability. The unsloth docs recommend keeping ≥75% reasoning examples to preserve it. For NL-to-Cypher this is an accepted tradeoff — Cypher generation does not require reasoning traces.

**Structure Decision**: Flat script layout under `training/`. No packages, no `src/` indirection. Three scripts mirror the three pipeline stages: data → train → eval.

---

## Complexity Tracking

> No constitution violations requiring justification.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|--------------------------------------|
| — | — | — |

---

## Phase 0: Research Summary

See [research.md](research.md) for full findings. Key resolved items:

| Item | Resolution |
|------|------------|
| Text-only API | `FastLanguageModel` (not `FastVisionModel`) — vision encoder excluded via `target_modules` |
| No-quantization LoRA | `load_in_4bit=False`, `load_in_16bit=True` (native bf16); no dtype arg needed |
| Paged optimizer | `paged_adamw_8bit` — confirmed from neo4j reference training config |
| Effective batch size 32 | `per_device_train_batch_size=1`, `gradient_accumulation_steps=32` (batch=1 only — 22GB model on 24GB card) |
| Dataset schema | `question` + `schema` → user turn; `cypher` → assistant turn; `database_reference` retained for exec eval |
| Prompt format | Full system prompt (Table 3) + user turn ending with `Cypher output:` suffix |
| Eval pass 1 | Translation-based: `sentence_gleu` on all 4,833 test examples |
| Eval pass 2 | Execution-based: Exact Match on result sets for ~2,471 examples with `database_reference` set |
| Syntax validation | Use `EXPLAIN <query>` on live Neo4j to validate `generate_dataset.py` output (R-012) |
| Transformers compat | bounds from unsloth pyproject.toml: `>=4.51.3,!=5.0.0,!=5.1.0,<=5.5.0`; torch+cu128 pre-installed |
| `generate_dataset.py` | Exists, not validated — treat TuneMap data as supplementary, validate via EXPLAIN before mixing |

---

## Phase 1: Design

### Training Configuration

```python
# train.py — confirmed hyperparameters
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3.5-9B",
    max_seq_length=1600,
    load_in_4bit=False,
    load_in_16bit=True,                # native bf16; correct unsloth param
    full_finetuning=False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=64,                              # docs show r=16; r=64 from neo4j paper — deliberate
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # here, not in from_pretrained
    random_state=3407,
    max_seq_length=1600,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    args=SFTConfig(
        per_device_train_batch_size=1,     # only viable option: 22GB model on 24GB card
        gradient_accumulation_steps=32,    # → effective batch = 32
        num_train_epochs=1,
        learning_rate=2e-5,
        max_seq_length=1600,
        bf16=True,
        fp16=False,
        optim="paged_adamw_8bit",          # docs show adamw_8bit; paged kept for CPU offload on 3090
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        output_dir="training/outputs",
        seed=3407,
        dataset_num_proc=1,
    ),
)
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)
```

### Prompt Format

Following the exact instructions from the neo4j paper (Table 3), wrapped in chatml turns:

```
<|im_start|>system
Task: Generate Cypher statement to query a graph database.
Instructions: Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided in the schema.
Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.<|im_end|>
<|im_start|>user
Generate Cypher statement to query a graph database. Use only the provided relationship types and properties in the schema.
Schema: {schema}
Question: {question}
Cypher output:<|im_end|>
<|im_start|>assistant
{cypher}<|im_end|>
```

The `Cypher output:` suffix on the user turn primes the model to emit only the query. The extended system prompt's prohibitions prevent the model from wrapping the output in prose or explanations.

### Data Pipeline (prepare_data.py)

1. Load `neo4j/text2cypher-2024v1` train split (39,554 rows)
2. Load TuneMap-specific data from `training/data/cypher_dataset.jsonl` (if validated; skip otherwise)
3. Format each row → 3-turn chatml conversation
4. Merge: external train + TuneMap-specific train rows
5. Shuffle with fixed seed (3407)
6. Write `training/data/train.jsonl`
7. Write `training/data/eval.jsonl` from external `test` split (4,833 rows) — TuneMap-specific examples added to eval only after validation

### Evaluation Script (eval.py)

**Pass 1 — Translation-based (all 4,833 examples)**:
1. Load adapter checkpoint + base model
2. Load `training/data/eval.jsonl`
3. For each example: generate Cypher (greedy, `max_new_tokens=256`)
4. Strip any markdown fences from output
5. Compute `sentence_gleu` per example
6. Report: mean GLEU, 10 worst-GLEU failures

**Pass 2 — Execution-based (~2,471 examples)**:
7. Filter eval.jsonl to rows where `database_reference` is non-null
8. For each: execute generated Cypher and reference Cypher against the target Neo4j database (routed via `database_reference`)
9. Convert both result sets to lexicographically-ordered string representations
10. Compute Exact Match on those strings
11. Report: execution exact-match %

**Combined output** → `training/outputs/eval_report.json`:
- `mean_gleu` (translation pass, n=4,833)
- `exec_exact_match_pct` (execution pass, n=~2,471)
- `per_example` records with both metrics where applicable
- Re-run on TuneMap-specific eval subset separately when available

---

## Post-Phase-1 Constitution Re-check

- **Principle I**: All scripts, data files, checkpoints remain in `training/`. ✅
- **Principle II**: `FastLanguageModel` confirmed from official Qwen3.5 unsloth docs text-only section. `load_in_16bit=True`, `use_gradient_checkpointing` placement, `target_modules` list all confirmed from docs. ✅
- **Principle III**: Three flat scripts, no abstraction layers. Data formatted inline in `prepare_data.py`, no custom dataset class. Two-pass eval is sequential logic in one script, not a new abstraction. ✅
