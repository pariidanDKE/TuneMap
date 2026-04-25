# Tasks: NL-to-Cypher LoRA Fine-Tuning

**Input**: Design documents from `specs/001-nl-to-cypher-lora/`
**Branch**: `cyper_finetune` | **Date**: 2026-04-23

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: User story this task belongs to ([US1], [US2], [US3])
- No test tasks — spec does not request TDD approach; acceptance is manual pipeline validation

---

## Phase 1: Setup

**Purpose**: Create the environment prerequisites that every script depends on.

- [x] T001 Create `training/requirements.txt` with all deps: `unsloth>=2026.4.7`, `transformers>=5.0.0`, `trl>=0.17`, `peft>=0.15`, `datasets>=3.0`, `nltk>=3.9`, `accelerate>=1.0`, `neo4j>=5.0` (for execution eval driver). Add a comment at the top of the file: torch must be installed first via `pip install torch --index-url https://download.pytorch.org/whl/cu128` — the unsloth extra (`unsloth[cu128-ampere-torch290]`) is only needed if bootstrapping a fresh env without torch already present.

**Checkpoint**: `pip install -r training/requirements.txt` completes without error (assumes torch+cu128 already installed).

---

## Phase 2: Foundational — TuneMap Data Validation Gate

**Purpose**: Determine whether `generate_dataset.py` output is safe to mix into training. This gate result flows directly into US1 — `prepare_data.py` will either include or skip TuneMap data based on the outcome.

**⚠️ CRITICAL**: Must complete before US1 implementation begins. Outcome informs T004.

- [x] T002 Run `training/generate_dataset.py` against the live Neo4j instance and write output to `training/data/cypher_dataset.jsonl`. Then validate each generated Cypher by running `EXPLAIN <query>` via the Neo4j Python driver — log any `CypherSyntaxError` rows. Produce a cleaned file `training/data/cypher_dataset_validated.jsonl` containing only syntax-error-free rows.

**Checkpoint**: `training/data/cypher_dataset_validated.jsonl` exists. If it contains ≥10 rows, the TuneMap data gate is OPEN and T004 will include it. If empty or absent, T004 skips TuneMap data.

---

## Phase 3: User Story 1 — Dataset Preparation (Priority: P1) 🎯 MVP

**Goal**: Produce `training/data/train.jsonl` and `training/data/eval.jsonl` in chatml format, ready for SFTTrainer, with `database_reference` retained in eval rows.

**Independent Test**: Both JSONL files load cleanly. Each row has a `conversations` list with exactly 3 turns (system/user/assistant). User turn ends with `Cypher output:`. Assistant turn contains no markdown fences. `database_reference` field is present (possibly null) in eval rows. A random sample of 20 `assistant` values are syntactically valid Cypher (spot-check via EXPLAIN).

### Implementation

- [x] T003 [US1] Implement download and chatml formatting in `training/prepare_data.py`: load `neo4j/text2cypher-2024v1` train split (39,554 rows) via `datasets`, format each row as a 3-turn conversation using the full system prompt from the paper (Table 3) and user turn ending with `Cypher output:`. Retain `database_reference` and `source` fields alongside `conversations`.

- [x] T004 [US1] Add TuneMap merge logic to `training/prepare_data.py`: if `training/data/cypher_dataset_validated.jsonl` exists (gate from T002), load it and append rows to the train set (targeting 90/10 train/eval split from TuneMap data). If file absent, skip silently with a log line. Merge external + TuneMap rows, shuffle with `seed=3407`.

- [x] T005 [US1] Add split and write logic to `training/prepare_data.py`: write the shuffled merged set to `training/data/train.jsonl` and the external `test` split (4,833 rows, with `database_reference` retained) to `training/data/eval.jsonl`. Print row counts on completion.

- [x] T006 [US1] Run `python training/prepare_data.py` and manually verify: (a) `train.jsonl` has ≥39,554 rows, (b) `eval.jsonl` has 4,833 rows, (c) spot-check 5 rows from each for correct chatml structure and `Cypher output:` suffix on user turn, (d) confirm ~2,471 eval rows have non-null `database_reference`.

**Checkpoint**: User Story 1 complete — `train.jsonl` and `eval.jsonl` exist and pass spot-checks.

---

## Phase 4: User Story 2 — Fine-Tuning Run (Priority: P2)

**Goal**: Produce a LoRA adapter checkpoint in `training/outputs/` that can be loaded and used to generate Cypher. Also produce a merged bf16 model for vLLM integration.

**Independent Test**: `training/outputs/final_adapter/` exists and contains adapter config + weights. Load it with `FastLanguageModel.from_pretrained` + `load_adapter`, pass one test question, confirm output is raw Cypher with no prose or markdown fences.

### Implementation

- [x] T007 [US2] Implement model loading in `training/train.py`: `FastLanguageModel.from_pretrained("Qwen/Qwen3.5-9B", max_seq_length=1600, load_in_4bit=False, load_in_16bit=True, full_finetuning=False)`. Add NLTK `punkt` download at top of script.

- [x] T008 [US2] Implement LoRA configuration in `training/train.py`: `FastLanguageModel.get_peft_model` with `r=64`, `target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]`, `lora_alpha=64`, `lora_dropout=0`, `bias="none"`, `use_gradient_checkpointing="unsloth"`, `random_state=3407`, `max_seq_length=1600`.

- [x] T009 [US2] Implement dataset loading and `SFTTrainer` in `training/train.py`: load `training/data/train.jsonl`, apply chatml chat template via `get_chat_template`, configure `SFTTrainer` with `SFTConfig(per_device_train_batch_size=1, gradient_accumulation_steps=32, num_train_epochs=1, learning_rate=2e-5, max_seq_length=1600, bf16=True, fp16=False, optim="paged_adamw_8bit", lr_scheduler_type="linear", warmup_ratio=0.1, logging_steps=5, save_steps=50, save_total_limit=2, output_dir="training/outputs", seed=3407, dataset_num_proc=1)`.

- [x] T010 [US2] Wrap trainer with `train_on_responses_only` in `training/train.py`: `instruction_part="<|im_start|>user\n"`, `response_part="<|im_start|>assistant\n"`. Call `trainer.train()`.

- [x] T011 [US2] Add checkpoint saving to `training/train.py`: after training completes, call `model.save_pretrained("training/outputs/final_adapter")` and `tokenizer.save_pretrained("training/outputs/final_adapter")`. Add `model.save_pretrained_merged("training/outputs/merged_16bit", tokenizer, save_method="merged_16bit")` for vLLM integration.

- [ ] T012 [US2] Run `python training/train.py`. Monitor VRAM usage at start — if OOM, the 2GB headroom has been exceeded (investigate gradient checkpointing or reduce max_seq_length). On completion: verify `training/outputs/final_adapter/` exists, load adapter, generate Cypher for one test question (e.g. "What are all tracks by The Beatles?"), confirm output is raw Cypher only.

**Checkpoint**: User Story 2 complete — adapter checkpoint exists, loads cleanly, generates raw Cypher.

---

## Phase 5: User Story 3 — Evaluation (Priority: P3)

**Goal**: Produce `training/outputs/eval_report.json` comparing adapter vs. baseline on translation-based (GLEU, 4,833 examples) and execution-based (Exact Match on result sets, ~2,471 examples) metrics. Confirm adapter outperforms baseline on TuneMap-specific questions.

**Independent Test**: `training/outputs/eval_report.json` exists. Contains `mean_gleu`, `exec_exact_match_pct`, and a `baseline` section with the same metrics. Report shows adapter syntax validity ≥90% on TuneMap-specific questions (SC-001) and lower syntax error rate than baseline (SC-002).

### Implementation

- [ ] T013 [US3] Implement Pass 1 (translation-based) in `training/eval.py`: load adapter checkpoint + base model, load `training/data/eval.jsonl`, for each example generate Cypher (greedy decode, `max_new_tokens=256`), strip markdown fences, compute `sentence_gleu` from `nltk.translate.gleu_score`. Aggregate `mean_gleu` over all 4,833 rows. Store `per_example` list with question, reference_cypher, predicted_cypher, gleu.

- [ ] T014 [US3] Implement Pass 2 (execution-based) in `training/eval.py`: filter eval rows to those where `database_reference` is non-null (~2,471 rows), execute both generated Cypher and reference Cypher against the target Neo4j database (routed by `database_reference` field), convert both result sets to `str(sorted(str(r) for r in results))`, compute `exec_exact_match`. Aggregate `exec_exact_match_pct`.

- [ ] T015 [US3] Add TuneMap-specific syntax validation to `training/eval.py`: for eval rows tagged `source="tunemap"`, run generated Cypher via `EXPLAIN` against the live TuneMap Neo4j graph. Track `tunemap_syntax_valid_pct` as an additional metric in the report.

- [ ] T016 [US3] Add baseline comparison to `training/eval.py`: re-run Pass 1 and Pass 2 with the base model loaded without the adapter. Write both adapter and baseline metric blocks to `training/outputs/eval_report.json`. Schema: `{run_id, checkpoint, n_translation, n_execution, mean_gleu, exec_exact_match_pct, tunemap_syntax_valid_pct, baseline: {mean_gleu, exec_exact_match_pct, tunemap_syntax_valid_pct}, per_example: [...], failures: [...]}`.

- [ ] T017 [US3] Run `python training/eval.py --checkpoint training/outputs/final_adapter`. Verify report is produced. Confirm SC-001 (≥90% syntax valid on TuneMap subset) and SC-002 (adapter `mean_gleu` and `exec_exact_match_pct` higher than baseline). Record results.

**Checkpoint**: User Story 3 complete — eval report exists with both adapter and baseline metrics.

---

## Phase 6: Polish & Cross-Cutting Concerns

- [ ] T018 Verify full pipeline runs end-to-end from scratch: `prepare_data.py` → `train.py` → `eval.py` with no manual steps in between. Document the single entry-point invocation in a comment block at the top of each script (SC-003).

- [ ] T019 Confirm constitution compliance — audit all created files are within `training/`: run `find . -newer training/requirements.txt -not -path "./training/*" -not -path "./.git/*"` and confirm empty output (FR-008).

- [ ] T020 Verify checkpoint resume (FR-006, US2 AS3): kill a fresh training run mid-epoch, restart with same command, confirm it picks up from the last `checkpoint-XXXX/` and does not restart from step 0.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Phase 1. Outcome (gate open/closed) informs T004
- **US1 (Phase 3)**: Depends on Phases 1 and 2 — T004 behaviour set by T002 result
- **US2 (Phase 4)**: Depends on US1 completion — requires `train.jsonl`
- **US3 (Phase 5)**: Depends on US2 completion — requires adapter checkpoint and `eval.jsonl`
- **Polish (Phase 6)**: Depends on US1 + US2 + US3

### Within Each User Story

- T003 → T004 → T005 → T006 (sequential within US1, same file)
- T007 → T008 → T009 → T010 → T011 → T012 (sequential within US2, same file)
- T013 → T014 → T015 → T016 → T017 (sequential within US3, same file)

### Parallel Opportunities

This is a sequential pipeline (data → train → eval). No meaningful intra-story parallelism — all tasks within a story touch the same file. US stories themselves are also sequential by data dependency.

---

## Implementation Strategy

### MVP (User Story 1 + 2 only)

1. T001 — requirements.txt
2. T002 — validate generate_dataset.py
3. T003–T006 — prepare_data.py → `train.jsonl` + `eval.jsonl`
4. T007–T012 — train.py → adapter checkpoint
5. **STOP and VALIDATE**: load adapter, generate sample Cypher, confirm it works
6. Skip US3 until adapter quality is confirmed worth evaluating

### Full Delivery

Complete all phases in order. Each phase produces a concrete artefact that can be independently verified before proceeding.

---

## Notes

- All files MUST be created under `training/` — constitution Principle I
- Verify unsloth API calls against installed source before implementation — constitution Principle II
- No abstraction layers or helper modules — constitution Principle III
- VRAM is tight (22GB model on 24GB card). If OOM at T012: first try reducing `max_seq_length` from 1600 → 1024; if still OOM, open a discussion on the no-quantization constraint before adding BnB
- OI-004 (database endpoints for execution-based eval) must be confirmed reachable before T014
