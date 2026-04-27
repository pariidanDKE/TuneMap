# Feature Specification: NL-to-Cypher LoRA Fine-Tuning

**Feature Branch**: `001-nl-to-cypher-lora`
**Created**: 2026-04-14
**Status**: Draft
**Scope**: `training/` only — no changes to the main TuneMap application

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Dataset Preparation (Priority: P1)

A developer needs a curated collection of (natural language question, TuneMap schema context,
expected Cypher query) examples that cover all node types, relationship patterns, and the known
Cypher constraint rules of the TuneMap graph, so that there is a clean domain-specific benchmark
to evaluate the adapter against — and so that the external training set is correctly formatted
and ready for fine-tuning.

**Why this priority**: Without a validated domain benchmark and a formatted training set there
is nothing to train on or measure against. Every downstream story depends on this.

**Independent Test**: Both dataset files load cleanly. The TuneMap benchmark contains ≥100
validated examples; every example has a question, a schema block, and a Cypher query that
executes without syntax errors against the live TuneMap graph.

**Acceptance Scenarios**:

1. **Given** no prior dataset exists, **When** the dataset pipeline runs, **Then** it produces
   ≥100 validated (question, schema, Cypher) TuneMap examples spanning all 10 node types and
   all 12 relationship types defined in the TuneMap schema, used exclusively as a domain
   benchmark (evaluation only — no TuneMap rows in the training split).
2. **Given** the TuneMap benchmark, **When** any random sample of 20 Cypher queries is executed
   against a populated TuneMap graph, **Then** all 20 complete without syntax errors.
3. **Given** the external dataset, **When** it is finalised, **Then** the built-in held-out test
   split (4,833 rows) is used as the primary evaluation split, separate from the training split.
4. **Given** the dataset, **When** it is reviewed, **Then** no TuneMap benchmark example contains
   a Cypher pattern that violates the documented schema rules (e.g., implicit GROUP BY, cartesian
   products, property-access RETURN for graph visualisation queries).

---

### User Story 2 — Fine-Tuning Run (Priority: P2)

A developer needs to run LoRA fine-tuning against the prepared dataset and receive a serialised
adapter checkpoint on disk, so that there is a model specialised for TuneMap Cypher generation
that can be evaluated and eventually served.

**Why this priority**: The adapter is the primary deliverable of this feature.

**Independent Test**: A fine-tuning run can be launched, completes without crashing, and writes
a checkpoint directory. The checkpoint can be loaded and used to generate a Cypher query for a
sample question.

**Acceptance Scenarios**:

1. **Given** a prepared training split, **When** fine-tuning is launched, **Then** it completes
   and writes a LoRA adapter checkpoint to `training/` without modifying any file outside that
   directory.
2. **Given** the adapter checkpoint, **When** it is loaded and a test question is passed through
   it, **Then** the output is a Cypher query with no surrounding explanation text or markdown
   fences.
3. **Given** an interruption (crash, OOM, manual stop), **When** fine-tuning is restarted,
   **Then** it can resume from the last saved checkpoint without restarting from scratch.

---

### User Story 3 — Evaluation (Priority: P3)

A developer needs to measure the fine-tuned adapter's Cypher generation quality on the
held-out evaluation split, comparing it to the baseline (same model without the adapter), so
that it is clear whether fine-tuning produced a measurable improvement.

**Why this priority**: Confirms whether the adapter is worth using before any integration work.

**Independent Test**: The evaluation script runs end-to-end on the eval split and produces a
metrics report with at least: GLEU on all eval rows and TuneMap syntax validity rate.

**Acceptance Scenarios**:

1. **Given** an adapter checkpoint and the held-out eval split, **When** the evaluation script
   runs, **Then** it produces a report showing GLEU score and TuneMap syntax validity rate for
   both the fine-tuned adapter and the baseline model.
2. **Given** the evaluation report, **When** adapter and baseline are compared, **Then** the
   adapter achieves a lower Cypher syntax error rate on TuneMap-specific questions.

---

### Edge Cases

- What happens when a generated Cypher example violates schema rules (GROUP BY, cartesian
  product)? A validation step must catch and discard these during dataset construction.
- What happens if the model outputs markdown fences or explanation text instead of raw Cypher
  during evaluation? Stripping logic must be applied consistently and identically to both the
  adapter and baseline outputs.
- What if the GPU runs out of memory mid-training? The last checkpoint must be preserved so
  fine-tuning can resume.
- What happens when a Cypher query is syntactically valid but returns empty results on the
  test graph? This is tracked separately as an execution-quality metric, not a syntax failure.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The dataset pipeline MUST produce ≥200 training examples covering all 10 node
  types in the TuneMap schema.
- **FR-002**: The dataset MUST include examples that exercise each documented Cypher constraint
  rule: no implicit GROUP BY, no cartesian products, correct string-quote delimiters, bare-variable
  RETURN for graph visualisation patterns.
- **FR-003**: The training split MUST use only the external dataset (neo4j/text2cypher-2024v1
  train split, ~39K rows). The TuneMap-specific examples MUST be held out entirely as a
  domain benchmark evaluation set — no TuneMap rows may appear in the training split.
- **FR-004**: Fine-tuning MUST produce a LoRA adapter checkpoint stored in a portable format
  that can be loaded without re-running training.
- **FR-005**: The adapter MUST be compatible with the serving format used by the existing
  TuneMap vLLM inference pipeline so it can be loaded without infrastructure changes.
- **FR-006**: Fine-tuning MUST support checkpoint resumption if the run is interrupted.
- **FR-007**: The evaluation pipeline MUST be split into two independent scripts.
  `translation_eval.py` applies translation-based evaluation (GLEU) across all held-out eval
  rows (4,833) and writes `translation_report.json`. `execution_eval.py` applies
  execution-based evaluation against the live TuneMap AuraDB instance for rows tagged
  `source="tunemap"` (~120 rows) — syntax validation via `EXPLAIN` and Exact Match on result
  sets — and writes `execution_report.json`. No execution eval is performed against external
  database references. Both scripts accept a `--checkpoint` arg and include a baseline pass.
- **FR-008**: All code, datasets, checkpoints, and evaluation outputs MUST reside exclusively
  within `training/`. No file outside `training/` may be created or modified.

### Key Entities

- **Training Example**: A tuple of (natural language question, schema context string, expected
  Cypher query). The atomic unit of the dataset.
- **LoRA Adapter**: The fine-tuned weight delta saved as a portable checkpoint. Must be loadable
  by the existing vLLM serving setup without other changes.
- **Evaluation Report**: A structured summary of adapter vs. baseline performance on the held-out
  split: syntax error rate, execution success rate, and any representative failure examples.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: When the adapter is loaded into the existing query pipeline without any other
  changes, it generates syntactically valid Cypher for ≥90% of eval-set questions (no Neo4j
  syntax errors on execution).
- **SC-002**: The adapter achieves a lower Cypher syntax error rate than the baseline (same
  model, no adapter) on the TuneMap-specific eval set, with the improvement documented in the
  evaluation report.
- **SC-003**: A developer can run the full pipeline (dataset → fine-tuning → evaluation) from
  a single documented entry point, with all steps completing successfully on reference hardware.
- **SC-004**: All training artefacts (dataset, adapter checkpoint, evaluation report) are
  reproducible given the same seed and inputs.

## Assumptions

- The base model for fine-tuning is from the same model family as the currently deployed vLLM
  model (Qwen3.5 series) to ensure adapter compatibility with the existing inference setup.
- The TuneMap graph schema (nodes, relationships, property names, Cypher constraint rules) as
  defined in `query_engine.py` is stable for the duration of this feature.
- A GPU environment with sufficient VRAM (≥16 GB) is available for fine-tuning.
- A running Neo4j instance with a populated TuneMap graph is available for Cypher execution
  validation during dataset construction and evaluation.
- vLLM LoRA serving configuration (e.g., `--enable-lora` flags, serving the adapter via the
  OpenAI-compatible API) is out of scope for this feature; only producing the adapter weights
  is in scope.

## Open Questions

Training uses only the external NL-to-Cypher dataset (neo4j/text2cypher-2024v1, ~39K rows).
The TuneMap-specific examples (~120 validated rows, generated by `generate_datasets.py`) are
held out entirely as a domain benchmark — they are never mixed into the training split. At
~120 rows they are too few to provide meaningful training signal, but they are a near-perfect
representative test set for the target domain: real questions, real schema, validated Cypher,
executable against the live graph.

Evaluation uses two complementary strategies:

- **External held-out split (4,833 rows)** — translation-based evaluation only (GLEU).
  Generated Cypher is compared against reference Cypher on textual content; the external KG
  is not available for live execution.

- **TuneMap domain benchmark (~120 rows)** — both translation-based (GLEU) and
  execution-based evaluation (Exact Match on result sets against the live TuneMap Neo4j graph).
  This is the definitive quality signal for the target use case.
