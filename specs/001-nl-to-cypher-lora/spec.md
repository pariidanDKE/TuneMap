# Feature Specification: NL-to-Cypher LoRA Fine-Tuning

**Feature Branch**: `001-nl-to-cypher-lora`
**Created**: 2026-04-14
**Status**: Draft
**Scope**: `training/` only — no changes to the main TuneMap application

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Dataset Preparation (Priority: P1)

A developer needs a curated collection of (natural language question, TuneMap schema context,
expected Cypher query) examples that cover all node types, relationship patterns, and the known
Cypher constraint rules of the TuneMap graph, so that fine-tuning has clean, representative
signal.

**Why this priority**: Without a high-quality dataset there is nothing to fine-tune on. Every
downstream story depends on this.

**Independent Test**: The dataset file can be loaded and spot-checked — every example contains
a question, a schema block, and a Cypher query. A random sample of 20 Cyphers executes against
a test graph instance without syntax errors.

**Acceptance Scenarios**:

1. **Given** no prior dataset exists, **When** the dataset pipeline runs, **Then** it produces
   ≥200 (question, schema, Cypher) examples spanning all 10 node types and all 12 relationship
   types defined in the TuneMap schema.
2. **Given** the full dataset, **When** any random sample of 20 Cypher queries is executed
   against a populated test graph, **Then** all 20 complete without syntax errors.
3. **Given** the full dataset, **When** it is finalised, **Then** at least 20% of examples are
   held out as an evaluation split, separate from the training split.
4. **Given** the dataset, **When** it is reviewed, **Then** no example contains a Cypher
   pattern that violates the documented schema rules (e.g., implicit GROUP BY, cartesian
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
metrics report with at least: Cypher syntax validity rate and query execution success rate.

**Acceptance Scenarios**:

1. **Given** an adapter checkpoint and the held-out eval split, **When** the evaluation script
   runs, **Then** it produces a report showing syntax error rate and execution success rate for
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
- **FR-003**: The dataset MUST be partitioned into a training split and a held-out evaluation
  split (≥20% eval) before any fine-tuning run.
- **FR-004**: Fine-tuning MUST produce a LoRA adapter checkpoint stored in a portable format
  that can be loaded without re-running training.
- **FR-005**: The adapter MUST be compatible with the serving format used by the existing
  TuneMap vLLM inference pipeline so it can be loaded without infrastructure changes.
- **FR-006**: Fine-tuning MUST support checkpoint resumption if the run is interrupted.
- **FR-007**: The evaluation pipeline MUST apply translation-based evaluation (generated Cypher
  vs. reference Cypher, text comparison) across both dataset splits. For the TuneMap-specific
  split it MUST additionally apply execution-based evaluation: the query is run against the
  live graph, the model produces a natural language response from the result, and that response
  is compared against a labelled reference answer.
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

Training data uses an existing external NL-to-Cypher dataset as the primary source, augmented
with a small TuneMap-specific set (~100 rows) generated by an existing `generate_datasets.py`
script. The TuneMap-specific examples serve as a domain-adaptation "end phase" to ensure the
adapter learns the exact schema, node types, and Cypher constraint rules of this graph.

Evaluation uses two complementary strategies depending on which dataset split is being assessed:

- **Pre-existing external dataset** — translation-based evaluation only. Generated Cypher
  queries are compared against reference Cypher queries on textual content (the external KG
  is not available for live execution, so no execution-based check is possible here).

- **TuneMap-specific dataset** — both translation-based and execution-based evaluation.
  Queries are executed against the live TuneMap Neo4j graph; the model reads the result and
  produces a natural language response, which is then compared against a labelled reference
  response. This allows quality to be assessed end-to-end, not just at the Cypher level.
