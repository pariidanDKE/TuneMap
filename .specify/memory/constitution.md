<!--
SYNC IMPACT REPORT
==================
Version change: 1.0.0 → 1.0.1 (PATCH — clarification added to Principle II)
Modified principles:
  - II. Source-First Documentation for Evolving Libraries: added known API gotcha for
    unsloth multimodal models (FastModel vs FastLanguageModel)
Added sections: none
Removed sections: none
Templates requiring updates:
  ✅ .specify/templates/plan-template.md — no changes required
  ✅ .specify/templates/spec-template.md — no changes required
  ✅ .specify/templates/tasks-template.md — no changes required
Deferred TODOs: none
-->

# TuneMap Constitution

## Core Principles

### I. Training Module Isolation

All fine-tuning and model-training code MUST reside exclusively within the `training/` directory.
Training work MUST NOT modify, import into, or couple with any existing application code
(`app.py`, agents, parsers, graph ingestion, Streamlit UI, etc.). The `training/` folder is a
self-contained workspace: it may read from shared data artefacts (e.g., exported graph data)
but MUST NOT write back to any path outside `training/` during development or execution.

**Rationale**: The main TuneMap application is stable. Isolating fine-tuning experiments
prevents regressions, keeps dependency trees separate (ML-heavy deps vs. app deps), and
allows the training work to evolve independently without requiring app-level review or
coordination.

**Constitution Gate (plan/tasks check)**: Confirm all new file paths, imports, and outputs
are scoped to `training/`. Flag any cross-boundary reference as a violation requiring explicit
justification.

### II. Source-First Documentation for Evolving Libraries

When working with rapidly evolving or less-documented ML packages — including but not limited
to `unsloth`, `transformers`, `trl`, `peft`, and `bitsandbytes` — the agent MUST verify
exact method signatures, parameter names, and default values by reading the installed
source directly in site-packages before writing or reviewing code.

The agent SHOULD also use the context7 MCP tool for additional documentation lookup,
especially when site-packages navigation is ambiguous or the relevant class is deeply nested.

**Rationale**: PyPI documentation and LLM training data lag behind library releases.
Unsloth in particular changes APIs frequently across minor versions. Using installed source
as the ground truth prevents subtle bugs from stale parameter assumptions.

**How to apply**: Before implementing any call to an evolving library, locate the relevant
class or function in site-packages (e.g., `~/.venv/lib/python3.x/site-packages/<pkg>/`)
and confirm the signature matches the intended usage.

**Known unsloth API distinction**: When working with multimodal models (e.g., vision-language
models), MUST use `FastModel` (from `unsloth`) instead of `FastLanguageModel`. `FastLanguageModel`
is for text-only models; using it with a multimodal model will fail or silently degrade.
Verify the correct entry point in site-packages before loading any model.

### III. Simplicity Over Abstraction

Code MUST be written to solve the immediate, concrete problem. Generic, reusable, or
configurable designs are a bonus — they MUST NOT be introduced when they add complexity,
verbosity, or risk of bugs.

Specific rules:
- Do NOT create abstraction layers, base classes, or utility modules for one-time use.
- Do NOT add configuration flags or pluggable components for hypothetical future requirements.
- When a simpler direct implementation exists, prefer it over the elegant-but-complex one.
- If an abstraction is already causing bugs or making the code harder to follow, simplify
  immediately — do not patch around it.
- Three explicit lines are better than a premature helper.

**Rationale**: The training work involves iterative experimentation. Clarity and debuggability
matter more than architectural elegance at this stage. Over-engineering in an experimental
context compounds debt without delivering value.

## Technical Stack & Scope

**Application stack (read-only for training work)**:
Python 3.12+, Streamlit, Neo4j 5, LlamaIndex, vLLM / OpenAI-compatible API, neovis.js.

**Training stack (lives in `training/`)**:
Python 3.12+, unsloth, transformers, trl, peft, bitsandbytes, datasets.
GPU environment assumed (16 GB+ VRAM); CPU fallback is out of scope.

**Scope boundary**: The `training/` folder produces fine-tuned model weights or LoRA
adapters as artefacts. Integration of those artefacts back into the app is a separate
future concern and MUST be specified as a new feature, not assumed as part of training work.

## Agent Workflow Guidance

- Before implementing any training-related code, verify the working directory and file
  paths are within `training/`. Do not create files at the project root or in app directories.
- Before calling any `unsloth`, `transformers`, or `trl` API, read the relevant source
  in site-packages to confirm the method signature. Do not rely on training-data knowledge
  alone for these packages.
- When in doubt between a clean-but-simple solution and a flexible-but-complex one,
  choose simple. Ask the user only if the simpler path has a genuine functional gap.
- Constitution Check in every plan MUST explicitly confirm Principle I (isolation) and
  Principle III (no unnecessary abstraction) before any implementation begins.

## Governance

This constitution supersedes all other informal practices for the TuneMap project.
Amendments require: (1) a clear rationale tied to a concrete problem encountered,
(2) a version bump per the semantic versioning policy below, and (3) this file updated
before any implementation that depends on the new guidance.

**Versioning policy**:
- MAJOR: Removal or redefinition of an existing principle (backward-incompatible governance change).
- MINOR: New principle or section added, or materially expanded guidance.
- PATCH: Clarifications, wording fixes, non-semantic refinements.

**Compliance review**: Every implementation plan (`plan.md`) MUST include a Constitution
Check section that gates Phase 0 research and is re-checked after Phase 1 design. Any
violation MUST be documented in the plan's Complexity Tracking table with justification.

**Version**: 1.0.1 | **Ratified**: 2026-04-13 | **Last Amended**: 2026-04-14
