# AppleMusciKG Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-04-15

## Active Technologies

- Python 3.12+ + torch+cu128 (pre-installed), unsloth>=2026.4.7, transformers>=4.51.3,!=5.0.0,!=5.1.0,<=5.5.0, trl>=0.18.2,<=0.24.0, peft>=0.18.0, datasets>=3.4.1,<4.4.0, nltk>=3.9, accelerate>=0.34.1, neo4j>=5.0 (cyper_finetune)

## Project Structure

```text
src/
tests/
```

## Commands

cd src && pytest && ruff check .

## Code Style

Python 3.12+: Follow standard conventions

## Recent Changes

- cyper_finetune: Added torch+cu128 (pre-installed), unsloth>=2026.4.7, transformers (4.51.3–5.5.0 with exclusions), trl>=0.18.2, peft>=0.18.0, datasets>=3.4.1,<4.4.0, nltk>=3.9, accelerate>=0.34.1, neo4j>=5.0

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
