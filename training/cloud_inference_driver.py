#!/usr/bin/env python3
# TuneMap NL-to-Cypher — Eval Driver (Vast.ai)
# VRAM requirement: Qwen3.5-9B bf16 needs ~18 GB.
#
# Usage (run from repo root):
#   HF_TOKEN=hf_xxx python training/cloud_inference_driver.py
#   HF_TOKEN=hf_xxx python training/cloud_inference_driver.py --checkpoint danp27/other-adapter
#
# Download results from your local terminal:
#   scp -P <port> root@<ip>:/root/AppleMusciKG/training/outputs/translation_report.json .

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

CHECKPOINT = "danp27/qwen3.5-9b-nl2cypher-lora"

# ── 1. Repo root ──────────────────────────────────────────────────────────────

root = Path(__file__).resolve().parents[1]
os.chdir(root)
assert (root / "training").is_dir(), f"training/ not found under {root}"
sys.path.insert(0, str(root))

# ── 2. Credentials ────────────────────────────────────────────────────────────

load_dotenv()
print("HF_TOKEN set:", bool(os.environ.get("HF_TOKEN")))
print("Checkpoint:", CHECKPOINT)

# ── 3. Regenerate eval.jsonl ──────────────────────────────────────────────────

from training.prepare_data import main as prepare_data
prepare_data()

# ── 4. Translation eval (GLEU — no DB required) ───────────────────────────────
# Expected runtime: ~5–6 h on a single GPU.
# Output: training/outputs/translation_report.json

from training.evaluation.translation_eval import main as translation_eval
translation_eval(checkpoint=CHECKPOINT)

# ── 5. Summary ────────────────────────────────────────────────────────────────

report = json.loads(open("training/outputs/translation_report.json").read())
print(f"Adapter mean GLEU:  {report['mean_gleu']:.4f}")
print(f"Baseline mean GLEU: {report['baseline']['mean_gleu']:.4f}")
print(f"Rows evaluated:     {report['n_translation']}")
print("\n--- 3 worst failures ---")
for f in report["failures"][:3]:
    print(f"  GLEU {f['gleu']:.3f} | Q: {f['question'][:80]}")
    print(f"          Ref: {f['reference_cypher'][:100]}")
    print(f"          Pred: {f['predicted_cypher'][:100]}")
