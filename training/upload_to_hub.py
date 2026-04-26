"""
training/upload_to_hub.py

Uploads the fine-tuned LoRA adapter and the TuneMap Cypher dataset to
HuggingFace Hub. Run this once after training completes.

Uploads:
  - training/outputs/final_adapter/         → HF model repo (464MB)
  - training/data/cypher_dataset_validated.jsonl → HF dataset repo (120 rows)

Optionally also uploads the merged bf16 model (18GB) — disabled by default.

Usage:
    huggingface-cli login        # once, saves token to ~/.huggingface/token
    python training/upload_to_hub.py
"""

from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from datasets import load_dataset
from huggingface_hub import HfApi

# ---------------------------------------------------------------------------
# Config — update these before running
# ---------------------------------------------------------------------------
HF_USERNAME = "danp27"
MODEL_REPO  = f"{HF_USERNAME}/qwen3.5-9b-nl2cypher-lora"
DATASET_REPO = f"{HF_USERNAME}/tunemap-cypher-dataset"
PRIVATE = True                  # set False to make repos public

UPLOAD_ADAPTER = True           # LoRA adapter only (464MB) — for fine-tuning / PEFT inference
UPLOAD_MERGED  = False          # merged bf16 model (18GB)  — for vLLM / direct inference
UPLOAD_DATASET = True           # TuneMap validated Cypher dataset (120 rows)

ADAPTER_DIR = Path("training/outputs/final_adapter")
MERGED_DIR  = Path("training/outputs/merged_16bit")
DATASET_FILE = Path("training/data/cypher_dataset_validated.jsonl")

api = HfApi()

# ---------------------------------------------------------------------------
# Upload LoRA adapter
# ---------------------------------------------------------------------------
if UPLOAD_ADAPTER:
    print(f"Creating model repo {MODEL_REPO} ...")
    api.create_repo(repo_id=MODEL_REPO, repo_type="model", private=PRIVATE, exist_ok=True)

    print(f"Uploading adapter ({ADAPTER_DIR}) ...")
    api.upload_folder(
        folder_path=str(ADAPTER_DIR),
        repo_id=MODEL_REPO,
        repo_type="model",
    )
    print(f"Adapter uploaded → https://huggingface.co/{MODEL_REPO}")

# ---------------------------------------------------------------------------
# Upload merged bf16 model (optional — 18GB)
# ---------------------------------------------------------------------------
if UPLOAD_MERGED:
    print(f"Uploading merged bf16 model ({MERGED_DIR}) — this will take a while ...")
    api.create_repo(repo_id=MODEL_REPO, repo_type="model", private=PRIVATE, exist_ok=True)
    api.upload_folder(
        folder_path=str(MERGED_DIR),
        repo_id=MODEL_REPO,
        repo_type="model",
        commit_message="Add merged bf16 model",
    )
    print(f"Merged model uploaded → https://huggingface.co/{MODEL_REPO}")

# ---------------------------------------------------------------------------
# Upload TuneMap dataset
# ---------------------------------------------------------------------------
if UPLOAD_DATASET:
    print(f"Creating dataset repo {DATASET_REPO} ...")
    api.create_repo(repo_id=DATASET_REPO, repo_type="dataset", private=PRIVATE, exist_ok=True)

    print(f"Loading {DATASET_FILE} ...")
    dataset = load_dataset("json", data_files=str(DATASET_FILE), split="train")
    print(f"Rows: {len(dataset)}")

    print(f"Pushing to {DATASET_REPO} ...")
    dataset.push_to_hub(DATASET_REPO, private=PRIVATE)
    print(f"Dataset uploaded → https://huggingface.co/datasets/{DATASET_REPO}")

print("\nDone.")
