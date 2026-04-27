"""
training/evaluation/translation_eval.py

T013: Translation-based evaluation (GLEU) on all eval.jsonl rows.
No database required — runs on cloud or locally.

Runs two passes: adapter and baseline (base model without adapter).
Writes training/outputs/translation_report.json.

Usage:
    python training/evaluation/translation_eval.py --checkpoint training/outputs/final_adapter
    python training/evaluation/translation_eval.py --checkpoint danp27/qwen3.5-9b-nl2cypher-lora
"""

import unsloth  # must be first — patches torch/transformers at import time

import argparse
import json
import re
import uuid
from pathlib import Path

import nltk
import torch
from nltk.translate.gleu_score import sentence_gleu
from tqdm import tqdm

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=ROOT_DIR / ".env")

from peft import PeftModel
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

EVAL_DATA = ROOT_DIR / "training" / "data" / "eval.jsonl"
OUTPUT_DIR = ROOT_DIR / "training" / "outputs"
BASE_MODEL = "Qwen/Qwen3.5-9B"
MAX_SEQ_LEN = 1600


def strip_fences(text: str) -> str:
    # Remove <think>...</think> reasoning blocks (Qwen3.5 baseline emits these)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Remove markdown code fences
    text = re.sub(r"^```[\w]*\n?", "", text, flags=re.MULTILINE)
    return text.strip("`").strip()


def extract_question(user_content: str) -> str:
    m = re.search(r"Question:\s*(.+?)(?:\nCypher output:|$)", user_content, re.DOTALL)
    return m.group(1).strip() if m else user_content[-120:].strip()


PREVIEW_ROWS = 3  # print first N examples per pass so you can sanity-check early
EVAL_BATCH_SIZE = 4


def run_pass(model, tokenizer, rows, label: str) -> list[dict]:
    results = []
    tokenizer.padding_side = "left"

    for batch_start in tqdm(range(0, len(rows), EVAL_BATCH_SIZE), desc=label):
        batch = rows[batch_start : batch_start + EVAL_BATCH_SIZE]

        texts, references, convos_list = [], [], []
        for row in batch:
            convos = row["conversations"]
            text = tokenizer.apply_chat_template(
                [convos[0], convos[1]], tokenize=False, add_generation_prompt=True
            )
            texts.append(text)
            references.append(convos[2]["content"])
            convos_list.append(convos)

        encoded = tokenizer(text=texts, return_tensors="pt", padding=True).to("cuda")
        input_len = encoded["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_new_tokens=256,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
            )

        for i, (row, convos, reference) in enumerate(zip(batch, convos_list, references)):
            predicted = tokenizer.decode(outputs[i][input_len:], skip_special_tokens=True).strip()
            predicted = strip_fences(predicted)
            gleu = float(sentence_gleu([reference.lower().split()], predicted.lower().split()))

            global_i = batch_start + i
            if global_i < PREVIEW_ROWS:
                tqdm.write(f"\n[{label} sample {global_i+1}]")
                tqdm.write(f"  Q:    {extract_question(convos[1]['content'])}")
                tqdm.write(f"  Ref:  {reference}")
                tqdm.write(f"  Pred: {predicted}")
                tqdm.write(f"  GLEU: {gleu:.3f}")

            results.append({
                "question": extract_question(convos[1]["content"]),
                "reference_cypher": reference,
                "predicted_cypher": predicted,
                "gleu": gleu,
                "source": row.get("source", "external"),
            })

    return results


def main(checkpoint: str | None = None):
    nltk.download("punkt_tab", quiet=True)

    if checkpoint is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--checkpoint", required=True, help="Local adapter path or HF repo ID")
        checkpoint = parser.parse_args().checkpoint

    rows = [json.loads(l) for l in EVAL_DATA.read_text().splitlines() if l.strip()]
    print(f"Eval rows: {len(rows)}")

    # Load base model once
    print("\nLoading base model ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="chatml")

    # --- Baseline pass first (no adapter) ---
    FastLanguageModel.for_inference(model)
    baseline_results = run_pass(model, tokenizer, rows, label="baseline")
    baseline_mean_gleu = sum(r["gleu"] for r in baseline_results) / len(baseline_results)
    print(f"Baseline mean GLEU: {baseline_mean_gleu:.4f}")

    # --- Attach adapter and run adapter pass ---
    print(f"\nAttaching adapter: {checkpoint}")
    model = PeftModel.from_pretrained(model, checkpoint)
    FastLanguageModel.for_inference(model)
    adapter_results = run_pass(model, tokenizer, rows, label="adapter")
    mean_gleu = sum(r["gleu"] for r in adapter_results) / len(adapter_results)
    print(f"Adapter mean GLEU: {mean_gleu:.4f}")

    failures = sorted(adapter_results, key=lambda x: x["gleu"])[:10]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "run_id": str(uuid.uuid4()),
        "checkpoint": checkpoint,
        "n_translation": len(rows),
        "mean_gleu": mean_gleu,
        "baseline": {"mean_gleu": baseline_mean_gleu},
        "per_example": adapter_results,
        "failures": failures,
    }

    out_path = OUTPUT_DIR / "translation_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReport → {out_path}")


if __name__ == "__main__":
    main()
