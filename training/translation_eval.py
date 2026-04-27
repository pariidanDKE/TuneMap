"""
training/translation_eval.py

T013: Translation-based evaluation (GLEU) on all eval.jsonl rows.
No database required — runs on cloud or locally.

Runs two passes: adapter and baseline (base model without adapter).
Writes training/outputs/translation_report.json.

Usage:
    python training/translation_eval.py --checkpoint training/outputs/final_adapter
    python training/translation_eval.py --checkpoint danp27/qwen3.5-9b-nl2cypher-lora
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
load_dotenv()

from peft import PeftModel
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

EVAL_DATA = Path("training/data/eval.jsonl")
OUTPUT_DIR = Path("training/outputs")
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


def run_pass(model, tokenizer, rows, label: str) -> list[dict]:
    results = []
    for i, row in enumerate(tqdm(rows, desc=label)):
        convos = row["conversations"]
        messages = [convos[0], convos[1]]
        reference = convos[2]["content"]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(text=text, return_tensors="pt")
        input_ids = encoded["input_ids"].to("cuda")
        attention_mask = encoded["attention_mask"].to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=256,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
            )
        predicted = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        predicted = strip_fences(predicted)

        gleu = float(sentence_gleu([reference.lower().split()], predicted.lower().split()))

        if i < PREVIEW_ROWS:
            tqdm.write(f"\n[{label} sample {i+1}]")
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


def main():
    nltk.download("punkt_tab", quiet=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Local adapter path or HF repo ID")
    args = parser.parse_args()

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
    print(f"\nAttaching adapter: {args.checkpoint}")
    model = PeftModel.from_pretrained(model, args.checkpoint)
    FastLanguageModel.for_inference(model)
    adapter_results = run_pass(model, tokenizer, rows, label="adapter")
    mean_gleu = sum(r["gleu"] for r in adapter_results) / len(adapter_results)
    print(f"Adapter mean GLEU: {mean_gleu:.4f}")

    failures = sorted(adapter_results, key=lambda x: x["gleu"])[:10]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "run_id": str(uuid.uuid4()),
        "checkpoint": args.checkpoint,
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
