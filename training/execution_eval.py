"""
training/execution_eval.py

T014: TuneMap execution eval on source="tunemap" rows.
Runs EXPLAIN (syntax check) + result-set exact match against local Neo4j.
Writes training/outputs/execution_report.json.

Usage:
    python training/execution_eval.py --checkpoint training/outputs/final_adapter
    python training/execution_eval.py --checkpoint danp27/qwen3.5-9b-nl2cypher-lora
"""

import unsloth  # must be first — patches torch/transformers at import time

import argparse
import json
import logging
import os
import re
import uuid
from pathlib import Path

import torch
from neo4j import GraphDatabase
from neo4j.exceptions import ClientError
from tqdm import tqdm

# Suppress Neo4j server notification noise (UnknownPropertyKey, UnknownLabel warnings)
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

from dotenv import load_dotenv
load_dotenv()

from peft import PeftModel
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

EVAL_DATA   = Path("training/data/eval.jsonl")
OUTPUT_DIR  = Path("training/outputs")
BASE_MODEL  = "Qwen/Qwen3.5-9B"
MAX_SEQ_LEN = 1600
PREVIEW_ROWS = 3

NEO4J_URI  = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "54321Dan")


def strip_output(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^```[\w]*\n?", "", text, flags=re.MULTILINE)
    return text.strip("`").strip()


def extract_question(user_content: str) -> str:
    m = re.search(r"Question:\s*(.+?)(?:\nCypher output:|$)", user_content, re.DOTALL)
    return m.group(1).strip() if m else user_content[-120:].strip()


def syntax_ok(driver, cypher: str) -> bool:
    try:
        with driver.session() as session:
            session.run(f"EXPLAIN {cypher}").consume()
        return True
    except ClientError:
        return False


def execute(driver, cypher: str):
    try:
        with driver.session() as session:
            return session.run(cypher).data()
    except ClientError:
        return None


def result_exact(ref_rows, pred_rows) -> bool:
    if ref_rows is None or pred_rows is None:
        return False
    return str(sorted(str(r) for r in ref_rows)) == str(sorted(str(r) for r in pred_rows))


def result_jaccard(ref_rows, pred_rows) -> float:
    if ref_rows is None or pred_rows is None:
        return 0.0
    ref_vals  = set(str(v) for row in ref_rows  for v in row.values())
    pred_vals = set(str(v) for row in pred_rows for v in row.values())
    if not ref_vals and not pred_vals:
        return 1.0
    return len(ref_vals & pred_vals) / len(ref_vals | pred_vals)


def run_pass(model, tokenizer, driver, rows, label: str) -> list[dict]:
    results = []
    for i, row in enumerate(tqdm(rows, desc=label)):
        convos = row["conversations"]
        messages = [convos[0], convos[1]]
        reference = convos[2]["content"]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(text=text, return_tensors="pt")
        input_ids     = encoded["input_ids"].to("cuda")
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
        predicted = strip_output(predicted)

        valid    = syntax_ok(driver, predicted)
        ref_res  = execute(driver, reference)
        pred_res = execute(driver, predicted) if valid else None
        exact    = result_exact(ref_res, pred_res)
        jaccard  = result_jaccard(ref_res, pred_res)

        if i < PREVIEW_ROWS:
            tqdm.write(f"\n[{label} sample {i+1}]")
            tqdm.write(f"  Q:         {extract_question(convos[1]['content'])}")
            tqdm.write(f"  Ref:       {reference}")
            tqdm.write(f"  Pred:      {predicted}")
            tqdm.write(f"  Syntax OK: {valid}  |  Exact: {exact}  |  Jaccard: {jaccard:.2f}")
            if ref_res is not None:
                tqdm.write(f"  Ref rows:  {ref_res[:3]}{'...' if len(ref_res) > 3 else ''}")
            if pred_res is not None:
                tqdm.write(f"  Pred rows: {pred_res[:3]}{'...' if len(pred_res) > 3 else ''}")

        results.append({
            "question":         extract_question(convos[1]["content"]),
            "reference_cypher": reference,
            "predicted_cypher": predicted,
            "syntax_valid":     valid,
            "exact_match":      exact,
            "jaccard":          jaccard,
        })
    return results


def metrics(results: list[dict]) -> tuple[float, float, float]:
    n = len(results)
    syntax_pct  = sum(r["syntax_valid"] for r in results) / n
    exact_pct   = sum(r["exact_match"]  for r in results) / n
    mean_jaccard = sum(r["jaccard"]     for r in results) / n
    return syntax_pct, exact_pct, mean_jaccard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Local adapter path or HF repo ID")
    args = parser.parse_args()

    all_rows = [json.loads(l) for l in EVAL_DATA.read_text().splitlines() if l.strip()]
    rows = [r for r in all_rows if r.get("source") == "tunemap"]
    print(f"TuneMap eval rows: {len(rows)}")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    driver.verify_connectivity()
    print(f"Neo4j connected: {NEO4J_URI}")

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

    # --- Baseline pass ---
    FastLanguageModel.for_inference(model)
    baseline_results = run_pass(model, tokenizer, driver, rows, label="baseline")
    b_syntax, b_exact, b_jaccard = metrics(baseline_results)
    print(f"Baseline  syntax: {b_syntax:.1%}  exact: {b_exact:.1%}  jaccard: {b_jaccard:.3f}")

    # --- Attach adapter ---
    print(f"\nAttaching adapter: {args.checkpoint}")
    model = PeftModel.from_pretrained(model, args.checkpoint)
    FastLanguageModel.for_inference(model)
    adapter_results = run_pass(model, tokenizer, driver, rows, label="adapter")
    a_syntax, a_exact, a_jaccard = metrics(adapter_results)
    print(f"Adapter   syntax: {a_syntax:.1%}  exact: {a_exact:.1%}  jaccard: {a_jaccard:.3f}")

    driver.close()

    failures = [r for r in adapter_results if not r["syntax_valid"]]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "run_id":     str(uuid.uuid4()),
        "checkpoint": args.checkpoint,
        "n_tunemap":  len(rows),
        "tunemap_syntax_valid_pct":    a_syntax,
        "tunemap_exec_exact_match_pct": a_exact,
        "tunemap_mean_jaccard":         a_jaccard,
        "baseline": {
            "tunemap_syntax_valid_pct":    b_syntax,
            "tunemap_exec_exact_match_pct": b_exact,
            "tunemap_mean_jaccard":         b_jaccard,
        },
        "per_example": adapter_results,
        "failures": failures,
    }

    out_path = OUTPUT_DIR / "execution_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReport → {out_path}")


if __name__ == "__main__":
    main()
