"""
training/smoke_test.py

T012 smoke test — loads the fine-tuned adapter, samples 5 rows from eval.jsonl,
generates Cypher, and prints reference vs predicted side by side.
Confirms output is raw Cypher only (no prose, no markdown fences).

Usage:
    python training/smoke_test.py
"""

import json
import random
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from peft import PeftModel
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

ADAPTER_DIR = "training/outputs/final_adapter"
EVAL_DATA = Path("training/data/eval.jsonl")
SEED = 3407
N_SAMPLES = 5

rows = [json.loads(l) for l in EVAL_DATA.read_text().splitlines() if l.strip()]
random.seed(SEED)
sample = random.sample(rows, N_SAMPLES)

print("Loading base model ...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3.5-9B",
    max_seq_length=1600,
    load_in_4bit=False,
    load_in_16bit=True,
    full_finetuning=False,
)

print(f"Loading adapter from {ADAPTER_DIR} ...")
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
FastLanguageModel.for_inference(model)
tokenizer = get_chat_template(tokenizer, chat_template="chatml")

print("\n" + "=" * 60)
for i, row in enumerate(sample, 1):
    convos = row["conversations"]
    user_content = convos[1]["content"]   # user turn (schema + question)
    reference = convos[2]["content"]      # assistant turn (reference Cypher)

    # Build input — system + user turns only, let model generate assistant turn
    messages = [convos[0], convos[1]]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text=text, return_tensors="pt")["input_ids"].to("cuda")
    outputs = model.generate(input_ids=inputs, max_new_tokens=128, use_cache=True)
    predicted = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip()

    print(f"[{i}] source={row.get('source', '?')}")
    print(f"  User:      {user_content[-120:].strip()}")
    print(f"  Reference: {reference}")
    print(f"  Predicted: {predicted}")
    print("-" * 60)
