"""
training/eval_loss.py

Computes validation loss on eval.jsonl using the fine-tuned adapter.
Uses the same train_on_responses_only masking as train.py so the loss
is computed only on assistant (Cypher) tokens — directly comparable to
training loss.

Usage:
    python training/eval_loss.py
"""

import math
import logging
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from datasets import load_dataset
from peft import PeftModel
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer, SFTConfig

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ADAPTER_DIR = "training/outputs/final_adapter"
EVAL_DATA = Path("training/data/eval.jsonl")
OUTPUT_DIR = "training/outputs/eval_loss_run"

log.info("Loading base model ...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3.5-9B",
    max_seq_length=1600,
    load_in_4bit=False,
    load_in_16bit=True,
    full_finetuning=False,
)

log.info("Loading adapter from %s ...", ADAPTER_DIR)
model = PeftModel.from_pretrained(model, ADAPTER_DIR)

tokenizer = get_chat_template(tokenizer, chat_template="chatml")

dataset = load_dataset("json", data_files={"eval": str(EVAL_DATA)}, split="eval")
dataset = dataset.filter(lambda x: x["source"] == "external")
log.info("Eval rows: %d", len(dataset))


def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        for convo in convos
    ]
    return {"text": texts}


dataset = dataset.map(formatting_prompts_func, batched=True, num_proc=1)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset.select([0]),  # unsloth requires non-None train_dataset at init
    eval_dataset=dataset,
    dataset_text_field="text",
    args=SFTConfig(
        per_device_eval_batch_size=1,
        max_seq_length=1600,
        bf16=True,
        fp16=False,
        output_dir=OUTPUT_DIR,
        seed=3407,
        dataset_num_proc=1,
        report_to="none",
    ),
)

# Same masking as train.py — loss on assistant tokens only
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

metrics = trainer.evaluate()

eval_loss = metrics["eval_loss"]
perplexity = math.exp(eval_loss)
print(f"\n{'='*40}")
print(f"Eval loss:   {eval_loss:.4f}")
print(f"Perplexity:  {perplexity:.4f}")
print(f"All metrics: {metrics}")
print(f"{'='*40}\n")
