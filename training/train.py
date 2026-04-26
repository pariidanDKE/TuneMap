"""
training/train.py

Fine-tunes Qwen/Qwen3.5-9B with LoRA (bf16, no quantization) on the prepared
text2cypher dataset, producing a LoRA adapter checkpoint and a merged bf16 model.

Usage:
    python training/train.py

Prerequisites:
    pip install torch --index-url https://download.pytorch.org/whl/cu128
    pip install -r training/requirements.txt
    python training/prepare_data.py   # produces training/data/train.jsonl

Outputs:
    training/outputs/final_adapter/    LoRA adapter weights + tokenizer
    training/outputs/merged_16bit/     merged bf16 model for vLLM

Resume: if training is interrupted, re-run the same command — SFTTrainer
automatically resumes from the latest checkpoint-XXXX/ in output_dir.
"""

# T007 — NLTK punkt download (required by eval.py; harmless to include here)
import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

import logging
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer, SFTConfig

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

TRAIN_DATA = Path("training/data/train.jsonl")
OUTPUT_DIR = "training/outputs"
ADAPTER_DIR = "training/outputs/final_adapter"
MERGED_DIR = "training/outputs/merged_16bit"

# ---------------------------------------------------------------------------
# T007 — Model loading (bf16, no quantization)
# ---------------------------------------------------------------------------
log.info("Loading Qwen/Qwen3.5-9B (bf16) ...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3.5-9B",
    max_seq_length=1600,
    load_in_4bit=False,
    load_in_16bit=True,       # native bf16; no BnB quantization
    full_finetuning=False,
)

# ---------------------------------------------------------------------------
# T008 — LoRA configuration
# ---------------------------------------------------------------------------
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # here, not in from_pretrained
    random_state=3407,
    max_seq_length=1600,
)

# ---------------------------------------------------------------------------
# T009 — Dataset loading and SFTTrainer
# ---------------------------------------------------------------------------
tokenizer = get_chat_template(tokenizer, chat_template="chatml")

dataset = load_dataset("json", data_files={"train": str(TRAIN_DATA)}, split="train")
dataset = dataset.filter(lambda x: x["source"] == "external")


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
    train_dataset=dataset,
    dataset_text_field="text",
    args=SFTConfig(
        per_device_train_batch_size=1,      # only viable: 22GB model on 24GB card
        gradient_accumulation_steps=32,     # effective batch = 32
        num_train_epochs=1,
        learning_rate=2e-5,
        max_seq_length=1600,
        bf16=True,
        fp16=False,
        optim="paged_adamw_8bit",
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        output_dir=OUTPUT_DIR,
        report_to="wandb",
        run_name="qwen3.5-9b-cypher",
        seed=3407,
        dataset_num_proc=1,
    ),
)

# ---------------------------------------------------------------------------
# T010 — Mask loss on instruction tokens; train on assistant responses only
# ---------------------------------------------------------------------------
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

log.info("Starting training ...")
trainer.train()

# ---------------------------------------------------------------------------
# T011 — Save adapter and merged bf16 model
# ---------------------------------------------------------------------------
log.info("Saving LoRA adapter to %s ...", ADAPTER_DIR)
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)

log.info("Saving merged bf16 model to %s ...", MERGED_DIR)
model.save_pretrained_merged(MERGED_DIR, tokenizer, save_method="merged_16bit")

log.info("Training complete.")
