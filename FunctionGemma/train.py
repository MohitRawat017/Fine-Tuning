"""
train_functiongemma.py - Fixed & Optimized Version

Key fixes applied:
  1. bf16 instead of fp16 (eliminates NaN gradient explosions)
  2. max_grad_norm=1.0 (gradient clipping)
  3. Pad token reuses EOS - no vocab resize / embedding instability
  4. Dynamic padding instead of max_length padding
  5. LR lowered to 1e-5 with warmup_ratio=0.1
  6. LoRA: r=16, alpha=16, extended target_modules
  7. adamw_torch_fused optimizer (faster, slightly less memory)
  8. torch.compile disabled (not needed, can cause issues on Windows)
  9. Cosine LR scheduler for smoother decay
 10. DataLoader prefetch + persistent workers (off for Windows compat)
 11. gradient_checkpointing_kwargs to silence reentrant warning
 12. Proper PEFT save (only adapter weights, not full model)
"""

import json
import os
import math
import random
import gc
import time
from typing import List, Dict, Any

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    get_cosine_schedule_with_warmup,
)

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

# ===========================
# CONFIG
# ===========================
MODEL_NAME_OR_PATH = os.getenv("MODEL_NAME", "google/functiongemma-270m-it")
DATASET_PATH       = os.getenv("DATASET_PATH", "functiongemma_dataset.jsonl")
OUTPUT_DIR         = os.getenv("OUTPUT_DIR", "fg_finetuned_ckpt")

MAX_LENGTH  = int(os.getenv("MAX_LENGTH",   "512"))   # your real p95 is ~118 tokens, 512 is plenty
BATCH_SIZE  = int(os.getenv("BATCH_SIZE",   "4"))
EPOCHS      = int(os.getenv("EPOCHS",       "6"))
LR          = float(os.getenv("LEARNING_RATE", "1e-5"))  # FIXED: was 4e-5
GRAD_ACCUM  = int(os.getenv("GRAD_ACCUM",   "4"))
SEED        = int(os.getenv("SEED",         "42"))
USE_PEFT    = bool(int(os.getenv("USE_PEFT", "1"))) and PEFT_AVAILABLE

GEN_MAX_NEW_TOKENS = 64
GEN_DO_SAMPLE      = False

# ===========================
# UTILITIES
# ===========================
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def _cuda_mem_str() -> str:
    if not torch.cuda.is_available():
        return "CUDA: n/a"
    alloc    = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved()  / (1024**3)
    return f"CUDA mem: allocated={alloc:.2f} GB, reserved={reserved:.2f} GB"

def _fmt_float(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}"

set_seed(SEED)

# ===========================
# DATA LOADING
# ===========================
def load_jsonl_dataset(path: str) -> Dataset:
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            obj  = json.loads(line)
            msgs = obj.get("messages") or []
            developer = next((m["content"] for m in msgs if m["role"] == "developer"), "")
            user      = next((m["content"] for m in msgs if m["role"] == "user"),      "")
            assistant = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            records.append({"developer": developer, "user": user, "assistant_text": assistant})
    return Dataset.from_list(records)

def build_prompt(developer: str, user: str) -> str:
    prompt = ""
    if developer:
        prompt += f"{developer.strip()}\n\n"
    prompt += f"User: {user.strip()}\nAssistant: "
    return prompt

# ===========================
# PREPROCESSING
# ===========================
def preprocess_batch(examples, tokenizer, max_length=MAX_LENGTH):
    input_ids_list, labels_list, attention_list = [], [], []

    for dev, usr, assistant in zip(
        examples["developer"],
        examples["user"],
        examples["assistant_text"]
    ):
        prompt    = build_prompt(dev, usr)
        full_text = prompt + assistant

        prompt_enc = tokenizer(
            prompt, truncation=True, max_length=max_length, add_special_tokens=False
        )
        full_enc = tokenizer(
            full_text, truncation=True, max_length=max_length, add_special_tokens=False
        )

        input_ids      = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]
        labels         = input_ids.copy()

        # Mask prompt tokens — only train on assistant output
        prompt_len        = len(prompt_enc["input_ids"])
        labels[:prompt_len] = [-100] * prompt_len

        # Guard: if entire sequence is masked, skip (shouldn't happen but be safe)
        if all(l == -100 for l in labels):
            continue

        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_list.append(attention_mask)

    return {
        "input_ids":      input_ids_list,
        "labels":         labels_list,
        "attention_mask": attention_list,
    }

# ===========================
# EVALUATION
# ===========================
def compute_generation_exact_match(model, tokenizer, dataset, device, max_eval=50):
    model.eval()
    matches, total = 0, 0

    for i, ex in enumerate(dataset):
        if i >= max_eval:
            break

        prompt = build_prompt(ex["developer"], ex["user"])
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH
        ).to(device)

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=GEN_MAX_NEW_TOKENS,
                do_sample=GEN_DO_SAMPLE,
                pad_token_id=tokenizer.pad_token_id,
            )

        decoded = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

        if decoded.startswith(prompt):
            generated = decoded[len(prompt):].strip()
        elif "Assistant:" in decoded:
            generated = decoded.split("Assistant:")[-1].strip()
        else:
            generated = decoded.strip()

        def norm(s):
            return " ".join(s.replace("\n", " ").strip().split())

        if norm(generated) == norm(ex["assistant_text"].strip()):
            matches += 1
        total += 1

        if i % 10 == 0:
            clear_memory()

    model.train()
    return matches / total if total > 0 else 0.0

# ===========================
# MAIN
# ===========================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("FunctionGemma Fine-tuning (Fixed + Optimized)")
    print("=" * 60)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU:  {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(_cuda_mem_str())

    print(f"\nMODEL : {MODEL_NAME_OR_PATH}")
    print(f"DATA  : {DATASET_PATH}")
    print(f"OUT   : {OUTPUT_DIR}")
    print(
        " | ".join([
            f"MAX_LEN={MAX_LENGTH}", f"BS={BATCH_SIZE}", f"ACCUM={GRAD_ACCUM}",
            f"EPOCHS={EPOCHS}", f"LR={LR:g}", f"SEED={SEED}", f"PEFT={USE_PEFT}",
        ])
    )

    # ===========================
    # DATASET
    # ===========================
    print(f"\nLoading dataset: {DATASET_PATH}")
    raw_ds  = load_jsonl_dataset(DATASET_PATH)
    print(f"Total examples : {len(raw_ds)}")

    empty = sum(1 for x in raw_ds if not x["assistant_text"].strip())
    lens  = [len(x["assistant_text"]) for x in raw_ds]
    print(f"Empty assistant: {empty}")
    print(f"Asst char stats: min={min(lens)}, max={max(lens)}, avg={sum(lens)//len(lens)}")

    split  = raw_ds.train_test_split(test_size=0.1, seed=SEED)
    train_ds, val_ds = split["train"], split["test"]
    print(f"Train / Val    : {len(train_ds)} / {len(val_ds)}")

    clear_memory()

    # ===========================
    # TOKENIZER
    # FIX: reuse EOS as pad — no vocab resize, no embedding instability
    # ===========================
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, use_fast=True)

    if tokenizer.pad_token_id is None:
        # Reuse EOS — avoids adding a new token and resizing embeddings
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print("Pad token → EOS (no resize needed)")
    else:
        print(f"Pad token already set: id={tokenizer.pad_token_id}")

    print(f"Vocab size: {len(tokenizer)}")

    # ===========================
    # MODEL
    # FIX: use dtype= instead of deprecated torch_dtype=
    # FIX: bf16 (stable) instead of fp16 (NaN-prone on this GPU)
    # ===========================
    print("\nLoading model...")

    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    load_dtype     = torch.bfloat16 if bf16_supported else torch.float32
    print(f"Load dtype: {load_dtype}  (bf16_supported={bf16_supported})")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME_OR_PATH,
            dtype=load_dtype,
            attn_implementation="eager",
            device_map="auto" if torch.cuda.is_available() else None,
        )
        print("Model loaded.")
    except Exception as e:
        print(f"Load with dtype failed ({e}), trying default...")
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME_OR_PATH)

    # No resize needed since we reused EOS as pad token
    # (If you later add genuinely new special tokens, resize here)

    # Gradient checkpointing — use_reentrant=False avoids a deprecation warning
    # and is slightly more memory efficient with PEFT
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    print("Gradient checkpointing: enabled (use_reentrant=False)")

    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_p/1e6:.1f}M total, {trainable_p/1e6:.1f}M trainable")
    print(_cuda_mem_str())

    # ===========================
    # LoRA / PEFT
    # FIX: r=16, alpha=16 (ratio 1:1), extended target modules
    # ===========================
    if USE_PEFT:
        print("\nApplying LoRA...")
        lora_config = LoraConfig(
            r=16,                   # was 8 — better capacity
            lora_alpha=16,          # FIX: was 32 (2x ratio = too aggressive); now 1:1
            target_modules=[        # extended from just q/v
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",   # MLP projections
            ],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print("LoRA applied.")
    else:
        print("\nFull fine-tuning (no LoRA)")

    clear_memory()

    # ===========================
    # TOKENIZE
    # ===========================
    print("\nTokenizing...")

    def _tok(ds, label):
        return ds.map(
            lambda ex: preprocess_batch(ex, tokenizer, max_length=MAX_LENGTH),
            batched=True,
            remove_columns=ds.column_names,
            desc=f"Tokenizing {label}",
        )

    train_tok = _tok(train_ds, "train")
    val_tok   = _tok(val_ds,   "val")
    print(f"Train: {len(train_tok)} | Val: {len(val_tok)}")

    # Quick length stats
    def _stats(ds, label):
        lens = [len(x) for x in ds["input_ids"]]
        s = sorted(lens)
        p50 = s[len(s)//2];  p95 = s[max(0, int(len(s)*0.95)-1)]
        print(f"{label}: min={min(lens)} p50={p50} p95={p95} max={max(lens)} avg={sum(lens)/len(lens):.1f}")
    _stats(train_tok, "train tokens")
    _stats(val_tok,   "val tokens  ")

    # ===========================
    # DATA COLLATOR
    # FIX: dynamic padding (was max_length=1024, wasting 8x compute for ~120-token seqs)
    # ===========================
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,          # dynamic: pad to longest in batch
        pad_to_multiple_of=8,  # keeps tensor dims efficient on GPU
        return_tensors="pt",
    )

    # ===========================
    # TRAINING ARGS
    # All fixes + extras for stability and speed
    # ===========================
    effective_bs    = BATCH_SIZE * GRAD_ACCUM
    steps_per_epoch = math.ceil(len(train_tok) / effective_bs)
    total_steps     = steps_per_epoch * EPOCHS
    warmup_steps    = max(1, int(total_steps * 0.06))

    print(f"\nEffective BS   : {effective_bs}")
    print(f"Steps/epoch    : {steps_per_epoch}")
    print(f"Total steps    : {total_steps}")
    print(f"Warmup steps   : {warmup_steps}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        # Batch / accumulation
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,

        # Epochs & LR
        num_train_epochs=EPOCHS,
        learning_rate=LR,                        # FIX: 1e-5 (was 4e-5)
        warmup_steps=warmup_steps,               # FIX: was 0
        lr_scheduler_type="cosine",              # smoother decay than linear

        # Precision — FIX: bf16 instead of fp16
        fp16=False,
        bf16=bf16_supported,

        # Stability — FIX: was missing
        max_grad_norm=1.0,

        # Memory
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # Optimizer: fused AdamW is faster + slightly lower memory
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",

        # Logging / saving
        logging_steps=10,
        logging_first_step=True,
        save_strategy="epoch",
        save_total_limit=2,           # keep best 2 checkpoints
        eval_strategy="no",

        # Misc
        dataloader_num_workers=0,     # Windows safe
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        seed=SEED,
        report_to=[],
        skip_memory_metrics=True,
        ddp_find_unused_parameters=False,
    )

    # ===========================
    # TRAINER
    # ===========================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=data_collator,
    )

    # ===========================
    # TRAIN
    # ===========================
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    t0 = time.time()
    try:
        train_result = trainer.train()
    except Exception as e:
        print(f"\nTraining error: {e}")
        clear_memory()
        raise
    t1 = time.time()

    print(f"\nTraining complete — {(t1-t0)/60:.1f} min")
    if train_result and hasattr(train_result, "metrics"):
        m = train_result.metrics
        print(f"Train loss : {m.get('train_loss', 'n/a')}")
        print(f"Runtime    : {m.get('train_runtime', 'n/a'):.1f} sec")
    print(_cuda_mem_str())

    # ===========================
    # SAVE
    # With PEFT: save_model saves only adapter weights (small + correct)
    # ===========================
    print("\nSaving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved to: {OUTPUT_DIR}")

    # ===========================
    # EVAL
    # ===========================
    print("\nRunning generation exact-match eval...")
    clear_memory()

    em = compute_generation_exact_match(
        trainer.model, tokenizer, val_ds, device, max_eval=50
    )
    n_eval = min(50, len(val_ds))
    print(f"\nExact-match: {em:.4f}  (on {n_eval} samples)")

    metrics = {"generation_exact_match": em}
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Metrics saved: {os.path.join(OUTPUT_DIR, 'metrics.json')}")
    print("\nDone!")


if __name__ == "__main__":
    main()