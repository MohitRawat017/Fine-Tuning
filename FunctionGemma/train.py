import json
import os
import math
import random
import gc
import time

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

# ── CONFIG ──────────────────────────────────────────────────────────────

MODEL_NAME   = os.getenv("MODEL_NAME", "google/functiongemma-270m-it")
DATASET_PATH = os.getenv("DATASET_PATH", "functiongemma_dataset.jsonl")
OUTPUT_DIR   = os.getenv("OUTPUT_DIR", "fg_finetuned_ckpt")

MAX_LENGTH = int(os.getenv("MAX_LENGTH", "512"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
EPOCHS     = int(os.getenv("EPOCHS", "6"))
LR         = float(os.getenv("LEARNING_RATE", "1e-5"))
GRAD_ACCUM = int(os.getenv("GRAD_ACCUM", "4"))
SEED       = int(os.getenv("SEED", "42"))
USE_PEFT   = bool(int(os.getenv("USE_PEFT", "1"))) and PEFT_AVAILABLE

GEN_MAX_NEW_TOKENS = 64


# ── UTILITIES ───────────────────────────────────────────────────────────

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


set_seed(SEED)


# ── DATA ────────────────────────────────────────────────────────────────

def load_jsonl_dataset(path: str) -> Dataset:
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            msgs = obj.get("messages") or []
            developer = next((m["content"] for m in msgs if m["role"] == "developer"), "")
            user      = next((m["content"] for m in msgs if m["role"] == "user"), "")
            assistant = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            records.append({"developer": developer, "user": user, "assistant_text": assistant})
    return Dataset.from_list(records)


def build_prompt(developer: str, user: str, tokenizer=None) -> str:
    """Build the prompt using the model's chat template (developer + user roles)."""
    messages = []
    if developer:
        messages.append({"role": "developer", "content": developer.strip()})
    messages.append({"role": "user", "content": user.strip()})

    if tokenizer is not None:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # Fallback (should not be reached during training)
    prompt = ""
    if developer:
        prompt += f"{developer.strip()}\n\n"
    prompt += f"User: {user.strip()}\nAssistant: "
    return prompt


def preprocess_batch(examples, tokenizer, max_length=MAX_LENGTH):
    """Tokenize examples: prompt tokens masked (-100), assistant tokens trained on."""
    input_ids_list, labels_list, attention_list = [], [], []

    for dev, usr, assistant in zip(
        examples["developer"], examples["user"], examples["assistant_text"]
    ):
        prompt    = build_prompt(dev, usr, tokenizer)
        full_text = prompt + assistant + tokenizer.eos_token

        prompt_enc = tokenizer(prompt, truncation=True, max_length=max_length, add_special_tokens=False)
        full_enc   = tokenizer(full_text, truncation=True, max_length=max_length, add_special_tokens=False)

        input_ids      = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]
        labels         = input_ids.copy()

        # Mask prompt tokens — only train on assistant output
        prompt_len = len(prompt_enc["input_ids"])
        labels[:prompt_len] = [-100] * prompt_len

        if all(l == -100 for l in labels):
            continue

        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_list.append(attention_mask)

    return {"input_ids": input_ids_list, "labels": labels_list, "attention_mask": attention_list}


# ── CALLBACK ────────────────────────────────────────────────────────────

class EvalPrintCallback(TrainerCallback):
    """Prints validation loss after each epoch."""

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        epoch = round(state.epoch) if state.epoch else "?"
        loss = metrics.get("eval_loss", float("nan"))
        print(f"\n{'='*50}")
        print(f"[Epoch {epoch}]  Val loss: {loss:.4f}")
        print(f"{'='*50}")


# ── GENERATION EVAL ─────────────────────────────────────────────────────

def compute_generation_exact_match(model, tokenizer, dataset, device, max_eval=50):
    """Generate outputs for val examples and check exact string match."""
    model.eval()
    matches, total = 0, 0

    for i, ex in enumerate(dataset):
        if i >= max_eval:
            break

        prompt = build_prompt(ex["developer"], ex["user"], tokenizer)
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=MAX_LENGTH, add_special_tokens=False,
        ).to(device)

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=GEN_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_tokens = gen_ids[0][inputs["input_ids"].shape[1]:]
        generated = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        norm = lambda s: " ".join(s.replace("\n", " ").strip().split())
        if norm(generated) == norm(ex["assistant_text"].strip()):
            matches += 1
        total += 1

        if i % 10 == 0:
            clear_memory()

    model.train()
    return matches / total if total > 0 else 0.0


# ── MAIN ────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("FunctionGemma Fine-tuning")
    print("=" * 60)
    print(f"Device : {device}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"Model  : {MODEL_NAME}")
    print(f"Data   : {DATASET_PATH}")
    print(f"Output : {OUTPUT_DIR}")
    print(f"Config : BS={BATCH_SIZE} ACCUM={GRAD_ACCUM} EPOCHS={EPOCHS} LR={LR:g} PEFT={USE_PEFT}")

    # ── Dataset ──
    print(f"\nLoading dataset: {DATASET_PATH}")
    raw_ds = load_jsonl_dataset(DATASET_PATH)
    print(f"Total examples: {len(raw_ds)}")

    split = raw_ds.train_test_split(test_size=0.1, seed=SEED)
    train_ds, val_ds = split["train"], split["test"]
    print(f"Train / Val   : {len(train_ds)} / {len(val_ds)}")

    clear_memory()

    # ── Tokenizer ──
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print("Pad token → EOS")

    # ── Model ──
    print("Loading model...")
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    load_dtype = torch.bfloat16 if bf16_ok else torch.float32
    print(f"Dtype: {load_dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=load_dtype,
        attn_implementation="eager",
        device_map="auto" if torch.cuda.is_available() else None,
    )

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # ── LoRA ──
    if USE_PEFT:
        print("Applying LoRA...")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    clear_memory()

    # ── Tokenize ──
    print("\nTokenizing...")

    def _tok(ds, label):
        return ds.map(
            lambda ex: preprocess_batch(ex, tokenizer, max_length=MAX_LENGTH),
            batched=True, remove_columns=ds.column_names, desc=f"Tokenizing {label}",
        )

    train_tok = _tok(train_ds, "train")
    val_tok   = _tok(val_ds, "val")
    print(f"Train: {len(train_tok)} | Val: {len(val_tok)}")

    # ── Training setup ──
    effective_bs    = BATCH_SIZE * GRAD_ACCUM
    steps_per_epoch = math.ceil(len(train_tok) / effective_bs)
    total_steps     = steps_per_epoch * EPOCHS
    warmup_steps    = max(1, int(total_steps * 0.06))

    print(f"\nEffective BS : {effective_bs}")
    print(f"Total steps  : {total_steps}")
    print(f"Warmup steps : {warmup_steps}")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model,
        padding=True, pad_to_multiple_of=8, return_tensors="pt",
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        fp16=False,
        bf16=bf16_ok,
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        logging_steps=10,
        logging_first_step=True,
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        seed=SEED,
        report_to=[],
        skip_memory_metrics=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=data_collator,
        callbacks=[EvalPrintCallback()],
    )

    # ── Train ──
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}\n")

    t0 = time.time()
    train_result = trainer.train()
    elapsed = (time.time() - t0) / 60

    print(f"\nTraining complete — {elapsed:.1f} min")
    if train_result and hasattr(train_result, "metrics"):
        print(f"Train loss: {train_result.metrics.get('train_loss', 'n/a')}")

    # ── Save ──
    print("\nSaving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved to: {OUTPUT_DIR}")

    # ── Eval ──
    print("\nRunning generation exact-match eval...")
    clear_memory()

    em = compute_generation_exact_match(trainer.model, tokenizer, val_ds, device, max_eval=50)
    n_eval = min(50, len(val_ds))
    print(f"Exact-match: {em:.2%}  ({int(em * n_eval)}/{n_eval} samples)")

    final_eval = trainer.evaluate()
    metrics = {
        "generation_exact_match": em,
        "final_eval_loss": final_eval.get("eval_loss"),
    }
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Metrics saved: {os.path.join(OUTPUT_DIR, 'metrics.json')}")
    print("\nDone!")


if __name__ == "__main__":
    main()
