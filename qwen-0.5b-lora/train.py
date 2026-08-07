"""LoRA fine-tune Qwen 2.5 0.5B via Unsloth (same recipe as llama-3.2-3B)."""

import json
import os

import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# Config

MODEL_NAME = os.getenv("MODEL_NAME", "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit")
DATASET_PATH = os.getenv("DATASET_PATH", "dataset.jsonl")  # produced by generate_dataset.py
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "lora_adapter")

MAX_SEQ_LENGTH = 1024
LORA_RANK = 16
LORA_ALPHA = 32          # standard practice: 2x the rank
EPOCHS = 3
MAX_STEPS = int(os.getenv("MAX_STEPS", "-1"))  # MAX_STEPS=1 for a smoke test
LEARNING_RATE = 2e-4

# Data

print("Loading dataset...")
raw_data = []
with open(DATASET_PATH, "r", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if line:
            raw_data.append(json.loads(line))

print(f"Loaded {len(raw_data)} examples")
dataset = Dataset.from_list(raw_data)

# MODEL + TOKENIZER (Unsloth 4-bit)

print("Loading model and tokenizer via Unsloth...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,            # auto-detect
    load_in_4bit=True,     # fits comfortably in ~4GB VRAM
)

tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")


def formatting_prompts_func(examples):
    """Turn {instruction, response} rows into Qwen chat-template strings."""
    convos = []
    for instruction, response in zip(examples["instruction"], examples["response"]):
        convo = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
        convos.append(convo)
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        for convo in convos
    ]
    return {"text": texts}


print("Applying LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,        # 0 is optimized for memory
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

print("Formatting dataset to chat template...")
dataset = dataset.map(formatting_prompts_func, batched=True)

# Train

print("Setting up trainer...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=1,   # Windows-safe
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,   # effective batch = 8
        warmup_steps=5,
        num_train_epochs=EPOCHS,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir=OUTPUT_DIR,
        report_to="none",
    ),
)

print("Starting training...")
trainer.train()

# SAVE + SANITY CHECK

print("Saving LoRA adapter...")
model.save_pretrained("lora_adapter")
tokenizer.save_pretrained("lora_adapter")

print("\n--- Running Sanity Check ---")
FastLanguageModel.for_inference(model)

for prompt in ["explain machine learning in simple terms", "i had a rough day"]:
    print(f"\nUser: {prompt}")
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=150,
            use_cache=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
    print(f"Assistant: {response}")
