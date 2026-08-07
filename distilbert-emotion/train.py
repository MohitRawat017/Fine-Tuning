"""Fine-tune DistilBERT into a 6-way emotion classifier (head-only or full)."""

import os

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Config

MODEL_NAME = os.getenv("MODEL_NAME", "distilbert-base-uncased")
DATASET_NAME = "dair-ai/emotion"
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "distilbert-emotion-ft")

NUM_LABELS = 6
FREEZE_ENCODER = os.getenv("FREEZE_ENCODER", "1") == "1"  # 1 = head-only

BATCH_SIZE = 32
LEARNING_RATE = 2e-5
EPOCHS = 3
SEED = 42

LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


# Metrics

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }


# Main

def main():
    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        # Truncate at 128 tokens — tweets are short, longer text adds noise.
        return tokenizer(batch["text"], truncation=True, max_length=128)

    train_ds = dataset["train"].map(tokenize, batched=True, remove_columns=["text"])
    test_ds = dataset["test"].map(tokenize, batched=True, remove_columns=["text"])

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )

    # Head-only mode: freeze the encoder, train just the classifier head.
    if FREEZE_ENCODER:
        print("Freezing encoder (head-only training)...")
        for param in model.distilbert.parameters():
            param.requires_grad = False

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        warmup_ratio=0.1,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        report_to=[],
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("\n=== Final test-set evaluation ===")
    eval_result = trainer.evaluate()
    for key, value in eval_result.items():
        print(f"  {key}: {value:.4f}")

    print(f"\nSaving model to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done!")


if __name__ == "__main__":
    main()
