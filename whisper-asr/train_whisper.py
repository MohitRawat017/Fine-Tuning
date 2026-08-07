"""Fine-tune Whisper on a small English ASR corpus."""

import os

import torch
from datasets import load_dataset
from evaluate import load as load_metric
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)
from transformers.trainer_utils import EvalPrediction

# Config

MODEL_NAME = os.getenv("MODEL_NAME", "openai/whisper-tiny")

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "whisper-ft")
DATASET_NAME = os.getenv("DATASET_NAME", "librispeech_asr")
TRAIN_SPLIT = os.getenv("TRAIN_SPLIT", "train.clean.100")
EVAL_SPLIT = os.getenv("EVAL_SPLIT", "test.clean")

# Keep the training slice small — this is a recipe demo, not a production run.
MAX_TRAIN_SAMPLES = int(os.getenv("MAX_TRAIN_SAMPLES", "1000"))
MAX_EVAL_SAMPLES = int(os.getenv("MAX_EVAL_SAMPLES", "300"))

BATCH_SIZE = 8
GRAD_ACCUM = 2
LEARNING_RATE = 1e-5
EPOCHS = 2
MAX_STEPS = int(os.getenv("MAX_STEPS", "-1"))  # set 1 for a smoke test
SEED = 42


# Data preparation

def prepare_dataset(batch, feature_extractor, tokenizer, processor):
    """Convert one audio row into log-mel features + tokenized labels."""
    audio = batch["audio"]
    # Load raw audio at native sample rate, then resample to 16 kHz.
    inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt")
    batch["input_features"] = inputs.input_features[0]

    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch


# Metrics

wer_metric = load_metric("wer")


def make_compute_metrics(tokenizer):
    """WER metric closure; captures the tokenizer for decoding."""
    def compute_metrics(pred: EvalPrediction):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # -100 marks masked positions; strip them before decoding.
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": round(wer, 4)}

    return compute_metrics


# Main

def main():
    print("Loading processor, feature extractor, and tokenizer...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, language="en", task="transcribe")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="en", task="transcribe")

    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, "clean", split=TRAIN_SPLIT)
    eval_full = load_dataset(DATASET_NAME, "clean", split=EVAL_SPLIT)

    # Slice for a quick recipe run; raise MAX_TRAIN_SAMPLES for more data.
    train_ds = dataset.select(range(min(MAX_TRAIN_SAMPLES, len(dataset))))
    eval_ds = eval_full.select(range(min(MAX_EVAL_SAMPLES, len(eval_full))))

    train_ds = train_ds.map(
        lambda b: prepare_dataset(b, feature_extractor, tokenizer, processor),
        remove_columns=train_ds.column_names,
        num_proc=1,  # 1 on Windows (spawn-safe)
    )
    eval_ds = eval_ds.map(
        lambda b: prepare_dataset(b, feature_extractor, tokenizer, processor),
        remove_columns=eval_ds.column_names,
        num_proc=1,
    )

    print("Loading model...")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_steps=100,
        num_train_epochs=EPOCHS,
        max_steps=MAX_STEPS,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch" if MAX_STEPS == -1 else "no",
        save_strategy="epoch",
        predict_with_generate=True,
        generation_max_length=128,
        logging_steps=25,
        report_to=[],
        seed=SEED,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=processor.feature_extractor,
        compute_metrics=make_compute_metrics(tokenizer),
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("Done!")


if __name__ == "__main__":
    main()
