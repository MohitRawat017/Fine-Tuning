"""Predict the emotion of a piece of text."""

import argparse
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def classify(text, model, tokenizer, device):
    inputs = tokenizer(text, truncation=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]

    ranked = sorted(zip(LABELS, probs.tolist()), key=lambda x: -x[1])
    return ranked


def main():
    parser = argparse.ArgumentParser(description="Predict emotion for a piece of text.")
    parser.add_argument("text", help="Text to classify (quote it if it has spaces).")
    parser.add_argument(
        "--model",
        default="distilbert-emotion-ft",
        help="HF model id or local folder (default: distilbert-emotion-ft).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = AutoModelForSequenceClassification.from_pretrained(args.model)
    except OSError:
        sys.exit(
            f"Could not load '{args.model}'. Train first with 'python train.py', "
            "or pass --model distilbert-base-uncased to use the untrained head."
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.to(device).eval()

    ranked = classify(args.text, model, tokenizer, device)
    print(f"\nText: {args.text}\n")
    for label, prob in ranked:
        print(f"  {label:<10} {prob:.4f}")
    print(f"\n→ Predicted: {ranked[0][0]} ({ranked[0][1]:.2%})")


if __name__ == "__main__":
    main()
