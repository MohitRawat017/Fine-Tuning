"""Retrieve top-K passages for a query with the fine-tuned BGE encoder."""

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# Encoding

def mean_pooling(outputs, attention_mask):
    """Same pooling used at training time — embeddings must match."""
    last_hidden = outputs.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    return torch.sum(last_hidden * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


def encode_texts(model, tokenizer, texts, device):
    """Encode a list of strings into L2-normalized embeddings."""
    model.eval()
    embeddings = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text, truncation=True, max_length=256, padding=True, return_tensors="pt"
            ).to(device)
            outputs = model(**inputs)
            pooled = mean_pooling(outputs, inputs["attention_mask"])
            embeddings.append(F.normalize(pooled, p=2, dim=1).cpu().numpy()[0])
    return np.stack(embeddings)


# Search

def build_index(passages, model, tokenizer, device):
    texts = [f"{p['title']}. {p['text']}" for p in passages]
    return encode_texts(model, tokenizer, texts, device)


def search(query, index_vectors, passages, k=3, model=None, tokenizer=None, device=None):
    q_vec = encode_texts(model, tokenizer, [query], device)[0]
    # Cosine similarity of normalized vectors == plain dot product.
    scores = index_vectors @ q_vec
    top = np.argsort(-scores)[:k]
    return [(passages[i], float(scores[i])) for i in top]


# Main

def main():
    parser = argparse.ArgumentParser(description="Retrieve passages for a query with BGE.")
    parser.add_argument("query", help="Query text.")
    parser.add_argument("--k", type=int, default=3, help="Number of results to return.")
    parser.add_argument(
        "--model", default="bge-ft",
        help="HF model id or local fine-tuned folder (default: bge-ft).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading data and model...")
    with open("passages.json", encoding="utf-8") as fh:
        passages = json.load(fh)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)

    print("Building index...")
    index_vectors = build_index(passages, model, tokenizer, device)

    results = search(
        args.query, index_vectors, passages, k=args.k, model=model, tokenizer=tokenizer, device=device
    )

    print(f"\nQuery: {args.query}\n")
    print("=" * 60)
    for i, (passage, score) in enumerate(results, 1):
        print(f"\n[{i}] {passage['title']}  (score {score:.4f})")
        print(f"    {passage['text']}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
