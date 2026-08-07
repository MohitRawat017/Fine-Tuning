"""Fine-tune BGE embeddings with a contrastive (InfoNCE) objective."""

import json
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Config

MODEL_NAME = "BAAI/bge-small-en-v1.5"
OUTPUT_DIR = "bge-ft"

BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 2e-5
MAX_LENGTH = 256
SEED = 42
TEMPERATURE = 0.05   # scales logits in InfoNCE; 0.05 is a common default

random.seed(SEED)
torch.manual_seed(SEED)


# Data

class QADataset(Dataset):
    """Holds a question and its positive passage (the negative = other passages)."""

    def __init__(self, pairs, passages, tokenizer):
        self.questions = [p["question"] for p in pairs]
        # Positive passage text for each pair.
        self.positives = [
            passages[p["positive_passage_id"]]["text"] for p in pairs
        ]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return {
            "question": self.questions[idx],
            "positive": self.positives[idx],
        }


def make_collate(tokenizer):
    """Build a batch collator that closes over the tokenizer."""
    def collate(batch):
        questions = [b["question"] for b in batch]
        positives = [b["positive"] for b in batch]
        return {
            "question_input": tokenizer(
                questions, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt"
            ),
            "positive_input": tokenizer(
                positives, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt"
            ),
        }
    return collate


# Model

def mean_pooling(outputs, attention_mask):
    """BGE's recommended pooling: mask-weighted average over token embeddings."""
    last_hidden = outputs.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    return torch.sum(last_hidden * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


def embed(model, tokenizer, inputs, device):
    """Encode a tokenized batch into normalized sentence embeddings."""
    outputs = model(
        input_ids=inputs["input_ids"].to(device),
        attention_mask=inputs["attention_mask"].to(device),
    )
    pooled = mean_pooling(outputs, inputs["attention_mask"].to(device))
    return F.normalize(pooled, p=2, dim=1)


def infonce_loss(q_emb, p_emb, temperature=TEMPERATURE):
    """InfoNCE: the diagonal of q@p.T is the positive pair, the rest are negatives."""
    logits = (q_emb @ p_emb.T) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)


# Main

def main():
    print("Loading data...")
    with open("qa_pairs.json", encoding="utf-8") as fh:
        pairs = json.load(fh)
    with open("passages.json", encoding="utf-8") as fh:
        passages = json.load(fh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)

    dataset = QADataset(pairs, passages, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=make_collate(tokenizer))

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print("\n===== TRAINING =====\n")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch + 1}"):
            q_emb = embed(model, tokenizer, batch["question_input"], device)
            p_emb = embed(model, tokenizer, batch["positive_input"], device)

            loss = infonce_loss(q_emb, p_emb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
        print(f"Epoch {epoch + 1} — loss: {total_loss / len(loader):.4f}")

    print(f"\nSaving fine-tuned encoder to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done!")


if __name__ == "__main__":
    main()
