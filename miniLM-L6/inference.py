"""Classify queries into intents with the fine-tuned MiniLM model."""

import argparse
import os
import sys

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

# Intent → the FunctionGemma tool groups that make sense for this intent.
# Kept in sync with LABEL_MAP in train_minilm.py.
ID_TO_LABEL = {
    0: "casual",
    1: "productivity",
    2: "system",
    3: "research",
    4: "communication",
}

LABEL_TO_ACTIONS = {
    "casual": [],
    "productivity": ["set_timer", "set_alarm", "create_calendar_event", "add_task", "get_tasks"],
    "system": ["open_app", "run_command", "get_system_info"],
    "research": ["web_search", "search_stackoverflow", "search_arxiv"],
    "communication": ["send_email", "read_emails"],
}


class MiniLMClassifier(nn.Module):
    """Same architecture as train_minilm.py — must match exactly to load weights."""

    def __init__(self, num_classes=5, dropout=0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(384, num_classes)  # MiniLM hidden size = 384

    def mean_pooling(self, outputs, attention_mask):
        last_hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        return torch.sum(last_hidden * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pooling(outputs, attention_mask)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


class IntentClassifier:
    """Loads the trained checkpoint and exposes a .predict(text) API."""

    def __init__(self, model_path="tsuzi_intent_model.pt", device=None):
        if not os.path.exists(model_path):
            sys.exit(
                f"Checkpoint not found: {model_path}\n"
                "Run 'python train_minilm.py' first to train and save the model."
            )

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Prefer the local tokenizer (offline-capable), else the HF hub copy.
        if os.path.isdir("tsuzi_tokenizer"):
            self.tokenizer = AutoTokenizer.from_pretrained("tsuzi_tokenizer")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

        self.model = MiniLMClassifier(num_classes=5)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text, return_confidence=True):
        """Classify one query → dict with intent, confidence, and available actions."""
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            pred_id = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_id].item()

        label = ID_TO_LABEL[pred_id]
        result = {
            "text": text,
            "intent": label,
            "label_id": pred_id,
            "available_actions": LABEL_TO_ACTIONS[label],
        }

        if return_confidence:
            result["confidence"] = round(confidence, 4)
            result["all_probs"] = {
                ID_TO_LABEL[i]: round(probs[0][i].item(), 4) for i in range(5)
            }

        return result


def demo(classifier):
    """Run a fixed set of demo queries, then an interactive loop."""
    test_queries = [
        "hey um can you set an alarm for 7am",
        "open vscode please",
        "search stackoverflow for python error",
        "check my emails",
        "thanks a lot",
        "hey what's up",
        "start a timer for 10 minutes",
        "run pip install requests",
        "find papers on machine learning",
        "send email to john",
    ]

    print("\n" + "=" * 60)
    print("TESTING INTENT CLASSIFICATION")
    print("=" * 60)

    for query in test_queries:
        result = classifier.predict(query)
        print(f"\nQuery: \"{query}\"")
        print(f"  Intent: {result['intent']} (confidence: {result['confidence']})")
        print(f"  Available actions: {result['available_actions']}")

    print("\n" + "=" * 60)
    print("INTERACTIVE MODE (type 'quit' to exit)")
    print("=" * 60)

    while True:
        query = input("\nEnter query: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        result = classifier.predict(query)
        print(f"  Intent: {result['intent']} (confidence: {result['confidence']})")
        print(f"  Available actions: {result['available_actions']}")


def main():
    parser = argparse.ArgumentParser(description="Classify queries into voice-assistant intents.")
    parser.add_argument("query", nargs="?", help="Optional: classify a single query and exit.")
    args = parser.parse_args()

    print("Loading model...")
    classifier = IntentClassifier()

    if args.query:
        result = classifier.predict(args.query)
        print(f"\nQuery: \"{result['text']}\"")
        print(f"  Intent: {result['intent']} (confidence: {result['confidence']})")
        print(f"  Available actions: {result['available_actions']}")
        return

    demo(classifier)


if __name__ == "__main__":
    main()
