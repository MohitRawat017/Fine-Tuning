"""
Tsuzi MiniLM Intent Classifier - Production Version

Features:
- Correct MiniLM mean pooling
- Dataset validation
- Debugging tools
- Early stopping
- Confidence scoring
- Overfitting detection
- Balanced training checks
- Clean export format
"""

import json
import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup,DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
import os

# =========================
# CONFIG
# =========================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-5
MAX_LENGTH = 128
EARLY_STOPPING_PATIENCE = 3

DATASET_PATH = "dataset.jsonl"
MODEL_OUTPUT = "tsuzi_intent_model.pt"

SEED = 42

LABEL_MAP = {
    "casual": 0,
    "productivity": 1,
    "system": 2,
    "research": 3,
    "communication": 4
}

ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


# =========================
# REPRODUCIBILITY
# =========================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)


# =========================
# DATASET VALIDATION
# =========================

def validate_dataset(texts, labels):

    print("\n===== DATASET VALIDATION =====")

    print("Total samples:", len(texts))

    label_counts = Counter(labels)

    print("\nLabel distribution:")

    for k,v in label_counts.items():
        print(ID_TO_LABEL[k],":",v)

    # Detect imbalance
    max_count = max(label_counts.values())
    min_count = min(label_counts.values())

    if max_count - min_count > 20:
        print("\nWARNING: Dataset is imbalanced!")

    # Check duplicates

    duplicates = len(texts) - len(set(texts))

    print("\nDuplicate samples:", duplicates)

    if duplicates > 10:
        print("WARNING: Too many duplicates!")

    # Length check

    lengths = [len(t.split()) for t in texts]

    print("\nAvg length:", np.mean(lengths))
    print("Max length:", np.max(lengths))

    print("============================\n")


# =========================
# DATASET CLASS
# =========================

class IntentDataset(Dataset):

    def __init__(self, texts, labels, tokenizer):

        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):

        return len(self.texts)

    def __getitem__(self, idx):

        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=MAX_LENGTH
        )

        encoding["labels"] = self.labels[idx]

        return encoding


# =========================
# MODEL
# =========================

class MiniLMClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = AutoModel.from_pretrained(MODEL_NAME)

        self.dropout = nn.Dropout(0.2)

        self.classifier = nn.Linear(384, len(LABEL_MAP))


    def mean_pooling(self, outputs, attention_mask):

        last_hidden = outputs.last_hidden_state

        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()

        return torch.sum(last_hidden * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


    def forward(self, input_ids, attention_mask):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled = self.mean_pooling(outputs, attention_mask)

        pooled = self.dropout(pooled)

        logits = self.classifier(pooled)

        return logits


# =========================
# LOAD DATA
# =========================

def load_data(path):

    texts=[]
    labels=[]

    with open(path,"r",encoding="utf-8") as f:

        for line in f:

            item=json.loads(line)

            texts.append(item["text"])
            labels.append(item["label_id"])

    return texts,labels


# =========================
# TRAIN
# =========================

def train_epoch(model,loader,optimizer,scheduler,device):

    model.train()

    total_loss=0

    loss_fn=nn.CrossEntropyLoss(label_smoothing=0.05)

    for batch in loader:

        optimizer.zero_grad()

        input_ids=batch["input_ids"].to(device)
        mask=batch["attention_mask"].to(device)
        labels=batch["labels"].to(device)

        logits=model(input_ids,mask)

        loss=loss_fn(logits,labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)

        optimizer.step()
        scheduler.step()

        total_loss+=loss.item()

    return total_loss/len(loader)


# =========================
# EVALUATION
# =========================

def evaluate(model,loader,device):

    model.eval()

    preds=[]
    labels_list=[]
    confidences=[]

    with torch.no_grad():

        for batch in loader:

            input_ids=batch["input_ids"].to(device)
            mask=batch["attention_mask"].to(device)
            labels=batch["labels"].to(device)

            logits=model(input_ids,mask)

            probs=torch.softmax(logits,dim=1)

            confidence,_=torch.max(probs,dim=1)

            pred=torch.argmax(logits,dim=1)

            preds.extend(pred.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
            confidences.extend(confidence.cpu().numpy())

    accuracy=accuracy_score(labels_list,preds)

    return accuracy,preds,labels_list,confidences


# =========================
# DEBUG PREDICTIONS
# =========================

def debug_predictions(texts,preds,true_labels,conf):

    print("\n===== SAMPLE PREDICTIONS =====\n")

    for i in range(min(15,len(texts))):

        print("TEXT:",texts[i])
        print("TRUE:",ID_TO_LABEL[true_labels[i]])
        print("PRED:",ID_TO_LABEL[preds[i]])
        print("CONF:",round(conf[i],3))

        print("---")


# =========================
# MAIN
# =========================

def main():

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:",device)

    texts,labels=load_data(DATASET_PATH)

    validate_dataset(texts,labels)

    train_texts,val_texts,train_labels,val_labels=train_test_split(

        texts,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=SEED

    )

    tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME)

    collator = DataCollatorWithPadding(tokenizer)

    train_dataset=IntentDataset(train_texts,train_labels,tokenizer)
    val_dataset=IntentDataset(val_texts,val_labels,tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collator
    )

    model=MiniLMClassifier().to(device)

    optimizer=AdamW(model.parameters(),lr=LEARNING_RATE)

    total_steps=len(train_loader)*EPOCHS
    num_warmup_steps = int(0.1 * total_steps)

    scheduler=get_linear_schedule_with_warmup(

        optimizer,
        num_warmup_steps,
        total_steps

    )

    best_accuracy=0
    patience_counter=0

    print("\n===== TRAINING =====\n")

    for epoch in range(EPOCHS):

        train_loss=train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device
        )

        val_accuracy,preds,true_labels,conf=evaluate(
            model,
            val_loader,
            device
        )

        print(f"Epoch {epoch+1}")
        print("Train Loss:",round(train_loss,4))
        print("Val Accuracy:",round(val_accuracy,4))

        # Overfitting detection

        avg_conf=np.mean(conf)

        print("Avg Confidence:",round(avg_conf,3))

        if val_accuracy>best_accuracy:

            best_accuracy=val_accuracy

            torch.save({

                "model_state_dict":model.state_dict(),
                "label_map":LABEL_MAP

            },MODEL_OUTPUT)

            print("Saved best model")

            patience_counter=0

        else:

            patience_counter+=1

        if patience_counter>=EARLY_STOPPING_PATIENCE:

            print("\nEarly stopping triggered")

            break


    print("\n===== FINAL EVALUATION =====\n")

    checkpoint=torch.load(MODEL_OUTPUT)

    model.load_state_dict(checkpoint["model_state_dict"])

    acc,preds,true_labels,conf=evaluate(
        model,
        val_loader,
        device
    )

    print("Final Accuracy:",acc)

    print("\nClassification Report:\n")

    print(classification_report(

        true_labels,
        preds,
        target_names=list(LABEL_MAP.keys())

    ))


    debug_predictions(
        val_texts,
        preds,
        true_labels,
        conf
    )

    tokenizer.save_pretrained("./tsuzi_tokenizer")

    print("\nSaved tokenizer")

    print("\nTraining Complete")


if __name__=="__main__":

    main()