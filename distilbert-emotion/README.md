# 🎭 DistilBERT Emotion — Emotion Classifier

Fine-tunes **`distilbert-base-uncased`** (67M params) to classify short text into
one of **6 emotions** — the "gauge how the user is feeling" stage of the
voice-assistant pipeline. It pairs naturally with the support-companion Llama
model: knowing the user's emotional state lets the assistant adapt its tone.

| | |
|---|---|
| Base model | `distilbert-base-uncased` (67M) |
| Task | Text → emotion (6 classes) |
| Technique | Classification head (pooler output) |
| Dataset | `dair-ai/emotion` (20k tweets, 6 emotions) |
| VRAM | ~2 GB |

Emotion classes: `sadness, joy, love, anger, fear, surprise`.

---

## Dataset

Uses the widely-used **`dair-ai/emotion`** dataset (tweets labeled with 6
emotions), loaded on the fly from Hugging Face. No manual download required.

- `train` split: 16,000 examples
- `test` split: 2,000 examples (for final evaluation)

## Training

```bash
python train.py
```

The script:

1. Loads the tokenizer and model with `AutoModelForSequenceClassification`.
2. **Freezes the encoder** by default and trains only the classification head —
   fast and low-VRAM, with near-full accuracy on this dataset. Set
   `FREEZE_ENCODER = False` to fine-tune the whole model for slightly better
   accuracy.
3. Trains with the HF `Trainer` and reports **accuracy + F1** on the test set.

### Expected results

With a frozen encoder: **~85–88%** test accuracy. With full fine-tuning:
**~90%+**. This is a classic "head-only vs full fine-tune" comparison — both
configs are one flag apart.

## Inference

```bash
python predict.py "I feel so lonely today"
python predict.py "This is the best news ever!"
```

Prints the predicted emotion with confidence scores across all 6 classes.

---

## License

- Code in this folder: Apache-2.0 (see root `LICENSE`).
- Base model: **Apache-2.0** — see
  [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased).
- Dataset: `dair-ai/emotion` (CC BY 4.0).
