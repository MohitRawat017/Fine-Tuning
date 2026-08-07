<div align="center">

# 🧠 Fine-Tuning Playground

**A hands-on showcase of open-weight model fine-tuning — from tiny classifiers to small LLMs.**

Fine-tune, evaluate, and demo **7 open-source models** across 4 tasks, all on a single consumer GPU.
This repo is built around a real product story: a **voice assistant pipeline** — *hear it → understand it → act on it → respond to it*.

</div>

---

## 📖 Table of Contents

- [What's Inside](#-whats-inside)
- [The Big Picture](#-the-big-picture)
- [Models on Display](#-models-on-display)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [How Each Model Was Fine-Tuned](#-how-each-model-was-fine-tuned)
- [Safety & Responsible Use](#-safety--responsible-use)
- [Roadmap](#-roadmap)
- [License](#-license)

---

## 🗺️ What's Inside

This repository contains **complete, reproducible fine-tuning projects** for 7 models. Every project ships with:

| Ingredient | What you get |
|---|---|
| 🏋️ **Training script** | Complete, commented, configurable via env vars / constants |
| 📊 **Dataset generator** | Deterministic (seeded) script that reproduces the exact dataset |
| 🔮 **Inference / demo** | Run the fine-tuned model, chat with it, or query it |
| 📄 **README** | Task description, results, hyperparameters, and how to run |

**No API keys. No hidden datasets. No black boxes.** Everything is in the repo.

---

## 🎯 The Big Picture

All the models in this repo form one coherent pipeline — a **voice assistant** that:

```
🎤  Speech input
 │
 ▼
🧠  whisper-asr          (NEW: hear the user)
 │  Speech → Text
 ▼
🔀  distilbert-emotion   (NEW: gauge the user's emotional state)
 │  Text → Sentiment/Emotion
 ▼
🧭  miniLM-L6            (intent router)
 │  Text → Intent category (casual / productivity / system / research / communication)
 ▼
⚙️  functiongemma-270m   (function caller)
 │  Intent → Structured tool call (JSON)
 ▼
💬  llama-3.2-3B         (response generator)
 │  Intent + Tool result → Empathetic reply
 ▼
🧠  qwen-0.5b-lora       (NEW: lightweight on-device chat fallback)
 │
▼
🔎  bge-rag              (NEW: retrieval over your documents for grounded answers)
```

Each model is independently fine-tunable — pick the stage you want to learn from, or train the whole stack.

---

## 🤖 Models on Display

| Model | Task | Base Model | Technique | VRAM (train) |
|---|---|---|---|---|
| [`whisper-asr`](whisper-asr/) | Speech → Text | `openai/whisper-tiny` | Full fine-tune (Seq2Seq) | ~4 GB |
| [`distilbert-emotion`](distilbert-emotion/) | Text → Emotion | `distilbert-base-uncased` | Classification head | ~2 GB |
| [`miniLM-L6`](miniLM-L6/) | Text → Intent | `sentence-transformers/all-MiniLM-L6-v2` | Classification head + pooling | ~2 GB |
| [`functiongemma`](FunctionGemma/) | Text → Tool call (JSON) | `google/functiongemma-270m-it` | LoRA | ~6 GB |
| [`llama-3.2-3B`](llama-3.2-3B/) | Text → Empathetic reply | `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` | LoRA (4-bit via Unsloth) | ~8 GB |
| [`qwen-0.5b-lora`](qwen-0.5b-lora/) | Text → Chat reply | `Qwen/Qwen2.5-0.5B-Instruct` | LoRA | ~4 GB |
| [`bge-rag`](bge-rag/) | Retrieval over docs | `BAAI/bge-small-en-v1.5` | Contrastive fine-tune + FAISS index | ~2 GB |

> **Tip:** The two LoRA projects (`llama-3.2-3B`, `qwen-0.5b-lora`) use **Unsloth**, which is up to 2× faster and uses ~70% less VRAM than standard fine-tuning.

---

## 📂 Repository Structure

```
fine-tuning/
├── FunctionGemma/          # 270M function-calling model → structured tool calls
├── llama-3.2-3B/           # 3B support-companion LLM → empathetic replies (LoRA)
├── miniLM-L6/              # 22M intent router → query classification
├── whisper-asr/            # Speech-to-text fine-tune
├── distilbert-emotion/     # Emotion/sentiment classifier
├── qwen-0.5b-lora/         # Small chat LLM fine-tune (LoRA)
├── bge-rag/                # Embedding fine-tune + retrieval demo
├── docs/                   # Architecture, dataset design, safety notes
├── main.py                 # CLI entry point — explore every model
├── pyproject.toml          # Project metadata + pinned deps (uv)
├── requirements.txt        # Quick-install deps
└── uv.lock                 # Locked environment (uv)
```

---

## 🚀 Getting Started

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/fine-tuning.git
cd fine-tuning

# Option A — uv (recommended, matches the lockfile)
uv sync

# Option B — pip
pip install -r requirements.txt
```

> **Windows + PyTorch CUDA note:** the pinned `torch==2.6.0+cu124` wheel comes from the official PyTorch index.
> With `uv` this is configured automatically in `pyproject.toml`. With plain pip use:
> `pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124`

### 2. Explore from the CLI

```bash
python main.py                # list all 7 models
python main.py miniLM-L6      # show one model's details + dataset stats
```

### 3. Run a full project (example: MiniLM intent router)

```bash
cd miniLM-L6
python generate_dataset.py     # reproduce dataset.jsonl (seeded)
python train_minilm.py         # train on GPU (falls back to CPU)
python inference.py            # interactive intent classification
```

Each project folder has its own `README.md` with exact commands.

---

## 🔬 How Each Model Was Fine-Tuned

| Project | Data | Loss / Objective | Key hyperparameters |
|---|---|---|---|
| `whisper-asr` | LibriSpeech subset | CTC / cross-entropy on tokens | 2 epochs, lr 1e-5, bs 8 |
| `distilbert-emotion` | dair-ai/emotion | Cross-entropy (6 classes) | 3 epochs, lr 2e-5, bs 32 |
| `miniLM-L6` | 400 hand-crafted queries (5 intents) | Cross-entropy + label smoothing | 5 epochs, lr 1e-5, early stopping |
| `functiongemma` | 520 function-call examples (12 tools) | LM on assistant turn only | 6 epochs, lr 1e-5, LoRA r=8 |
| `llama-3.2-3B` | 3.5k context→response pairs (995 contexts) | SFT (LM loss) | 2 epochs, lr 2e-4, LoRA r=16, 4-bit |
| `qwen-0.5b-lora` | Generated instruction set | SFT (LM loss) | 3 epochs, lr 2e-4, LoRA r=16 |
| `bge-rag` | Generated passage/question pairs | Contrastive (InfoNCE) | 5 epochs, lr 2e-5 |

> Full details — data formats, preprocessing, rationale — live in [`docs/dataset-design.md`](docs/dataset-design.md).

---

## 🛡️ Safety & Responsible Use

⚠️ **The `llama-3.2-3B` project fine-tunes a support-companion model that users may turn to when experiencing depression.**

- This model is **not a crisis-response system** and cannot reliably route users to hotlines.
- An **external crisis-detection layer** must intercept high-risk keywords *before* they reach this model.
- Before any real-world deployment, run the manual **red-team checklist** included at the bottom of `llama-3.2-3B/train.py`.
- The training data is published as-is with a provenance note — see [`docs/safety.md`](docs/safety.md).

Please read [`docs/safety.md`](docs/safety.md) in full before using any part of this repository.

---

## 🗓️ Roadmap

- [x] Core pipeline: intent → function-call → reply (`miniLM-L6`, `functiongemma`, `llama-3.2-3B`)
- [x] Speech input stage (`whisper-asr`)
- [x] Emotional state stage (`distilbert-emotion`)
- [x] Lightweight chat fallback (`qwen-0.5b-lora`)
- [x] Grounded retrieval (`bge-rag`)
- [ ] End-to-end pipeline orchestration script
- [ ] ONNX / quantization export for all models
- [ ] Docker + Gradio all-in-one demo

---

## 📜 License

This project is licensed under the **Apache License 2.0** — see [LICENSE](LICENSE).

**Base model licenses apply** — check each model's Hugging Face page (Gemma, Llama, Qwen, Whisper, MiniLM, BGE, DistilBERT) for their respective terms before commercial use.
