# 🐼 Qwen 2.5 0.5B — Lightweight Chat LoRA

Fine-tunes **`Qwen/Qwen2.5-0.5B-Instruct`** (494M params) with **LoRA via
Unsloth** — a tiny, on-device chat model that can run as a low-cost fallback or
edge deployment of the voice assistant's response stage.

| | |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` (494M) |
| Task | Text → chat reply |
| Technique | LoRA (`r=16`, `alpha=32`) via Unsloth |
| Dataset | ~100 generated instruction→reply pairs (paraphrase-augmented) |
| VRAM | ~4 GB |

This project mirrors the `llama-3.2-3B` recipe almost 1:1 — the point is to show
**how portable an Unsloth LoRA pipeline is across model families**. If you can
train one, you can train the other.

---

## The Dataset

`generate_dataset.py` combines a hand-written pool of `{instruction, response}`
pairs (how-to requests, casual Q&A, short creative tasks) with template-based
paraphrasing — each answer is asked 3–4 different ways, so the fine-tune
learns to respond to rephrasings, not just memorized prompts.

```bash
python generate_dataset.py   # → dataset.jsonl (reproducible)
```

## Training

```bash
python train.py

# Quick smoke test:
MAX_STEPS=1 python train.py
```

Same Unsloth machinery as the Llama project: 4-bit base, LoRA on all attention +
MLP projections, chat-template formatting, and an `SFTTrainer`. Hyperparameters
(r=16, alpha=32, lr 2e-4, 3 epochs) are all at the top of `train.py`.

The script ends with a **sanity-check generation** on two test prompts.

## Chat

```bash
python chat.py                 # interactive REPL
python chat.py "explain what a tensor is in simple words"
```

Loads the saved `lora_adapter/` and chats with you. Use `--model` to point at a
different checkpoint.

## Expected Behavior

0.5B is a *small* model — after LoRA fine-tuning it handles the fine-tune domain
(short, friendly answers) well, but don't expect GPT-4-level reasoning. That's
exactly why it's a good **fallback**: it's cheap, fast, and private.

---

## License

- Code in this folder: Apache-2.0 (see root `LICENSE`).
- Base model: **Apache-2.0** — see
  [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).
