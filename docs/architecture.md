# Architecture — The Voice Assistant Pipeline

This repository's seven models are not random experiments: they form one
coherent pipeline for a **voice assistant** that can *hear, understand, act,
and respond*. Each folder is a drop-in stage you can train, evaluate, and
replace independently.

## The Pipeline

```
┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  User speaks │ -> │  whisper-asr     │ -> │  distilbert-    │ -> │  miniLM-L6      │
│              │    │  (hear the user) │    │  emotion        │    │  (understand)   │
└──────────────┘    └──────────────────┘    │  (gauge feeling)│    └────────┬────────┘
                                            └─────────────────┘             │
                                                                           ▼
        ┌────────────────────────────────────────────────────────┐  ┌─────────────┐
        │   Response generator: llama-3.2-3B  /  qwen-0.5b-lora  │  │ functiongemma│
        │   (empathetic reply, grounded by bge-rag retrieval)    │  │ (tool call)  │
        └────────────────────────────────────────────────────────┘  └─────────────┘
```

### Stage by stage

| # | Stage | Model | Input → Output | Why it exists |
|---|-------|-------|----------------|---------------|
| 1 | Speech-to-text | [`whisper-asr`](../whisper-asr/) | Audio → text | The user talks, not types |
| 2 | Emotion | [`distilbert-emotion`](../distilbert-emotion/) | Text → emotion | Adapt tone to how the user feels |
| 3 | Intent routing | [`miniLM-L6`](../miniLM-L6/) | Text → intent category | Decide *what kind* of request it is |
| 4 | Function calling | [`FunctionGemma`](../FunctionGemma/) | Text → JSON tool call | Actually *do* something (set alarm, search, email) |
| 5 | Response | [`llama-3.2-3B`](../llama-3.2-3B/) / [`qwen-0.5b-lora`](../qwen-0.5b-lora/) | Text → reply | Talk back naturally |
| 6 | Grounding | [`bge-rag`](../bge-rag/) | Query → documents | Answer from your own docs, not from memory |

### Why split the pipeline?

- **Each stage is small enough to fine-tune on a consumer GPU.** The largest
  project (Llama 3.2 3B) needs ~8 GB; everything else fits in 2–6 GB.
- **Stages are independently testable.** The intent router has its own eval set;
  the function caller has an exact-match eval; Whisper reports WER.
- **Real systems are layered.** A production voice assistant *would* have a
  pre-filter, an intent router, a tool executor, and a generator — this repo is
  a faithful miniature of that architecture.

## Data Flow

Every project follows the same shape:

```
generate_dataset.py ──> dataset.jsonl ──> train.py ──> model artifact ──> inference/demo
        (seeded)            (committed)     (comments)    (git-ignored)
```

- Dataset generators are **deterministic** (fixed seed) — the committed dataset
  can always be reproduced.
- Training scripts are **self-documenting**: config at the top, comments where
  the *why* matters, and a sanity-check or eval at the end.
- Model weights are **git-ignored**; the repo ships scripts + datasets, not
  gigabytes of `.safetensors`.

## The Two LoRA Recipes

[`llama-3.2-3B/train.py`](../llama-3.2-3B/train.py) and
[`qwen-0.5b-lora/train.py`](../qwen-0.5b-lora/train.py) are the same Unsloth
recipe pointed at two different model families. The reusable pieces:

1. `FastLanguageModel.from_pretrained(..., load_in_4bit=True)` — 4-bit base.
2. `get_peft_model(r=16, alpha=32, target_modules=[...])` — LoRA on attention + MLP.
3. `get_chat_template(...)` — family-specific chat formatting.
4. `SFTTrainer` with `adamw_8bit`, cosine schedule, small batch + grad accumulation.

Switching families is literally a `MODEL_NAME` change plus the chat template.
