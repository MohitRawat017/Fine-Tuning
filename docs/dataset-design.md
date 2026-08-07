# Dataset Design

Every project in this repo ships its dataset **and** the script that generated
it. This document explains the design decisions behind each one.

---

## 1. `miniLM-L6/dataset.jsonl` — Intent queries (400 rows)

| Field | Meaning |
|---|---|
| `text` | A voice-style query |
| `label` | Intent name |
| `label_id` | Intent id (casual=0 … communication=4) |

**Design choices**

- **80 examples × 5 intents** — perfectly balanced, so accuracy is a fair
  headline metric.
- **Hand-crafted, not scraped.** Queries deliberately include the noise a real
  ASR pipeline produces:
  - hesitations: *"hey um start a timer for 10 minutes please"*
  - typos: *"set alram 7"*, *"wat are my todos"*
  - Indian-English phrasings: *"kindly send email"*, *"do the needful"*, *"prepone meeting"*
- **No tool names in queries** — the classifier must learn intent from natural
  language, not keyword matching (`"set_alarm"` never appears).

**Why it matters:** a classifier trained on clean, canonical queries falls apart
on the messiness of real speech. This dataset bakes robustness in from the start.

---

## 2. `FunctionGemma/functiongemma_dataset_original.jsonl` → `functiongemma_dataset.jsonl` (520 rows)

| Field | Meaning |
|---|---|
| `messages[0]` | `developer` — the available-tools schema for this example |
| `messages[1]` | `user` — the natural-language query |
| `messages[2]` | `assistant` — the expected JSON tool call |

**Design choices**

- **40 examples × 13 tools** across 4 categories — enough variety per tool for
  the model to learn arg extraction without memorizing.
- **Queries include typos and casual phrasing** (same rationale as MiniLM).
- **Two dataset files, one pipeline:**
  - `generate_dataset.py` writes the raw output with *human-friendly* signatures
    (e.g. `send_email(recipient, subject?, message?)`).
  - `fix_dataset.py` reconciles them to the **real Pydantic schemas** the tool
    layer actually uses (`send_email(subject, body)`, `read_emails(count?,
    filter_type?)`, `create_calendar_event(..., duration?)`,
    `search_arxiv(..., max_results?)`) and remaps assistant-output arg names.
  - The canonical `functiongemma_dataset.jsonl` is what `train.py` reads.

**Why the fix step exists:** the training data must match the *production*
schemas, or the fine-tuned model will emit arguments the tool executor can't
validate. This is a common real-world data-pipeline bug, captured here as a
reproducible fix script instead of a silent mistake.

---

## 3. `llama-3.2-3B/dataset/combined_dataset.json` — Support conversations (~1.8k rows)

| Field | Meaning |
|---|---|
| `Context` | A user's message (often expressing distress) |
| `Response` | A supportive reply |

**Design choices**

- **Multiple responses per context is intentional.** Several different
  supportive replies to the same situation teach the model that there is no
  single "correct" answer — it learns the *tone*, not a lookup table.
- **Context-aware split.** `train.py` groups rows by `Context` *before* the
  90/10 split, so eval only contains situations never seen in training. A naive
  row-level split would leak the same context into both sets and flatter the
  eval loss.
- **Empty responses are filtered** before training (blank assistant turns are
  malformed training data).

**⚠️ Data note:** this dataset contains real mental-health conversations.
Provenance, license, and ethical-use guidance live in
[safety.md](safety.md#the-llama-32-3b-training-data).

---

## 4. `whisper-asr` — LibriSpeech (streamed from HF)

Loaded on the fly via `datasets` — `train.clean.100` for training,
`test.clean` for WER evaluation. Short, clean utterances are the right shape
for a voice assistant. A `MAX_TRAIN_SAMPLES` cap keeps the recipe runnable.

## 5. `distilbert-emotion` — `dair-ai/emotion` (streamed from HF)

20k tweets labeled `sadness, joy, love, anger, fear, surprise`. Standard
benchmark for emotion classification; no download step required.

## 6. `qwen-0.5b-lora/dataset.jsonl` — Generated chat pairs (~2.4k rows)

`generate_dataset.py` expands a compact pool of `{instruction, response}` pairs
with **paraphrased prompt variants** — the same answer asked 2–3 ways. Cheap
data augmentation that makes the small model robust to rephrasing.

## 7. `bge-rag` — `passages.json` + `qa_pairs.json`

A small document collection about this very repo's fine-tuning stack, plus
question→passage pairs. Each question's positive passage is known, which is
exactly what the contrastive (InfoNCE) objective needs.

---

## Reproducibility

All hand-written generators (`miniLM-L6`, `FunctionGemma`, `qwen-0.5b-lora`,
`bge-rag`) call `random.seed(...)` at the top, so re-running produces the exact
same dataset. The committed dataset files are the source of truth; the
generators exist so you can tweak and regenerate.
