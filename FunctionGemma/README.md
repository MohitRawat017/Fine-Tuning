# ⚙️ FunctionGemma — Function Calling Fine-Tune

Fine-tunes **`google/functiongemma-270m-it`** (a 270M-parameter instruction-tuned
model) to turn natural-language voice queries into **structured tool calls** — the
"act on it" stage of the voice-assistant pipeline.

```
"set an alarm for 7am"  →  {"tool": "set_alarm", "args": {"time": "7am"}}
```

| | |
|---|---|
| Base model | `google/functiongemma-270m-it` |
| Task | Text → tool call (JSON) |
| Technique | LoRA (`r=8`, `alpha=16`) |
| Dataset | 520 hand-written examples across **13 tools** (40 each) |
| VRAM | ~6 GB (4-bit quantized base) |

---

## The Dataset

`generate_dataset.py` produces 520 deterministic examples covering 4 categories
and 13 tools:

| Category | Tools |
|---|---|
| 🗓️ Productivity | `set_alarm`, `set_timer`, `add_task`, `get_tasks`, `create_calendar_event` |
| 💻 System | `open_app`, `run_command`, `get_system_info` |
| 🔬 Research | `web_search`, `search_stackoverflow`, `search_arxiv` |
| ✉️ Communication | `send_email`, `read_emails` |

Each example is a chat record: **developer** (available tools) → **user** (query)
→ **assistant** (expected JSON tool call). Queries include typos and casual
phrasings so the model learns robustness, not just templates.

### Why two dataset files?

| File | Contents |
|---|---|
| `functiongemma_dataset_original.jsonl` | Raw generator output (human-friendly signatures) |
| `functiongemma_dataset.jsonl` | **Canonical** — reconciled to real Pydantic schemas via `fix_dataset.py` |

The generator intentionally writes the `*_original` name so you never clobber
the canonical file. Regenerate and fix like this:

```bash
python generate_dataset.py      # → functiongemma_dataset_original.jsonl
python fix_dataset.py           # → functiongemma_dataset.jsonl (canonical)
```

---

## Training

```bash
# Optional environment overrides (defaults shown)
export MODEL_NAME=google/functiongemma-270m-it
export BATCH_SIZE=4
export EPOCHS=6
export LEARNING_RATE=1e-5
export USE_PEFT=1              # 0 = full fine-tune (more VRAM)

python train.py
```

The trainer:

- **Masks prompt tokens** — the loss only trains on the assistant's JSON output,
  so the model learns to *produce* tool calls, not parrot the schema.
- Uses `DataCollatorForSeq2Seq` with padding to multiples of 8.
- Saves the best checkpoint by validation loss (`load_best_model_at_end`).
- Runs a **generation exact-match eval** on up to 50 val examples after training
  and writes `metrics.json` (exact-match + final eval loss) next to the model.

Outputs land in `fg_finetuned_ckpt/` (git-ignored).

### What to expect

On the 90/10 split, the base model already handles well-formed queries; the
fine-tune improves accuracy on casual/typo'd variants and teaches the corrected
`send_email(subject, body)` / `read_emails(count?, filter_type?)` signatures.
Exact-match on the generation eval is the headline metric — aim for **>90%**.

---

## License

- Code in this folder: Apache-2.0 (see root `LICENSE`).
- Base model: Gemma terms of use — check
  [google/functiongemma-270m-it](https://huggingface.co/google/functiongemma-270m-it) before commercial use.
