---
base_model: unsloth/llama-3.2-3b-instruct-bnb-4bit
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- sft
- unsloth
- transformers
- trl
- mental-health
- empathetic
---

# Llama 3.2 3B — Support Companion (LoRA Adapter)

A LoRA adapter fine-tuned on top of **Llama 3.2 3B Instruct** (4-bit) to produce
warm, empathetic, non-judgmental replies for supportive conversations.

This folder contains **only the adapter weights** (~50–100 MB). To use it you
load the base model from Hugging Face and apply this adapter — see the
[project README](../README.md) for the exact commands and the Gradio demo.

## Model Details

- **Base model:** `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` (4-bit quantized)
- **Adapter type:** LoRA (`peft_type: LORA`, `r=16`, `lora_alpha=32`, dropout=0)
- **Target modules:** `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- **Training:** 2 epochs, lr 2e-4, cosine schedule, `adamw_8bit`
- **Framework versions:** PEFT 0.19.1 · TRL 0.19.1 · Transformers 4.53.3

## Intended Use

Generating supportive, validating replies for users who may be experiencing
depression or emotional distress — e.g. the response stage of a voice assistant.

### Out-of-Scope / Safety

⚠️ **This adapter is NOT a crisis-response system.**

- It cannot be relied upon to route users to hotlines or emergency services.
- An external safety layer must detect high-risk input (self-harm ideation)
  *before* the model is invoked.
- It should not be used to give medical, therapeutic, or clinical advice.

Before any deployment, run the red-team checklist in
[`train.py`](../train.py#red-team-checklist) with self-harm and hopelessness
prompts, and read [`docs/safety.md`](../../docs/safety.md).

## Training Data

3,512 `{Context, Response}` pairs (995 unique contexts) of supportive conversations. The dataset is
published in [`dataset/combined_dataset.json`](../dataset/combined_dataset.json)
for research/educational purposes; see the provenance and ethics note in
[`docs/safety.md`](../../docs/safety.md).

## Evaluation

The training run evaluates validation loss every 50 steps. For real-world
quality assessment, use the manual red-team checklist — automated metrics are
insufficient for a mental-health-facing model.

## Contact

Part of the [Fine-Tuning Playground](https://github.com/<your-username>/fine-tuning) repository.
