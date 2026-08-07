# Handover: LoRA Fine-Tuning for Depression-Support Voice Assistant
## Project Context
This model will power the response-generation component of a voice interface
designed for users who may have depression. The broader system includes an
adaptive pause-detection layer (separate from this task) that decides when to
respond; this fine-tune is only responsible for generating the actual reply
text once triggered.
## Task
Fine-tune a small open-source LLM using **LoRA** via **Unsloth**, to adapt its
tone and response style for supportive conversations with this user group.
This is a lightweight adaptation on top of an existing instruct model, not
training from scratch.
## Base Model
- **meta-llama/Llama-3.2-3B-Instruct**
- Load via Unsloth's 4-bit quantized version for faster/lower-memory training
  (e.g. `unsloth/Llama-3.2-3B-Instruct-bnb-4bit`)
## Dataset
- Format: rows of `context` + `response` pairs (not yet in chat-template
  format — will need conversion to Llama 3.2's chat template before training)
- Size: not yet specified — script should include a quick dataset stats check
  (row count, token length distribution) before training starts, since small
  mental-health datasets are prone to overfitting past 2–3 epochs
- Location: local file (path TBD by user)
## Environment
- **Local GPU** (exact VRAM not yet specified by user — script should target
  a single consumer GPU, e.g. work within ~8–16GB VRAM using 4-bit
  quantization + gradient checkpointing, and note where settings would change
  for more/less VRAM)
- Single-GPU training, no distributed setup needed
## Requested Script Requirements
1. Load base model + tokenizer via Unsloth (4-bit)
2. Convert context/response rows into Llama 3.2 chat template format
3. Apply LoRA config: rank 16–32, alpha = 2x rank, target modules at minimum
   `q_proj, k_proj, v_proj, o_proj`
4. Train with `SFTTrainer` (trl), learning rate ~2e-4, 2–3 epochs, with
   eval/loss logging to watch for overfitting
5. Save LoRA adapter (and optionally a merged model) locally
6. Include a small inference snippet at the end to sanity-check outputs
   after training
## Safety Note (important — do not skip)
This model will be used by people who may be experiencing depression.
Before any deployment:
- The script/notebook should leave room for a manual red-team pass: test
  model outputs against prompts expressing hopelessness, self-harm ideation,
  or "what's the point" style statements. This is not a training-data
  requirement, just a post-training eval step to flag before this goes near
  real users.
- This fine-tune is not a crisis-response system. Crisis detection/escalation
  (e.g. routing self-harm mentions to a human or hotline) is a separate
  system layer and is out of scope for this script, but should not be
  assumed to be "handled" by the fine-tuned model itself.
## What I need from you
A working Unsloth + LoRA training script matching the above, with inline
comments explaining each step (I'm following along, not just running it
blind).
