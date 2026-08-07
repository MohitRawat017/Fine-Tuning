# 🔎 BGE-small — Embedding Fine-Tune + RAG Demo

Fine-tunes **`BAAI/bge-small-en-v1.5`** (33M params) with a contrastive
(InfoNCE) objective so that question–answer pairs from your domain get close
embeddings — then demonstrates **retrieval-augmented generation (RAG)** over a
tiny document collection.

| | |
|---|---|
| Base model | `BAAI/bge-small-en-v1.5` (33M) |
| Task | Sentence embedding → dense retrieval |
| Technique | Contrastive fine-tune (mean pooling + InfoNCE loss) |
| Dataset | Generated passage/question pairs |
| VRAM | ~2 GB |

This is the "ground your answers in your own documents" stage: the assistant can
answer questions *from your docs* instead of hallucinating from memory.

---

## The Dataset

`generate_dataset.py` builds a tiny but realistic collection: a set of **passages**
(how-to snippets about the fine-tuning stack in this repo) plus **questions** that
should retrieve each passage.

```bash
python generate_dataset.py   # → passages.json + qa_pairs.json (seeded)
```

## Training

```bash
python train_embeddings.py
```

The script:

1. Loads `bge-small-en-v1.5` and computes embeddings with **mean pooling**
   (BGE's recommended pooling) + L2 normalization.
2. Trains with a **contrastive InfoNCE loss**: each question's embedding is
   pulled toward its positive passage and pushed away from the other passages
   in the batch.
3. Saves the fine-tuned encoder to `bge-ft/`.

No GPU needed for a demo-scale run (the model is tiny), but CUDA is used when
available.

## Retrieval Demo

```bash
python retrieval_demo.py "how do i free up vram when training a lora"
```

Builds a **NumPy-based index** over your passages and returns the top-K with
scores (add `faiss-cpu` to scale to larger corpora). The README of this
project's parent pipeline describes how this plugs into an LLM for RAG.

---

## License

- Code in this folder: Apache-2.0 (see root `LICENSE`).
- Base model: **MIT** — see
  [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5).
