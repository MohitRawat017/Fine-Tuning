"""Generate the passages.json corpus and qa_pairs.json retrieval pairs."""

import json

# Passages — each has an id so qa_pairs can reference it.
PASSAGES = [
    {
        "id": 0,
        "title": "LoRA fine-tuning",
        "text": "LoRA (Low-Rank Adaptation) fine-tunes only small rank-decomposed matrices added to the attention projections, instead of updating every weight. It needs far less VRAM and memory than full fine-tuning while keeping most of the quality.",
    },
    {
        "id": 1,
        "title": "Unsloth and 4-bit quantized models",
        "text": "Unsloth loads models in 4-bit via bitsandbytes so an 8GB consumer GPU can fine-tune models like Llama 3.2 3B. It also provides a fused, memory-optimized SFTTrainer that reduces VRAM further.",
    },
    {
        "id": 2,
        "title": "Gradient checkpointing",
        "text": "Gradient checkpointing trades a little compute for a lot of memory: instead of keeping every activation for backprop, it recomputes them on the fly. Unsloth's variant is tuned for LoRA training.",
    },
    {
        "id": 3,
        "title": "Chat templates",
        "text": "Chat templates turn a list of role/content messages into the exact token format a model expects. Llama and Qwen use different templates, which is why tokenizers ship a chat_template these days.",
    },
    {
        "id": 4,
        "title": "Mean pooling for sentence embeddings",
        "text": "Mean pooling averages the token embeddings of a sentence (masking padding) into a single vector. Sentence-transformers models like MiniLM and BGE use it to produce a good sentence representation.",
    },
    {
        "id": 5,
        "title": "Contrastive learning for retrieval",
        "text": "Contrastive objectives pull a query embedding close to its positive document and push it away from negatives in the same batch. This is how embedding models learn a retrieval-friendly space.",
    },
]

# Each question should retrieve exactly its positive passage.
QA_PAIRS = [
    {"question": "how do i fine-tune a model without much vram", "positive_passage_id": 0},
    {"question": "what is lora in plain terms", "positive_passage_id": 0},
    {"question": "why does lora use less memory", "positive_passage_id": 0},
    {"question": "can an 8gb gpu fine-tune llama 3.2", "positive_passage_id": 1},
    {"question": "what does unsloth do differently", "positive_passage_id": 1},
    {"question": "what is 4-bit quantization for", "positive_passage_id": 1},
    {"question": "how does gradient checkpointing save memory", "positive_passage_id": 2},
    {"question": "why do models recompute activations during training", "positive_passage_id": 2},
    {"question": "what is a chat template", "positive_passage_id": 3},
    {"question": "why does llama need a different template than qwen", "positive_passage_id": 3},
    {"question": "how do i turn tokens into one sentence vector", "positive_passage_id": 4},
    {"question": "what pooling does bge use", "positive_passage_id": 4},
    {"question": "how do embedding models learn to retrieve", "positive_passage_id": 5},
    {"question": "what is an infonce loss", "positive_passage_id": 5},
]


def main():
    with open("passages.json", "w", encoding="utf-8") as fh:
        json.dump(PASSAGES, fh, indent=2, ensure_ascii=False)
    print("Saved: passages.json")

    with open("qa_pairs.json", "w", encoding="utf-8") as fh:
        json.dump(QA_PAIRS, fh, indent=2, ensure_ascii=False)
    print("Saved: qa_pairs.json")


if __name__ == "__main__":
    main()
