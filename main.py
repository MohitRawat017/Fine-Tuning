"""
Fine-Tuning Playground - CLI entry point.

Lists every model project in the repo and runs lightweight, dependency-free
sanity checks (dataset shapes, config values). Heavy work (training, inference)
lives in each project folder - this script is the map, not the engine.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# name -> (folder, task, base_model, kind)
# NOTE: keep this file ASCII-only - Windows consoles default to cp1252.
MODELS = {
    "whisper-asr": {
        "folder": "whisper-asr",
        "task": "Speech to text",
        "base_model": "openai/whisper-tiny",
        "kind": "Full fine-tune (Seq2Seq)",
        "files": ("train_whisper.py", "transcribe.py"),
    },
    "distilbert-emotion": {
        "folder": "distilbert-emotion",
        "task": "Text to emotion",
        "base_model": "distilbert-base-uncased",
        "kind": "Classification head",
        "files": ("train.py", "predict.py"),
    },
    "miniLM-L6": {
        "folder": "miniLM-L6",
        "task": "Text to intent",
        "base_model": "sentence-transformers/all-MiniLM-L6-v2",
        "kind": "Classification head + pooling",
        "files": ("train_minilm.py", "inference.py"),
    },
    "functiongemma": {
        "folder": "FunctionGemma",
        "task": "Text to tool call (JSON)",
        "base_model": "google/functiongemma-270m-it",
        "kind": "LoRA",
        "files": ("train.py",),
    },
    "llama-3.2-3B": {
        "folder": "llama-3.2-3B",
        "task": "Text to empathetic reply",
        "base_model": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        "kind": "LoRA (4-bit via Unsloth)",
        "files": ("train.py", "demo.py"),
    },
    "qwen-0.5b-lora": {
        "folder": "qwen-0.5b-lora",
        "task": "Text to chat reply",
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "kind": "LoRA",
        "files": ("train.py", "chat.py"),
    },
    "bge-rag": {
        "folder": "bge-rag",
        "task": "Retrieval over documents",
        "base_model": "BAAI/bge-small-en-v1.5",
        "kind": "Contrastive fine-tune + FAISS index",
        "files": ("train_embeddings.py", "retrieval_demo.py"),
    },
}


def list_models():
    """Pretty-print every project with its folder location."""
    width = max(len(name) for name in MODELS)
    print("=" * 78)
    print("Fine-Tuning Playground - available models")
    print("=" * 78)
    for name, info in MODELS.items():
        folder = REPO_ROOT / info["folder"]
        print(f"\n  {name:<{width}}  {info['task']}")
        print(f"  {'':<{width}}  base: {info['base_model']}")
        print(f"  {'':<{width}}  tech: {info['kind']}")
        print(f"  {'':<{width}}  path: {folder}")


def describe(name: str) -> int:
    """Print one model's details plus the shape of its dataset (if any)."""
    if name not in MODELS:
        print(f"Unknown model '{name}'. Use one of: {', '.join(MODELS)}")
        return 1

    info = MODELS[name]
    print(f"\n=== {name} - {info['task']} ===")
    print(f"  base model : {info['base_model']}")
    print(f"  technique  : {info['kind']}")
    print(f"  folder     : {REPO_ROOT / info['folder']}")

    # Quick dataset sanity check - no heavy imports, just JSONL counting.
    folder = REPO_ROOT / info["folder"]
    for ds_name in ("dataset.jsonl", "functiongemma_dataset.jsonl", "tsuzi_intent_dataset.jsonl"):
        ds = folder / ds_name
        if not ds.is_file():
            continue
        n = sum(1 for _ in ds.open(encoding="utf-8") if _.strip())
        print(f"  dataset    : {ds_name} ({n} rows)")
        break

    for script in info["files"]:
        if (folder / script).is_file():
            print(f"  script     : {script}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="fine-tuning",
        description="Fine-Tuning Playground - explore all 7 fine-tuning projects.",
    )
    parser.add_argument(
        "model",
        nargs="?",
        help="Model name to inspect (see list). Omit to list everything.",
    )
    args = parser.parse_args()

    if args.model:
        return describe(args.model)

    list_models()
    print(
        "\n\nRun 'python main.py <model>' for details on a single project.\n"
        "Each project folder has its own README with training commands."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
