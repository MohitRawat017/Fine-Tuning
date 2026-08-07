"""Transcribe an audio file with the fine-tuned (or base) Whisper model."""

import argparse

import torch
from transformers import (
    pipeline,
)


def main():
    parser = argparse.ArgumentParser(description="Transcribe an audio file with Whisper.")
    parser.add_argument("audio", help="Path to an audio file (wav/mp3/flac/m4a).")
    parser.add_argument(
        "--model",
        default="openai/whisper-tiny",
        help="HF model id or local folder (e.g. 'whisper-ft' for the fine-tuned one).",
    )
    parser.add_argument("--language", default="en", help="Spoken language code.")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"Loading model '{args.model}' on {device}...")
    pipe = pipeline(
        "automatic-speech-recognition",
        model=args.model,
        device=device,
        generate_kwargs={"language": args.language, "task": "transcribe"},
    )

    print(f"Transcribing {args.audio}...")
    result = pipe(args.audio)
    print("\n" + "=" * 50)
    print("TRANSCRIPTION")
    print("=" * 50)
    print(result["text"].strip())
    print("=" * 50)


if __name__ == "__main__":
    main()
