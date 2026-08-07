"""Chat with the fine-tuned Qwen LoRA adapter."""

import argparse

import torch
from unsloth import FastLanguageModel

MAX_SEQ_LENGTH = 1024


def build_prompt(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )


def main():
    parser = argparse.ArgumentParser(description="Chat with the fine-tuned Qwen model.")
    parser.add_argument("prompt", nargs="?", help="Optional: single prompt, then exit.")
    parser.add_argument(
        "--model", default="lora_adapter", help="HF model id or local LoRA folder."
    )
    parser.add_argument("--max-new-tokens", type=int, default=200)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading '{args.model}' on {device}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    def respond(messages):
        inputs = build_prompt(tokenizer, messages).to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True).strip()

    if args.prompt:
        print("\n" + respond([{"role": "user", "content": args.prompt}]))
        return

    print("Chat mode — type 'quit' to exit.\n")
    history = []
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})
        reply = respond(history)
        history.append({"role": "assistant", "content": reply})
        print(f"Assistant: {reply}\n")


if __name__ == "__main__":
    main()
