import gradio as gr
import torch
from unsloth import FastLanguageModel

ADAPTER_DIR = "lora_adapter"
MAX_SEQ_LENGTH = 1024

SYSTEM_PROMPT = (
    "You are a warm, empathetic, and non-judgmental support companion. "
    "The user may be experiencing symptoms of depression. Your goal is to listen actively, "
    "validate their feelings, and offer gentle, supportive encouragement. "
    "You are not a licensed therapist or medical professional. "
    "Keep responses concise, conversational, and safe. "
    "If the user expresses intent to harm themselves, gently encourage them to contact a local crisis hotline or emergency services."
)


print("Loading fine-tuned adapter...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=ADAPTER_DIR,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)


def chat(message, history, max_new_tokens, temperature, top_p):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            do_sample=True,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True).strip()


demo = gr.ChatInterface(
    fn=chat,
    type="messages",
    title="Support Companion Demo",
    description="Local demo of the fine-tuned LoRA adapter. Demo only; crisis detection and escalation are not included.",
    examples=[
        "I feel exhausted and I don't know how to start the day.",
        "I keep thinking I am not good enough.",
        "I had a rough week and I feel really alone.",
    ],
    additional_inputs=[
        gr.Slider(32, 256, value=120, step=8, label="Max new tokens"),
        gr.Slider(0.1, 1.2, value=0.7, step=0.05, label="Temperature"),
        gr.Slider(0.5, 1.0, value=0.9, step=0.05, label="Top-p"),
    ],
)


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=7860)
