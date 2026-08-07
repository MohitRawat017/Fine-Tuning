"""Generate the {instruction, response} dataset for Qwen chat fine-tuning."""

import json
import random

# Hand-written (instruction, response) pool.
EXAMPLES = [
    # How-to / factual
    ("how do I boil an egg", "Fill a pot with water, bring it to a gentle boil, lower the egg in with a spoon, and cook for about 7 minutes for a soft yolk. Cool it under cold water before peeling."),
    ("what is a python decorator", "A decorator is a function that takes another function and extends its behavior without changing its code. You apply it with the @ symbol above the function definition."),
    ("explain machine learning in one sentence", "Machine learning is the practice of teaching a program to make predictions or decisions by learning patterns from data instead of following explicit rules."),
    ("how do I back up my phone photos", "Enable automatic cloud backup in your phone's settings, or connect it to a computer and copy the DCIM folder. Doing both gives you a safe second copy."),
    ("what is the capital of japan", "Tokyo is the capital of Japan and one of the largest metropolitan areas in the world."),
    ("how do i improve my sleep", "Keep a consistent sleep schedule, dim screens an hour before bed, keep the room cool and dark, and avoid caffeine after mid-afternoon. Small consistent changes beat occasional perfect nights."),
    ("what does git stash do", "git stash temporarily shelves your uncommitted changes so you can switch branches cleanly, then you can restore them later with git stash pop."),
    ("how do i make coffee without a machine", "Use a simple pour-over: put ground coffee in a filter, slowly pour hot water over it, and let it drip. It needs nothing more than hot water, grounds, and a filter."),
    ("what is a lora adapter", "LoRA adapters are small trainable matrices added to a frozen model's weights. Fine-tuning them is far cheaper than full fine-tuning, and they can be merged back or swapped at inference time."),
    ("what is gradient checkpointing", "Gradient checkpointing trades a little extra compute for much lower memory: instead of keeping every activation for backprop, the model recomputes them on the fly during the backward pass."),
    ("how do i install pytorch", "The easiest way is pip: 'pip install torch'. On Windows with an NVIDIA GPU, use the official CUDA wheel: pip install torch --index-url https://download.pytorch.org/whl/cu124."),
    ("what does normalization mean in deep learning", "Normalization rescales activations so their mean and variance stay stable across layers, which helps gradients flow and lets you train faster with higher learning rates."),

    # Casual conversation
    ("i had a rough day", "I'm sorry to hear that. Give yourself permission to rest for a bit — even ten quiet minutes can help reset. Want to talk about what happened?"),
    ("tell me a fun fact", "Octopuses have three hearts, and two of them stop beating when they swim. That's part of why they prefer crawling along the ocean floor."),
    ("what should i eat for lunch", "Something with protein, a vegetable, and a bit of carb keeps you full through the afternoon — a grain bowl, a wrap, or leftovers all work well."),
    ("i can't focus today", "Try the two-minute rule: commit to just two minutes of the task. Often starting is the hardest part, and momentum does the rest."),
    ("any good movie recommendations", "If you like sci-fi with heart, 'Arrival' is a great pick. For something lighter, 'The Grand Budapest Hotel' is charming and visually stunning."),
    ("i'm learning to code, any tips", "Pick one small project and finish it end to end, even if it's tiny. Building beats reading — and don't be afraid to read other people's code."),
    ("how was your day", "I don't experience days the way people do, but I'm always happy to help with yours. What's on your mind?"),
    ("i feel tired of studying", "That's normal — studying is genuinely draining. Take a real break for 15 minutes: walk, stretch, drink water. Your brain consolidates better after rest."),

    # Creative writing
    ("write a haiku about autumn", "Leaves drift to the ground,\nCool air carries golden light,\nQuiet turns to rest."),
    ("give me a catchy name for a cat cafe", "Purr & Pour, Whisker Bean, or Catfeine — they all hint at coffee plus cats in a playful way."),
    ("write a two line poem about rain", "The sky lets go its patient grey,\nand thirsty earth drinks in the day."),
    ("suggest a title for a blog about minimalism", "'Less, Actually' has a nice dry twist on the usual minimalism titles, and 'Room to Breathe' is warmer if that fits your tone."),
    ("write a short morning routine", "Wake up, drink a glass of water, stretch for five minutes, and spend ten minutes on one thing that matters — before you check your phone."),
    ("invent a password that is hard to guess", "Try a phrase-based one: pick four random words and join them, like 'BlueGuitarWinterCloud'. It's long enough to be strong and easy to remember."),
]


def paraphrases(instruction):
    """Return several natural rephrasings of an instruction."""
    plain = instruction
    lowered = instruction.lower()

    variants = {plain}

    # Capitalized as a question
    if not plain.endswith("?"):
        variants.add(plain + "?")

    # Common softening prefixes
    for prefix in ("hey, ", "can you ", "could you ", "please ", "hey um "):
        variants.add(prefix + lowered)

    # Question forms
    variants.add(f"how do i {lowered}" if not lowered.startswith(("how", "what", "why", "when", "who", "where", "write", "invent", "give", "suggest", "tell", "explain", "any", "i ")) else lowered)

    return list(variants)


def generate_dataset(seed=42, variants_per_example=4):
    """Return a list of {"instruction", "response"} dicts (shuffled)."""
    random.seed(seed)

    dataset = []
    for instruction, response in EXAMPLES:
        pool = paraphrases(instruction)
        # Keep at most variants_per_example phrasings per answer.
        chosen = random.sample(pool, min(variants_per_example, len(pool)))
        for variant in chosen:
            dataset.append({"instruction": variant, "response": response})

    random.shuffle(dataset)
    return dataset


if __name__ == "__main__":
    dataset = generate_dataset()

    with open("dataset.jsonl", "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(item, ensure_ascii=False) + "\n" for item in dataset)

    print(f"Saved: dataset.jsonl ({len(dataset)} examples)")
