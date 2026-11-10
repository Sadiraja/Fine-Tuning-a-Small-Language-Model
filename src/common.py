"""Shared prompt formatting and config helpers."""
import json
import yaml

SYSTEM_PROMPT = (
    "You are a support assistant. Summarize the conversation in 1-2 concise "
    "sentences, written in the third person."
)


def load_config(path="configs/train_config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def build_prompt(tokenizer, dialogue: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Conversation:\n{dialogue}\n\nSummary:"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
