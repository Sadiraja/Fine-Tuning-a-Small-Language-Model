"""Evaluate base vs. fine-tuned model on the held-out test set.

Metrics: ROUGE-1/2/L, BERTScore-F1, mean output length (words).
Outputs: outputs/predictions.jsonl, reports/metrics.json, reports/side_by_side.md
"""
import argparse
import json
import random
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import evaluate as hf_evaluate
from common import load_config, build_prompt, read_jsonl, write_jsonl


@torch.no_grad()
def generate_all(model, tok, rows, bs, max_new):
    outs = []
    for i in tqdm(range(0, len(rows), bs)):
        prompts = [build_prompt(tok, r["dialogue"]) for r in rows[i:i + bs]]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        outs += [s.strip() for s in tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
    return outs


def score(preds, refs):
    rouge = hf_evaluate.load("rouge").compute(predictions=preds, references=refs)
    bs = hf_evaluate.load("bertscore").compute(predictions=preds, references=refs, lang="en")
    return {
        "rouge1": rouge["rouge1"], "rouge2": rouge["rouge2"], "rougeL": rouge["rougeL"],
        "bertscore_f1": float(np.mean(bs["f1"])),
        "avg_words": float(np.mean([len(p.split()) for p in preds])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="outputs/lora_adapter")
    ap.add_argument("--config", default="configs/train_config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ev = cfg["eval"]

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    # Same 4-bit base for both runs so the comparison isolates the LoRA adapter.
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], quantization_config=bnb,
                                                 device_map="auto", torch_dtype=dtype)
    model.eval()
    test = read_jsonl("data/test.jsonl")
    refs = [r["summary"] for r in test]

    print("Generating with BASE model...")
    base_out = generate_all(model, tok, test, ev["batch_size"], ev["max_new_tokens"])

    print("Generating with FINE-TUNED model...")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    ft_out = generate_all(model, tok, test, ev["batch_size"], ev["max_new_tokens"])

    write_jsonl("outputs/predictions.jsonl", [
        {"dialogue": r["dialogue"], "reference": r["summary"], "base": b, "finetuned": f}
        for r, b, f in zip(test, base_out, ft_out)])

    metrics = {"base": score(base_out, refs), "finetuned": score(ft_out, refs), "n_test": len(test)}
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    idx = random.Random(0).sample(range(len(test)), ev["num_examples"])
    clean = lambda s: s.replace("\n", " ").replace("|", "/")
    lines = ["# Side-by-Side Examples\n"]
    for n, i in enumerate(idx, 1):
        lines += [f"## Example {n} (test index {i})\n",
                  f"**Dialogue:**\n\n```\n{test[i]['dialogue']}\n```\n",
                  "| Source | Output |", "|---|---|",
                  f"| Reference | {clean(refs[i])} |",
                  f"| Base model | {clean(base_out[i])} |",
                  f"| Fine-tuned | {clean(ft_out[i])} |\n"]
    with open("reports/side_by_side.md", "w") as f:
        f.write("\n".join(lines))
    print("Wrote reports/side_by_side.md")


if __name__ == "__main__":
    main()
