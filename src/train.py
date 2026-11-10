"""QLoRA fine-tuning (4-bit NF4 base + LoRA adapters) with a plain HF Trainer."""
import argparse
import json
import time
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    Trainer, TrainingArguments, TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from common import load_config, build_prompt, read_jsonl


class JsonlLogger(TrainerCallback):
    """Writes every Trainer log event to logs/train_log.jsonl."""
    def __init__(self, path):
        self.path = path
        open(path, "w").close()

    def on_log(self, args, state, control, logs=None, **kw):
        if logs:
            with open(self.path, "a") as f:
                f.write(json.dumps({"step": state.global_step, **logs}) + "\n")


def tokenize_example(tok, row, max_len):
    prompt = build_prompt(tok, row["dialogue"])
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(row["summary"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (p_ids + a_ids)[:max_len]
    labels = ([-100] * len(p_ids) + a_ids)[:max_len]  # loss on summary only
    return {"input_ids": ids, "labels": labels}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        m = max(len(b["input_ids"]) for b in batch)
        ids, labs, att = [], [], []
        for b in batch:
            pad = m - len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad_id] * pad)
            labs.append(b["labels"] + [-100] * pad)
            att.append([1] * len(b["input_ids"]) + [0] * pad)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labs),
                "attention_mask": torch.tensor(att)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_config.yaml")
    ap.add_argument("--out", default="outputs/lora_adapter")
    ap.add_argument("--gpu_hourly_usd", type=float, default=0.0,
                    help="Price per GPU hour (0 for free Colab/Kaggle).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    t, l = cfg["train"], cfg["lora"]
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], quantization_config=bnb, device_map="auto", torch_dtype=dtype)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=l["r"], lora_alpha=l["alpha"], lora_dropout=l["dropout"],
        target_modules=l["target_modules"], bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    train = [tokenize_example(tok, r, cfg["max_seq_len"]) for r in read_jsonl("data/train.jsonl")]
    val = [tokenize_example(tok, r, cfg["max_seq_len"]) for r in read_jsonl("data/val.jsonl")]

    targs = TrainingArguments(
        output_dir="outputs/checkpoints",
        num_train_epochs=t["epochs"],
        per_device_train_batch_size=t["per_device_batch_size"],
        per_device_eval_batch_size=t["per_device_batch_size"],
        gradient_accumulation_steps=t["grad_accum_steps"],
        learning_rate=t["learning_rate"], lr_scheduler_type=t["lr_scheduler"],
        warmup_ratio=t["warmup_ratio"], weight_decay=t["weight_decay"],
        logging_steps=t["logging_steps"], eval_strategy="steps", eval_steps=t["eval_steps"],
        save_steps=t["save_steps"], save_total_limit=2, optim=t["optim"],
        bf16=use_bf16, fp16=not use_bf16, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=t["seed"], report_to="none", remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train, eval_dataset=val,
                      data_collator=Collator(tok.pad_token_id),
                      callbacks=[JsonlLogger("logs/train_log.jsonl")])

    start = time.time()
    result = trainer.train()
    hours = (time.time() - start) / 3600
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)

    cuda = torch.cuda.is_available()
    info = {
        "base_model": cfg["base_model"],
        "gpu": torch.cuda.get_device_name(0) if cuda else "CPU",
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if cuda else None,
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if cuda else None,
        "precision": "bf16" if use_bf16 else "fp16", "quantization": "4-bit NF4 + double quant",
        "trainable_params": trainable, "total_params": total,
        "trainable_pct": round(100 * trainable / total, 3),
        "train_examples": len(train), "val_examples": len(val),
        "train_hours": round(hours, 3), "gpu_hourly_usd": args.gpu_hourly_usd,
        "estimated_cost_usd": round(hours * args.gpu_hourly_usd, 3),
        "final_train_loss": result.training_loss, "config": cfg,
    }
    with open("logs/run_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
