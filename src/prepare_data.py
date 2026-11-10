"""Download DialogSum (customer-service style dialogues) and create train/val/test jsonl.

To use your own support tickets instead, create data/{train,val,test}.jsonl with
fields {"dialogue": ..., "summary": ...} and skip this script.
"""
import random
from datasets import load_dataset
from common import load_config, write_jsonl


def main():
    cfg = load_config()["data"]
    ds = load_dataset("knkarthick/dialogsum")
    rng = random.Random(cfg["seed"])

    def take(split, n):
        rows = [{"dialogue": r["dialogue"], "summary": r["summary"]} for r in ds[split]]
        rng.shuffle(rows)
        return rows[:n]

    write_jsonl("data/train.jsonl", take("train", cfg["train_size"]))
    write_jsonl("data/val.jsonl", take("validation", cfg["val_size"]))
    write_jsonl("data/test.jsonl", take("test", cfg["test_size"]))
    print("Wrote data/train.jsonl, data/val.jsonl, data/test.jsonl")


if __name__ == "__main__":
    main()
