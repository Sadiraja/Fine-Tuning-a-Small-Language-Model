# Fine-Tuning a Small Language Model with QLoRA

Fine-tunes **Qwen2.5-1.5B-Instruct** with **QLoRA** (4-bit NF4 base + LoRA adapters) for
**dialogue / support-conversation summarization**, then compares it against the un-tuned
base model on a held-out test set.

| | |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct` (1.5B params, Apache-2.0) |
| Dataset | [DialogSum](https://huggingface.co/datasets/knkarthick/dialogsum) (2,000 train / 200 val / 200 test sampled, seed 42) |
| Method | QLoRA, r=16, alpha=32, all attention + MLP projections |
| Metrics | ROUGE-1 / ROUGE-2 / ROUGE-L, BERTScore-F1, average output length |
| Deliverables | code, training logs, evaluation report, 10 side-by-side examples |

## Project layout

```
configs/train_config.yaml   all hyperparameters
src/prepare_data.py         builds data/{train,val,test}.jsonl
src/train.py                QLoRA training; writes logs/train_log.jsonl + logs/run_info.json
src/evaluate.py             base vs fine-tuned generation + metrics + 10 side-by-side examples
src/report.py               builds reports/EVALUATION_REPORT.md
run_all.sh                  runs the whole pipeline
logs/                       training logs (jsonl, stdout, run_info with hardware/time/cost)
reports/                    metrics.json, side_by_side.md, EVALUATION_REPORT.md
```

## Quick start

Requires an NVIDIA GPU with >= 8 GB VRAM (a free Colab / Kaggle T4 works).

```bash
pip install -r requirements.txt
GPU_HOURLY_USD=0 bash run_all.sh        # set to your GPU's hourly price for cost tracking
```

Or step by step:

```bash
cd slm-finetune-lora
python src/prepare_data.py
python src/train.py --gpu_hourly_usd 0.35
python src/evaluate.py
python src/report.py
```

### Run on Google Colab

```python
!git clone https://github.com/<your-username>/slm-finetune-lora.git
%cd slm-finetune-lora
!pip install -q -r requirements.txt
!GPU_HOURLY_USD=0 bash run_all.sh
```
(Runtime -> Change runtime type -> T4 GPU.)

## Training settings

See `configs/train_config.yaml`. Summary: 2 epochs, effective batch size 16 (4 x 4 accumulation),
lr 2e-4 cosine with 3% warmup, `paged_adamw_8bit`, gradient checkpointing, max length 1024,
loss computed on the summary tokens only (prompt tokens masked).

## Hardware and cost

Training time, GPU name, peak memory and estimated cost are measured automatically and saved to
`logs/run_info.json` and copied into `reports/EVALUATION_REPORT.md`.
Reference estimates for this config (~2,000 examples x 2 epochs): about 20-40 min on a T4,
i.e. **$0** on free Colab/Kaggle, or roughly **$0.15-0.30** on a rented T4/A10 at $0.35-0.50/h.
Replace these with the measured numbers after your run.

## Evaluation methodology

- Both models are loaded with the *same* 4-bit base weights, so the only difference is the LoRA adapter.
- Greedy decoding, `max_new_tokens=96`, identical prompt for both.
- Test set is never seen in training or validation.
- 10 side-by-side examples (random, fixed seed) are in `reports/side_by_side.md`.

## Results

After running, see `reports/EVALUATION_REPORT.md` (results table, settings, hardware, cost,
examples), `reports/metrics.json` and `logs/train_log.jsonl`.

## Adapting to support tickets / entity extraction

Replace `data/*.jsonl` with `{"dialogue": <ticket text>, "summary": <target>}` rows. For entity
extraction, make `summary` a JSON string and swap in an exact-match / field-level F1 metric in
`src/evaluate.py`.

## License

MIT for the code. Check the licenses of the base model and dataset before redistribution.
