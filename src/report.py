"""Assemble reports/EVALUATION_REPORT.md from metrics.json, run_info.json and side_by_side.md."""
import json

m = json.load(open("reports/metrics.json"))
r = json.load(open("logs/run_info.json"))
ex = open("reports/side_by_side.md").read()
b, f = m["base"], m["finetuned"]


def row(name, key, fmt="{:.4f}"):
    d = f[key] - b[key]
    return f"| {name} | {fmt.format(b[key])} | {fmt.format(f[key])} | {d:+.4f} |"


t = r["config"]["train"]
l = r["config"]["lora"]
md = f"""# Evaluation Report

**Task:** Dialogue / support-conversation summarization (DialogSum)
**Base model:** `{r['base_model']}`  |  **Method:** QLoRA (4-bit NF4 + LoRA)
**Held-out test set:** {m['n_test']} examples (never used in training or validation)

## Results

| Metric | Base | Fine-tuned | Delta |
|---|---|---|---|
{row('ROUGE-1', 'rouge1')}
{row('ROUGE-2', 'rouge2')}
{row('ROUGE-L', 'rougeL')}
{row('BERTScore F1', 'bertscore_f1')}
{row('Avg. output length (words)', 'avg_words', '{:.1f}')}

## Training settings

| Setting | Value |
|---|---|
| LoRA rank / alpha / dropout | {l['r']} / {l['alpha']} / {l['dropout']} |
| Target modules | {', '.join(l['target_modules'])} |
| Epochs | {t['epochs']} |
| Batch size x grad accum | {t['per_device_batch_size']} x {t['grad_accum_steps']} |
| Learning rate / scheduler | {t['learning_rate']} / {t['lr_scheduler']} |
| Optimizer | {t['optim']} |
| Max sequence length | {r['config']['max_seq_len']} |
| Train / val examples | {r['train_examples']} / {r['val_examples']} |
| Trainable params | {r['trainable_params']:,} ({r['trainable_pct']}% of {r['total_params']:,}) |
| Final train loss | {r['final_train_loss']:.4f} |

## Hardware and cost

| Item | Value |
|---|---|
| GPU | {r['gpu']} ({r['gpu_memory_gb']} GB) |
| Peak GPU memory | {r['peak_gpu_memory_gb']} GB |
| Precision | {r['precision']} |
| Training time | {r['train_hours']} h |
| GPU price | ${r['gpu_hourly_usd']}/h |
| **Estimated cost** | **${r['estimated_cost_usd']}** |

Training logs: `logs/train_log.jsonl`, `logs/train_stdout.log`.

---

{ex}
"""
open("reports/EVALUATION_REPORT.md", "w").write(md)
print("Wrote reports/EVALUATION_REPORT.md")
