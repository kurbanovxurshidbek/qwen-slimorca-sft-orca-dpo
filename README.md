# Qwen2.5-3B: SlimOrca SFT → Orca DPO

Full-parameter **Direct Preference Optimization (DPO)** of a SlimOrca-fine-tuned **Qwen2.5-3B-Instruct** model on **Intel/orca_dpo_pairs**, evaluated with **lm-evaluation-harness** using the same zero-shot protocol as the SFT stage.

This repository accompanies the paper:

> **Preference Optimization Without Capability Change: Direct Preference Optimization of a SlimOrca-Fine-Tuned Qwen2.5-3B Model**
> Khurshidbek Kurbanov, National University of Uzbekistan, 2026.

It continues the SFT study [*Task-Dependent Effects of SlimOrca Supervised Fine-Tuning on Qwen2.5-3B-Instruct*](https://github.com/kurbanovxurshidbek/qwen-slimorca-sft).

---

## TL;DR

- DPO learned the preference objective almost perfectly. On 129 held-out pairs, reward accuracy was **1.00**, the reward margin rose from **0 to 2.73**, and the loss fell from **0.693 to 0.134**.
- Zero-shot performance on six knowledge and reasoning benchmarks **did not change**. The average went from 63.20% to 63.23%, every change was smaller than one standard error, and paired McNemar tests found no significant difference on any benchmark (all p > 0.08).
- DPO met its objective mainly by **lowering the likelihood of rejected responses** (−29.6 nats); the likelihood of chosen responses fell only −2.3 nats.
- **No alignment tax:** all gains from SFT were preserved.

---

## Pipeline

```
Qwen/Qwen2.5-3B-Instruct
        │  SFT · SlimOrca (5,000 conversations, seed 42) · 3 epochs · LR 2e-5
        ▼
kurbanovxurshidbek/qwen2.5-3b-slimorca-sft  ──────────────┐ frozen reference model
        │  DPO · Intel/orca_dpo_pairs (12,857 pairs)        │
        │  β = 0.1 · LR 5e-7 · 3 epochs · effective batch 16 │
        ▼                                                   │
kurbanovxurshidbek/qwen2.5-3b-slimorca-sft-orca-dpo  ◄──────┘
        │
        ▼  lm-evaluation-harness · zero-shot · Qwen chat template
   ARC-E · ARC-C · HellaSwag · MMLU · WinoGrande · GSM8K
```

| Resource | Link |
|---|---|
| Base model | [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) |
| SFT model (DPO start + reference) | [kurbanovxurshidbek/qwen2.5-3b-slimorca-sft](https://huggingface.co/kurbanovxurshidbek/qwen2.5-3b-slimorca-sft) |
| **DPO model (this work)** | [kurbanovxurshidbek/qwen2.5-3b-slimorca-sft-orca-dpo](https://huggingface.co/kurbanovxurshidbek/qwen2.5-3b-slimorca-sft-orca-dpo) |
| SFT data | [Open-Orca/SlimOrca](https://huggingface.co/datasets/Open-Orca/SlimOrca) |
| DPO data | [Intel/orca_dpo_pairs](https://huggingface.co/datasets/Intel/orca_dpo_pairs) |

---

## Results

### Benchmarks (zero-shot, %)

| Benchmark | Metric | Base¹ | SFT | SFT + DPO | Δ DPO − SFT | McNemar p |
|---|---|---:|---:|---:|---:|---:|
| ARC-Easy | acc_norm | 54.59 | 74.37 ± 0.90 | 74.71 ± 0.89 | +0.34 | 0.312 |
| ARC-Challenge | acc_norm | 44.20 | 49.66 ± 1.46 | 50.43 ± 1.46 | +0.77 | 0.136 |
| HellaSwag | acc_norm | 64.34 | 68.91 ± 0.46 | 69.19 ± 0.46 | +0.28 | 0.085 |
| MMLU | acc | 64.48 | 65.10 ± 0.40 | 65.25 ± 0.40 | +0.15 | 0.170 |
| WinoGrande | acc | 63.14 | 65.90 ± 1.33 | 65.43 ± 1.34 | −0.47 | 0.651 |
| GSM8K | exact_match (flexible) | 45.87 | 55.27 ± 1.37 | 54.36 ± 1.37 | −0.91 | 0.437 |
| **Average** | | **56.10** | **63.20** | **63.23** | **+0.03** | |

¹ Base scores come from the SFT paper. SFT and SFT+DPO were evaluated here in the same environment. Re-evaluating SFT reproduced the SFT paper's numbers within ±0.25 points.

### MMLU by category (%)

| Category | Base | SFT | SFT + DPO | Δ |
|---|---:|---:|---:|---:|
| Humanities | 57.64 | 57.81 | 58.21 | +0.40 |
| Other | 70.33 | 70.61 | 70.61 | 0.00 |
| Social Sciences | 75.14 | 75.76 | 75.72 | −0.04 |
| STEM | 58.52 | 60.16 | 60.23 | +0.07 |

### Per-item changes (SFT → SFT + DPO)

| Benchmark | Items | Only SFT correct | Only DPO correct | Changed |
|---|---:|---:|---:|---:|
| ARC-Easy | 2,376 | 20 | 28 | 2.0% |
| ARC-Challenge | 1,172 | 10 | 19 | 2.5% |
| HellaSwag | 10,042 | 109 | 137 | 2.4% |
| MMLU | 14,042 | 86 | 106 | 1.4% |
| WinoGrande | 1,267 | 64 | 58 | 9.6% |
| GSM8K | 1,319 | 106 | 94 | 15.2% |

GSM8K answers did **not** get longer after DPO: mean 108.4 vs 113.5 words, median 115 vs 121.

### DPO training dynamics (129 held-out pairs)

| Epoch (step) | Loss | Reward acc. | Margin | log p (chosen) | log p (rejected) |
|---|---:|---:|---:|---:|---:|
| 0 (0) | 0.693 | — | 0.00 | −101.8 | −207.9 |
| 1 (796) | 0.228 | 1.00 | 1.68 | −102.8 | −225.7 |
| 2 (1,592) | 0.143 | 1.00 | 2.57 | −104.0 | −235.8 |
| 3 (2,387) | 0.134 | 1.00 | 2.73 | −104.1 | −237.5 |

Training-set reward accuracy (mean per epoch): 0.895 → 0.992 → 0.993. No NaN or Inf in any of the 2,387 steps.

### Data overlap (exact match after lower-casing and whitespace normalisation)

| DPO pairs vs | Prompt overlap | Chosen-answer overlap |
|---|---:|---:|
| SFT subset (5,000 SlimOrca conversations) | 79 (0.61%) | 467 (3.63%) |
| Full SlimOrca (517,982) | 6,652 (51.74%) | 5,885 (45.77%) |

---

## Repository structure

```
.
├── README.md
├── prepare_dataset.py              # Intel/orca_dpo_pairs → train.jsonl (DPO messages format)
├── qwen_slimorca_sft_orca_dpo.yaml # Axolotl DPO config (exact config used)
├── paper_stats.py                  # training summary, McNemar tests, answer lengths, overlap
├── training/
│   ├── trainer_state.json          # full step-by-step log (loss, rewards, margins, logps)
│   ├── env_train.txt               # training environment (pip freeze), if present
│   └── train_dpo_tail.log          # last lines of the training log
└── results/
    ├── sft/results_*.json          # lm-eval output, SFT model
    ├── dpo/results_*.json          # lm-eval output, DPO model
    ├── env_eval.txt                # evaluation environment (pip freeze)
    └── paper_stats.txt             # output of paper_stats.py
```

---

## Training configuration

| Parameter | SFT stage | DPO stage |
|---|---|---|
| Initial model | Qwen2.5-3B-Instruct | SFT checkpoint |
| Reference model | — | SFT checkpoint (frozen) |
| Data | SlimOrca, 5,000 conversations | Intel/orca_dpo_pairs, 12,857 pairs |
| Train / held-out | 99% / 1% | 12,728 / 129 pairs |
| Objective | Cross-entropy | Sigmoid DPO, β = 0.1 |
| Method | Full-parameter | Full-parameter |
| Micro-batch × grad. accum. | 16 × 1 (effective 16) | 4 × 4 (effective 16) |
| Epochs / steps | 3 | 3 / 2,387 |
| Optimizer | AdamW (fused) | AdamW (fused) |
| Peak LR / schedule | 2e-5 / cosine | 5e-7 / cosine |
| Warm-up / weight decay | 0.05 / 0.01 | 0.10 / 0.0 |
| Max sequence length | 8,192 | 4,096 |
| Attention | FlashAttention-2 | Eager |
| Precision | BF16 | BF16 (+ TF32 matmul) |
| Hardware | NVIDIA H200 | 1× NVIDIA H200 141 GB (RunPod) |
| Training time | — | about 1.9 h (about 38 min per epoch) |

### Software

| | Version |
|---|---|
| Training | PyTorch 2.14.0+cu130 · Transformers 5.20.1 · TRL 1.14.1 · Axolotl 0.21.1 · Datasets 5.0.1 · Accelerate 1.15.0 |
| Evaluation | PyTorch 2.14.1+cu130 · Transformers 5.22.2 · lm-evaluation-harness 0.4.13 |

---

## Reproduce

Tested on a RunPod pod with **1× H200 (141 GB)**, Ubuntu 24.04 and an NVIDIA driver supporting CUDA 13.x.

### 1. Environments

Training and evaluation use separate virtual environments, so their `transformers` versions don't conflict.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
export UV_TORCH_BACKEND=cu130

# training
cd /root && uv venv --python 3.12 axolotl-env && source axolotl-env/bin/activate
uv pip install torch==2.14.0 torchvision
uv pip install axolotl==0.21.1
deactivate

# evaluation
cd /root && uv venv --python 3.12 eval-env && source eval-env/bin/activate
uv pip install "lm_eval[hf]==0.4.13"
deactivate
```

### 2. Log in

```bash
export HF_HOME=/root/hf_cache
hf auth login      # token with read access (and write access, to push the model)
wandb login        # optional
```

### 3. Prepare data

```bash
source /root/axolotl-env/bin/activate
python prepare_dataset.py          # → train.jsonl (12,857 pairs)
```

Cleaning steps: strip whitespace; drop pairs with an empty question or response; drop pairs where chosen equals rejected; drop exact duplicates. An empty system prompt is omitted, so the Qwen chat template inserts its default system prompt, exactly as in SFT and evaluation.

### 4. Train

```bash
axolotl train qwen_slimorca_sft_orca_dpo.yaml 2>&1 | tee train_dpo.log
```

The YAML writes to `/root/outputs` and `/root/preprocess` on the container disk; change these to suit your machine. `hub_model_id` pushes the final model to the Hub. If the automatic push fails (see Troubleshooting), upload manually:

```bash
hf upload <user>/<repo> /root/outputs . --exclude "checkpoint-*/*" --exclude "debug.log" --exclude "training_args.bin"
```

### 5. Evaluate (same protocol as the SFT paper)

```bash
source /root/eval-env/bin/activate
for M in dpo sft; do
  if [ $M = dpo ]; then P=kurbanovxurshidbek/qwen2.5-3b-slimorca-sft-orca-dpo; else P=kurbanovxurshidbek/qwen2.5-3b-slimorca-sft; fi
  lm_eval --model hf --model_args pretrained=$P,dtype=bfloat16 \
    --tasks arc_easy,arc_challenge,hellaswag,mmlu,winogrande,gsm8k \
    --num_fewshot 0 --apply_chat_template --batch_size auto --device cuda:0 \
    --log_samples --output_path results/$M 2>&1 | tee eval_$M.log
done
```

Reported metrics: `acc_norm` for ARC-E, ARC-C and HellaSwag; `acc` for MMLU and WinoGrande; `exact_match,flexible-extract` for GSM8K. The GSM8K `strict-match` filter scores 0 for chat models because it expects the `#### N` answer format; it isn't used.

### 6. Statistics for the paper

```bash
source /root/axolotl-env/bin/activate
python paper_stats.py | tee paper_stats.txt
```

This prints the per-epoch training summary, the exact McNemar tests (it needs the per-item files that `--log_samples` writes), GSM8K answer lengths and the SlimOrca overlap.

---

## Troubleshooting (things that happened during this run)

| Symptom | Cause | Fix |
|---|---|---|
| `grad_norm: nan` from about step 11; every metric becomes `nan` afterwards | NaN gradients with PyTorch SDPA attention in this DPO setup; a single bad update makes all weights NaN | `attn_implementation: eager`, `sequence_len: 4096`, micro-batch 4 × grad. accum. 4. Check with a 60-step run (`max_steps: 60`) before the full run. |
| `ModuleNotFoundError: flash_attn` | flash-attn isn't installed by Axolotl | install flash-attn, or use `eager` |
| `... is not a valid model identifier` for a private repo | `HF_HOME` changed after `hf auth login`, so the token isn't found | run `hf auth login` again after setting `HF_HOME` |
| `TypeError: create_model_card() got an unexpected keyword argument 'dataset_tags'` at the end of training | Model-card incompatibility between Axolotl and TRL during `push_to_hub` | Training and saving already succeeded. Upload `/root/outputs` with `hf upload`. |
| Many `CUDACachingAllocator ... OOM` warnings during GSM8K evaluation | `--batch_size auto` probing for the largest batch | Normal; wait. Or set `--batch_size 32`. |

---

## Limitations

The results come from a single seed, one preference dataset and one hyperparameter setting (β and LR weren't tuned). The SFT stage used only 5,000 SlimOrca conversations. The evaluation covers six knowledge and reasoning benchmarks, five of them scored by log-likelihood. Instruction following and open-ended generation quality (for example IFEval, MT-Bench and AlpacaEval) weren't measured. See the paper for details.

---

## Citation

```bibtex
@misc{kurbanov2026dpo,
  title  = {Preference Optimization Without Capability Change: Direct Preference
            Optimization of a SlimOrca-Fine-Tuned Qwen2.5-3B Model},
  author = {Kurbanov, Khurshidbek},
  year   = {2026},
  note   = {National University of Uzbekistan},
  url    = {https://github.com/kurbanovxurshidbek/qwen-slimorca-sft-orca-dpo}
}
```

## Contact

Khurshidbek Kurbanov, National University of Uzbekistan, Tashkent: shamsun.com@gmail.com
