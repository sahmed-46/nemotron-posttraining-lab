# Nemotron Post-Training Lab

Reproduce and compare **Nemotron-style LLM post-training** with three alignment methods:

- **SFT** — Supervised Fine-Tuning
- **DPO** — Direct Preference Optimization
- **GRPO** — Group Relative Policy Optimization (RLVR with verifiable math rewards)

Built for NVIDIA Nemotron / NeMo-RL internship prep: local ablations with TRL, plus NeMo-RL JSONL exports for scale.

## Project structure

```
nemotron-posttraining-lab/
├── configs/                 # YAML experiment configs
│   ├── base.yaml
│   ├── sft.yaml
│   ├── dpo.yaml
│   ├── grpo.yaml
│   └── nemo_rl/             # NeMo-RL cluster templates
├── data/
│   ├── processed/           # generated datasets (gitignored)
│   └── README.md
├── docs/
│   ├── ARCHITECTURE.md
│   └── NEMO_RL_SETUP.md
├── experiments/
│   └── ablations/
├── outputs/                 # checkpoints + eval reports (gitignored)
├── scripts/
│   ├── prepare_datasets.py
│   ├── train_sft.py
│   ├── train_dpo.py
│   ├── train_grpo.py
│   ├── run_eval.py
│   ├── run_ablation.py
│   └── run_nemo_rl.ps1
├── src/
│   ├── config.py
│   ├── data/
│   ├── rewards/
│   ├── training/
│   ├── eval/
│   └── utils/
├── tests/
├── requirements.txt
└── README.md
```

## Quick start

### Google Colab (no local GPU)

1. Upload `notebooks/colab_quickstart.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Runtime → **T4 GPU**
3. Set `REPO_URL` (GitHub) or `DRIVE_PROJECT_PATH` in the notebook
4. Run all cells

See [docs/COLAB.md](docs/COLAB.md) for full instructions.

### Local setup

```powershell
cd "D:\Omar\Side Projects\nemotron-posttraining-lab"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

### 2. Prepare data

```powershell
python scripts/prepare_datasets.py
```

Default dataset: **GSM8K** (512 train / 128 eval samples for fast dev).

### 3. Run full ablation (SFT → DPO → GRPO → eval)

```powershell
python scripts/run_ablation.py
```

### 4. Or run stages individually

```powershell
python scripts/train_sft.py
python scripts/train_dpo.py --model-path outputs/checkpoints/sft/final
python scripts/train_grpo.py --model-path outputs/checkpoints/sft/final
python scripts/run_eval.py --model-path outputs/checkpoints/grpo/final
```

## Configuration

Edit `configs/base.yaml`:

```yaml
model:
  name: Qwen/Qwen2.5-0.5B-Instruct   # use 1.5B/3B if you have GPU memory

data:
  max_train_samples: 512
  max_eval_samples: 128
```

Enable W&B:

```yaml
wandb:
  enabled: true
```

## NeMo-RL path (scale)

See [docs/NEMO_RL_SETUP.md](docs/NEMO_RL_SETUP.md).

`prepare_datasets.py` exports `rl_train.jsonl` / `rl_val.jsonl` for NeMo-RL GRPO jobs.

## Tests

```powershell
pytest tests/ -q
```

## Expected workflow

| Stage | Script | Output |
|---|---|---|
| Data prep | `prepare_datasets.py` | `data/processed/*` |
| SFT warm-start | `train_sft.py` | `outputs/checkpoints/sft/final` |
| DPO | `train_dpo.py` | `outputs/checkpoints/dpo/final` |
| GRPO / RLVR | `train_grpo.py` | `outputs/checkpoints/grpo/final` |
| Eval | `run_eval.py` | `outputs/eval/report.json` |
| Ablation | `run_ablation.py` | `outputs/eval/ablation_summary.json` |

## Hardware notes

| Model | Min GPU |
|---|---|
| Qwen2.5-0.5B + LoRA | 8–12 GB |
| Qwen2.5-1.5B + LoRA | 16–24 GB |
| Nemotron-3-Nano + NeMo-RL | Multi-GPU cluster |

## License

Apache-2.0 (project code). Follow licenses for models and datasets.
