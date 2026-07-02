# NeMo-RL Setup (Production Path)

This project’s **local path** uses Hugging Face TRL for fast iteration.
The **production path** aligns with NVIDIA Nemotron post-training using **NeMo-RL**.

## Prerequisites

- Linux (recommended)
- NVIDIA GPU cluster (multi-GPU for Nemotron-Nano scale)
- CUDA 12.x
- Python 3.10+

## Clone NeMo-RL

```bash
git clone https://github.com/NVIDIA-NeMo/RL.git
cd RL
git submodule update --init --recursive
uv venv && source .venv/bin/activate
uv sync
```

Or use the NGC container documented in the NeMo-RL README.

## Prepare data from this repo

```bash
cd /path/to/nemotron-posttraining-lab
python scripts/prepare_datasets.py
```

This creates:

- `data/processed/rl_train.jsonl`
- `data/processed/rl_val.jsonl`

## Run GRPO (example)

See NVIDIA guides:

- [Nemotron-3-Nano NeMo-RL guide](https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/nemotron-3-nano.md)
- [NeMo-RL documentation](https://docs.nvidia.com/nemo/rl/latest/)

Use the template config in `configs/nemo_rl/grpo_tiny.yaml` and point JSONL paths to this repo’s processed data.

## OSS contribution targets

- Fix/reproduce tiny-config examples
- AddE2E repro steps on 1×A100
- Add metric parsing scripts for KL / reward variance / response length
- Unit tests for reward utilities shared with NeMo-Gym patterns

## Citation

If you use NeMo-RL in research, cite the repository listed in NVIDIA’s docs.
