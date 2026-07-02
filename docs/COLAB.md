# Google Colab Guide (No Local GPU)

## 1. Open Colab with GPU

1. Go to [Google Colab](https://colab.research.google.com/)
2. **File → Upload notebook** → select `notebooks/colab_quickstart.ipynb`
3. **Runtime → Change runtime type → T4 GPU → Save**

## 2. Get the code into Colab

### Option A — GitHub (recommended)

```bash
# On your machine
cd "D:\Omar\Side Projects\nemotron-posttraining-lab"
git init
git add .
git commit -m "Initial post-training lab"
git remote add origin https://github.com/YOUR_USER/nemotron-posttraining-lab.git
git push -u origin main
```

In the notebook, set:

```python
REPO_URL = "https://github.com/YOUR_USER/nemotron-posttraining-lab.git"
```

### Option B — Google Drive (no GitHub)

1. Zip the `nemotron-posttraining-lab` folder
2. Upload zip to Google Drive and unzip (or upload the folder)
3. In the notebook:

```python
DRIVE_PROJECT_PATH = "/content/drive/MyDrive/nemotron-posttraining-lab"
```

## 3. Run all cells

The notebook will:

1. Verify GPU
2. Clone/copy project
3. `pip install -r requirements.txt`
4. Prepare GSM8K subsets (`configs/colab.yaml` — 256 train / 64 eval)
5. **SFT** warm-start
6. **GRPO** post-training (optional toggle)
7. **Eval** → `outputs/eval/colab_report.json`
8. Save `outputs/` to Drive (optional)

## 4. Colab-optimized settings

See `configs/colab.yaml`:

- Model: `Qwen2.5-0.5B-Instruct` + LoRA
- Batch size 1, grad accum 8
- GRPO: 2 generations per prompt (saves VRAM)

Set `RUN_GRPO = False` in the notebook if you run out of memory — SFT-only still produces a valid checkpoint + eval.

## 5. Free tier limits

- Session may disconnect after ~12 hours or idle timeout
- GPU not always available — retry later or use Colab Pro
- Save to Drive after each stage

## Troubleshooting

### `torchao` / `peft` ImportError on SFT

Colab may ship an incompatible `torchao` version. Before training, run:

```python
!pip uninstall -y torchao
```

Or use the project helper:

```python
!python scripts/colab_setup.py
```

Then rerun `train_sft.py`.

## 6. NVIDIA NeMo-RL Colab (scale path)

For the production Nemotron stack, also try NVIDIA’s official NeMo-RL Colab linked in their README:

https://github.com/NVIDIA-NeMo/RL

Your repo exports `data/processed/rl_train.jsonl` for that path.
