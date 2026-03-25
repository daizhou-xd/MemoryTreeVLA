# MemoryTreeVLA
NeurIPS 2026

**MemoryTreeVLA** is a Vision-Language-Action model augmented with a hierarchical memory tree for long-horizon robot manipulation.

---

## Project Structure

```
MemoryTreeVLA/
├── MTVLA/                        # Core model code
│   ├── models/
│   │   ├── memory_tree.py        # Hierarchical memory tree module
│   │   └── vla_model.py          # MemoryTreeVLA model definition
│   ├── configs/
│   │   ├── config.py             # Config dataclasses & YAML loader
│   │   └── default.yaml          # Default training config
│   ├── utils/
│   │   ├── metrics.py            # Evaluation metrics (success rate, etc.)
│   │   └── logger.py             # Logging utility
│   └── train.py                  # Training entry point
│
├── LIBERO_evaluation/            # LIBERO benchmark evaluation
│   ├── libero_evaluator.py       # LIBEROEvaluator class
│   ├── libero_client_4tasks.py   # LIBERO client for 4-task suite
│   ├── eval_libero.py            # Evaluation entry point
│   └── configs/libero_eval.yaml
│
├── CALVIN_evaluation/            # CALVIN benchmark evaluation
│   ├── calvin_evaluator.py       # CALVINEvaluator class (SR1–SR5)
│   ├── eval_calvin.py            # Evaluation entry point
│   └── configs/calvin_eval.yaml
│
└── ROBOMME_evaluation/           # RoboMME benchmark evaluation
    ├── robomme_evaluator.py      # ROBOMMEEvaluator class
    ├── eval_robomme.py           # Evaluation entry point
    └── configs/robomme_eval.yaml
```

---

## Model Preparation

MemoryTreeVLA uses two lightweight Qwen2.5 language models as its LLM backbones. Visual features are extracted entirely by Vision Mamba — **no ViT or visual encoder download is needed**.

### Download Action LLM & Tree LLM

| Role | Model | Notes |
|------|-------|-------|
| `ACTION_LLM` | **Qwen2.5-0.5B** | High-frequency action generation; speed-first |
| `TREE_LLM` | **Qwen2.5-1.5B-Instruct** | Low-frequency subtask state reasoning; Instruct variant ensures stable JSON output |

**Option 1: huggingface-cli (recommended)**
```bash
pip install huggingface_hub

# Action LLM
huggingface-cli download Qwen/Qwen2.5-0.5B \
    --local-dir checkpoints/Qwen2.5-0.5B

# Tree LLM
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
    --local-dir checkpoints/Qwen2.5-1.5B-Instruct
```

**Option 2: HF mirror (if HuggingFace is inaccessible)**
```bash
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download Qwen/Qwen2.5-0.5B \
    --local-dir checkpoints/Qwen2.5-0.5B

huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
    --local-dir checkpoints/Qwen2.5-1.5B-Instruct
```

**Option 3: git lfs**
```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-0.5B checkpoints/Qwen2.5-0.5B
git clone https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct checkpoints/Qwen2.5-1.5B-Instruct
```

The paths above match the defaults already set in `MTVLA/configs/default.yaml`:
```yaml
model:
  action_llm_path: checkpoints/Qwen2.5-0.5B
  tree_llm_path:   checkpoints/Qwen2.5-1.5B-Instruct
```
Modify these values in `default.yaml` if you prefer a different location.

---

## Quick Start

**Training:**
```bash
python MTVLA/train.py --config MTVLA/configs/default.yaml
```

**LIBERO Evaluation:**
```bash
python LIBERO_evaluation/eval_libero.py --model_ckpt outputs/checkpoint.pth --suite all
```

**CALVIN Evaluation:**
```bash
python CALVIN_evaluation/eval_calvin.py --model_ckpt outputs/checkpoint.pth --split D->D
```

**RoboMME Evaluation:**
```bash
python ROBOMME_evaluation/eval_robomme.py --model_ckpt outputs/checkpoint.pth --category all
```

