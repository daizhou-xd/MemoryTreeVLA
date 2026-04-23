#!/bin/bash
# Pre-training: RoboCerebra → SemanticJumpHead + SGMTS + s_proj
# Requires: accelerate, 8× GPU (A100/80G recommended or 4× for small batch)
# ────────────────────────────────────────────────────────────────────────
# Usage:
#   bash scripts/pretrain.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/pretrain.sh   # 4 GPU
# ────────────────────────────────────────────────────────────────────────
set -e

# 若 Evo-1 未与 DualTreeVLA 处于常见同级目录，请手动设置其路径：
# export PYTHONPATH="${PYTHONPATH}:/path/to/Evo-1/Evo-1"

NUM_GPUS=${1:-8}
CONFIG=dual_tree_vla/config/pretrain.yaml

echo "[pretrain.sh] GPUs=$NUM_GPUS  config=$CONFIG"

accelerate launch \
    --num_processes $NUM_GPUS \
    --mixed_precision bf16 \
    --dynamo_backend no \
    pretrain.py \
        --config "$CONFIG"
