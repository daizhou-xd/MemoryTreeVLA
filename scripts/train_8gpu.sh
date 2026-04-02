#!/bin/bash
# ============================================================
# 8-GPU DeepSpeed training launcher for MemoryTreeVLA
# Usage: bash scripts/train_8gpu.sh [phase] [config] [extra args]
# Example:
#   bash scripts/train_8gpu.sh 1                              # Phase 1, default config
#   bash scripts/train_8gpu.sh 1 --epochs 20                 # Phase 1, 20 epochs
#   bash scripts/train_8gpu.sh 2 configs/default.yaml        # Phase 2, explicit config
#   bash scripts/train_8gpu.sh 1 configs/default.yaml --wandb
#   bash scripts/train_8gpu.sh 3 configs/default.yaml --deepspeed_config configs/ds_zero3.json
# ============================================================

set -euo pipefail

PHASE=${1:-1}
shift   # consume phase; remaining args are config (optional) + extra flags

if [ "${PHASE}" -lt 1 ] || [ "${PHASE}" -gt 3 ]; then
    echo "Error: phase must be 1, 2, or 3 (3-phase curriculum)."
    exit 1
fi

# If the next argument exists and does NOT start with '-', treat it as the
# config file path; otherwise use the default.
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then
    CONFIG="$1"
    shift
else
    CONFIG="configs/default.yaml"
fi

# All remaining arguments are passed through to train.py verbatim
EXTRA_ARGS="$@"

NUM_GPUS=8
MASTER_PORT=29500

# ── Environment variables ───────────────────────────────────────
# Extend NCCL watchdog timeout to 30 min (default 10 min is too short when
# the Python-level tree loop is large; 1800s gives ample headroom).
export NCCL_TIMEOUT=1800
# Suppress HuggingFace tokenizer fork warning
export TOKENIZERS_PARALLELISM=false
# Deterministic NCCL init across ranks
export NCCL_BLOCKING_WAIT=0

echo "======================================================"
echo "  MemoryTreeVLA  |  Phase ${PHASE}  |  8x A6000"
echo "  Config: ${CONFIG}"
echo "  Extra args: ${EXTRA_ARGS}"
echo "======================================================"

# Select DeepSpeed config based on phase
if [ "${PHASE}" -lt 3 ]; then
    DS_CONFIG="configs/ds_zero2.json"
else
    DS_CONFIG="configs/ds_zero3.json"
fi

# Build log directory
LOG_DIR="logs/phase${PHASE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

deepspeed \
    --num_gpus ${NUM_GPUS} \
    --master_port ${MASTER_PORT} \
    train.py \
    --deepspeed \
    --deepspeed_config ${DS_CONFIG} \
    --config ${CONFIG} \
    --phase ${PHASE} \
    ${EXTRA_ARGS} \
    2>&1 | tee "${LOG_DIR}/train.log"

echo "Logs saved to ${LOG_DIR}/train.log"
