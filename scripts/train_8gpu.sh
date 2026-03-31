#!/bin/bash
# ============================================================
# 8-GPU DeepSpeed training launcher for MemoryTreeVLA
# Usage: bash scripts/train_8gpu.sh [phase] [config] [extra args]
# Example:
#   bash scripts/train_8gpu.sh 1                            # Phase 1, default config
#   bash scripts/train_8gpu.sh 3 configs/default.yaml      # Phase 3
#   bash scripts/train_8gpu.sh 1 configs/default.yaml --wandb --wandb_project MyProject
#   bash scripts/train_8gpu.sh 4 configs/default.yaml --deepspeed_config configs/ds_zero3.json
# ============================================================

set -euo pipefail

PHASE=${1:-1}
CONFIG=${2:-configs/default.yaml}
EXTRA_ARGS=${@:3}

NUM_GPUS=8
MASTER_PORT=29500

echo "======================================================"
echo "  MemoryTreeVLA  |  Phase ${PHASE}  |  8x A6000"
echo "  Config: ${CONFIG}"
echo "  Extra args: ${EXTRA_ARGS}"
echo "======================================================"

# Select DeepSpeed config based on phase
if [ "${PHASE}" -lt 4 ]; then
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
