#!/usr/bin/env bash
set -euo pipefail

# Lightweight dual-memory finetuning entrypoint for a single-GPU workstation.
# Override DATASET_PATH, EXP_NAME, NUM_TRAIN_STEPS, BATCH_SIZE, or GPU_ID as needed.

GPU_ID="${GPU_ID:-0}"
EXP_NAME="${EXP_NAME:-dual-grounded-framesamp-modul-lite}"
DATASET_PATH="${DATASET_PATH:-data/robomme_preprocessed_data}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${PWD}/.uv-cache}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run scripts/train.py mme_vla_suite_dual_lite \
  --exp-name="${EXP_NAME}" \
  --dataset-path="${DATASET_PATH}" \
  --num-train-steps="${NUM_TRAIN_STEPS}" \
  --batch-size="${BATCH_SIZE}" \
  --num-workers="${NUM_WORKERS}" \
  --fsdp-devices=1 \
  --overwrite
