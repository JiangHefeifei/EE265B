#!/usr/bin/env bash
set -euo pipefail

# Counting-only evaluation for a single-GPU workstation.
# Usage:
#   MODEL_TYPE=perceptual-framesamp-modul CKPT_ID=79999 SEED=7 bash scripts/eval_counting_single_gpu.sh

MODEL_TYPE="${MODEL_TYPE:-perceptual-framesamp-modul}"
SEED="${SEED:-7}"
CKPT_ID="${CKPT_ID:-79999}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-8000}"
COUNTING_TASKS="${COUNTING_TASKS:-PickXtimes}"
MAX_STEPS="${MAX_STEPS:-1300}"
MAX_EPISODES_PER_TASK="${MAX_EPISODES_PER_TASK:-20}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${PWD}/.uv-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${PWD}/.mpl-cache}"
export SAPIEN_RENDER_DEVICE="${SAPIEN_RENDER_DEVICE:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.75}"

CONFIG_TYPE="mme_vla_suite"
POLICY_DIR_MODEL="${MODEL_TYPE}"

case "${MODEL_TYPE}" in
  pi05_baseline)
    CONFIG_TYPE="pi05_baseline"
    EXTRA_ARGS="${EXTRA_ARGS} --args.no-use-history"
    ;;
  symbolic_simpleSG_oracle)
    POLICY_DIR_MODEL="symbolic-simple-subgoal"
    EXTRA_ARGS="${EXTRA_ARGS} --args.use-oracle --args.subgoal-type=simple_subgoal"
    ;;
  symbolic_groundedSG_oracle)
    POLICY_DIR_MODEL="symbolic-grounded-subgoal"
    EXTRA_ARGS="${EXTRA_ARGS} --args.use-oracle --args.subgoal-type=grounded_subgoal"
    ;;
  symbolic_simpleSG_qwenvl)
    POLICY_DIR_MODEL="symbolic-simple-subgoal"
    EXTRA_ARGS="${EXTRA_ARGS} --args.use-qwenvl --args.subgoal-type=simple_subgoal"
    ;;
  symbolic_groundedSG_qwenvl)
    POLICY_DIR_MODEL="symbolic-grounded-subgoal"
    EXTRA_ARGS="${EXTRA_ARGS} --args.use-qwenvl --args.subgoal-type=grounded_subgoal"
    ;;
  symbolic_groundedSG_progress_qwenvl)
    POLICY_DIR_MODEL="symbolic-grounded-subgoal"
    EXTRA_ARGS="${EXTRA_ARGS} --args.use-qwenvl --args.use-pickxtimes-progress --args.subgoal-type=grounded_subgoal"
    ;;
  dual_grounded_framesamp_modul_oracle)
    POLICY_DIR_MODEL="dual-grounded-framesamp-modul"
    EXTRA_ARGS="${EXTRA_ARGS} --args.use-oracle --args.subgoal-type=grounded_subgoal"
    ;;
  dual_grounded_framesamp_modul_qwenvl)
    POLICY_DIR_MODEL="dual-grounded-framesamp-modul"
    EXTRA_ARGS="${EXTRA_ARGS} --args.use-qwenvl --args.subgoal-type=grounded_subgoal"
    ;;
  dual_lite_500_oracle)
    CONFIG_TYPE="mme_vla_suite_dual_lite"
    POLICY_DIR_MODEL="dual-lite-500-real"
    EXTRA_ARGS="${EXTRA_ARGS} --args.use-oracle --args.subgoal-type=grounded_subgoal"
    ;;
  dual_lite_500_qwenvl)
    CONFIG_TYPE="mme_vla_suite_dual_lite"
    POLICY_DIR_MODEL="dual-lite-500-real"
    EXTRA_ARGS="${EXTRA_ARGS} --args.use-qwenvl --args.subgoal-type=grounded_subgoal"
    ;;
  dual_merged_gated_500_oracle)
    CONFIG_TYPE="mme_vla_suite_dual_lite"
    POLICY_DIR_MODEL="dual-merged-gated-500"
    EXTRA_ARGS="${EXTRA_ARGS} --args.use-oracle --args.subgoal-type=grounded_subgoal"
    ;;
  dual_merged_gated_500_qwenvl)
    CONFIG_TYPE="mme_vla_suite_dual_lite"
    POLICY_DIR_MODEL="dual-merged-gated-500"
    EXTRA_ARGS="${EXTRA_ARGS} --args.use-qwenvl --args.subgoal-type=grounded_subgoal"
    ;;
  dual_merged_gated_500_progress_qwenvl)
    CONFIG_TYPE="mme_vla_suite_dual_lite"
    POLICY_DIR_MODEL="dual-merged-gated-500"
    EXTRA_ARGS="${EXTRA_ARGS} --args.use-qwenvl --args.use-pickxtimes-progress --args.subgoal-type=grounded_subgoal"
    ;;
  dual_correction_500_oracle)
    CONFIG_TYPE="mme_vla_suite_dual_correction"
    POLICY_DIR_MODEL="dual-correction-gated-500"
    EXTRA_ARGS="${EXTRA_ARGS} --args.use-oracle --args.subgoal-type=grounded_subgoal"
    ;;
  dual_correction_500_qwenvl)
    CONFIG_TYPE="mme_vla_suite_dual_correction"
    POLICY_DIR_MODEL="dual-correction-gated-500"
    EXTRA_ARGS="${EXTRA_ARGS} --args.use-qwenvl --args.subgoal-type=grounded_subgoal"
    ;;
  dual_correction_stale_500_oracle)
    CONFIG_TYPE="mme_vla_suite_dual_correction"
    POLICY_DIR_MODEL="dual-correction-stale-500"
    EXTRA_ARGS="${EXTRA_ARGS} --args.use-oracle --args.subgoal-type=grounded_subgoal"
    ;;
  dual_correction_stale_500_qwenvl)
    CONFIG_TYPE="mme_vla_suite_dual_correction"
    POLICY_DIR_MODEL="dual-correction-stale-500"
    EXTRA_ARGS="${EXTRA_ARGS} --args.use-qwenvl --args.subgoal-type=grounded_subgoal"
    ;;
  dual_correction_stale_500_fallback_qwenvl)
    CONFIG_TYPE="mme_vla_suite_dual_correction"
    POLICY_DIR_MODEL="dual-correction-stale-500"
    EXTRA_ARGS="${EXTRA_ARGS} --args.use-qwenvl --args.use-pickxtimes-stuck-fallback --args.subgoal-type=grounded_subgoal"
    ;;
  symbolic_simpleSG_gemini)
    POLICY_DIR_MODEL="symbolic-simple-subgoal"
    EXTRA_ARGS="${EXTRA_ARGS} --args.use-gemini --args.subgoal-type=simple_subgoal"
    ;;
  symbolic_groundedSG_gemini)
    POLICY_DIR_MODEL="symbolic-grounded-subgoal"
    EXTRA_ARGS="${EXTRA_ARGS} --args.use-gemini --args.subgoal-type=grounded_subgoal"
    ;;
  MemER)
    POLICY_DIR_MODEL="symbolic-grounded-subgoal"
    EXTRA_ARGS="${EXTRA_ARGS} --args.use-memer --args.subgoal-type=grounded_subgoal"
    ;;
esac

CHECKPOINT_DIR="runs/ckpts/${CONFIG_TYPE}/${POLICY_DIR_MODEL}/${CKPT_ID}"
if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "Missing checkpoint directory: ${CHECKPOINT_DIR}" >&2
  echo "Download and unzip the official checkpoint before running evaluation." >&2
  exit 1
fi

if [[ -n "${MAX_EPISODES_PER_TASK}" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --args.max-episodes-per-task=${MAX_EPISODES_PER_TASK}"
fi

activate_robomme_env() {
  if command -v micromamba >/dev/null 2>&1; then
    eval "$(micromamba shell hook --shell bash)"
    micromamba activate robomme
  elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate robomme
  else
    echo "Neither micromamba nor conda was found for the robomme simulator environment." >&2
    exit 1
  fi
}

echo "Checking GPU visibility in the simulator environment"
if ! nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi failed in this shell. Run from a shell where nvidia-smi can see your GPU." >&2
  exit 1
fi
activate_robomme_env
CUDA_VISIBLE_DEVICES="${GPU_ID}" python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is False in the robomme environment")
print("robomme torch cuda ok:", torch.cuda.get_device_name(0))
PY
conda deactivate >/dev/null 2>&1 || true

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "Starting ${MODEL_TYPE} policy server on GPU ${GPU_ID}, port ${PORT}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run scripts/serve_policy.py \
  --seed="${SEED}" \
  --port="${PORT}" \
  policy:checkpoint \
  --policy.dir="${CHECKPOINT_DIR}" \
  --policy.config="${CONFIG_TYPE}" &
SERVER_PID=$!

sleep 30

if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
  echo "Policy server exited before evaluation started. Check the traceback above." >&2
  exit 1
fi

echo "Evaluating Counting tasks: ${COUNTING_TASKS}"
activate_robomme_env
CUDA_VISIBLE_DEVICES="${GPU_ID}" python examples/robomme/eval.py \
  --args.model_seed="${SEED}" \
  --args.port="${PORT}" \
  --args.policy_name="${POLICY_DIR_MODEL}" \
  --args.model_ckpt_id="${CKPT_ID}" \
  --args.only-tasks="${COUNTING_TASKS}" \
  --args.max-steps="${MAX_STEPS}" \
  ${EXTRA_ARGS}
