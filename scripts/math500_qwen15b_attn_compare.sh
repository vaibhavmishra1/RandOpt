#!/usr/bin/env bash
# Compare two RandOpt baselines on MATH-500 / Qwen2.5-1.5B-Instruct:
#   (A) full perturbation     (no filter)
#   (B) attention frozen      (--perturb_exclude self_attn)
# Identical seeds, sigmas, population — only the filter differs.
set -euo pipefail
cd "$(dirname "$0")/.."

CUDA_DEVICES="${CUDA_DEVICES:-0}"
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
DATASET="math500"
TRAIN_DATA_PATH="data/math-500/test.jsonl"
TEST_DATA_PATH="data/math-500/test.jsonl"

# Same compute envelope for both runs
POPULATION="${POPULATION:-256}"
SIGMA_VALUES="${SIGMA_VALUES:-0.0005,0.001,0.002}"
TOP_K_RATIOS="${TOP_K_RATIOS:-0.16,0.08,0.02,0.005}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-200}"
GLOBAL_SEED="${GLOBAL_SEED:-42}"

TP=1
NUM_GPUS="$(awk -F',' '{print NF}' <<< "$CUDA_DEVICES")"
NUM_ENGINES=$((NUM_GPUS / TP))
(( NUM_GPUS % TP == 0 )) || { echo "NUM_GPUS must be divisible by TP"; exit 1; }

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export HF_TOKEN="${HF_TOKEN:-}"
export VLLM_NO_USAGE_STATS=1

run_one () {
  local label="$1"; shift
  local extra_args=("$@")
  local exp_dir="experiments/math500_qwen15b_${label}"
  echo
  echo "############################################################"
  echo "# RUN: ${label}"
  echo "# extra args: ${extra_args[*]:-(none)}"
  echo "############################################################"
  python3 randopt.py \
    --dataset "$DATASET" \
    --train_data_path "$TRAIN_DATA_PATH" \
    --test_data_path "$TEST_DATA_PATH" \
    --model_name "$MODEL" \
    --num_engines "$NUM_ENGINES" \
    --tp "$TP" \
    --train_samples "$TRAIN_SAMPLES" \
    --precision bfloat16 \
    --population_size "$POPULATION" \
    --top_k_ratios "$TOP_K_RATIOS" \
    --sigma_values "$SIGMA_VALUES" \
    --max_tokens 2048 \
    --global_seed "$GLOBAL_SEED" \
    --experiment_dir "$exp_dir" \
    --cuda_devices "$CUDA_DEVICES" \
    --gpu_memory_utilization 0.92 \
    --max_num_seqs 512 \
    --max_num_batched_tokens 16384 \
    --max_model_len 3072 \
    "${extra_args[@]}"
}

# (A) standard RandOpt — full perturbation
run_one "full"

# (B) attention frozen
run_one "attn_frozen" --perturb_exclude "self_attn"

echo
echo "Done. Compare:"
echo "  experiments/math500_qwen15b_full/        (full perturbation)"
echo "  experiments/math500_qwen15b_attn_frozen/ (attention frozen)"
