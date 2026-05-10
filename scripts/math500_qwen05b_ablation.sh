#!/usr/bin/env bash
# Module ablation on a completed baseline run: localize where in the network
# winning perturbations carry their gain.
set -euo pipefail
cd "$(dirname "$0")/.."

CUDA_DEVICES="${CUDA_DEVICES:-0}"
BASELINE_DIR="${BASELINE_DIR:?Set BASELINE_DIR to a completed RandOpt run dir, e.g. experiments/math500_qwen05b_baseline/math500_<TS>}"
TOP_K="${TOP_K:-3}"
# Optional override; leave empty to inherit from baseline_dir/top_k_seeds.json
MODEL_NAME="${MODEL_NAME:-}"

TP=1
NUM_GPUS="$(awk -F',' '{print NF}' <<< "$CUDA_DEVICES")"
NUM_ENGINES=$((NUM_GPUS / TP))
(( NUM_GPUS % TP == 0 )) || { echo "NUM_GPUS must be divisible by TP"; exit 1; }

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export VLLM_NO_USAGE_STATS=1

CMD=(python3 module_ablation.py
  --baseline_dir "$BASELINE_DIR"
  --top_k "$TOP_K"
  --train_samples 200
  --max_tokens 2048
  --num_engines "$NUM_ENGINES"
  --tp "$TP"
  --cuda_devices "$CUDA_DEVICES"
  --gpu_memory_utilization 0.92
  --max_num_seqs 512
  --max_num_batched_tokens 16384
  --max_model_len 3072)

if [[ -n "$MODEL_NAME" ]]; then
  CMD+=(--model_name "$MODEL_NAME")
fi

"${CMD[@]}"
