#!/usr/bin/env bash
# Flat RandOpt baseline on MATH-500 with Qwen2.5-0.5B-Instruct.
# Matches the per-level budget × depth of the fractal run for fair comparison.
set -euo pipefail
cd "$(dirname "$0")/.."

CUDA_DEVICES="${CUDA_DEVICES:-0}"
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
DATASET="math500"
TRAIN_DATA_PATH="data/math-500/test.jsonl"
TEST_DATA_PATH="data/math-500/test.jsonl"

# Budget knobs — keep in sync with the fractal script for fair comparison
DEPTH=4
SAMPLES_PER_LEVEL=64
POPULATION=$((DEPTH * SAMPLES_PER_LEVEL))   # 256

TP=1
NUM_GPUS="$(awk -F',' '{print NF}' <<< "$CUDA_DEVICES")"
NUM_ENGINES=$((NUM_GPUS / TP))
(( NUM_GPUS % TP == 0 )) || { echo "NUM_GPUS must be divisible by TP"; exit 1; }

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export HF_TOKEN="${HF_TOKEN:-}"
export VLLM_NO_USAGE_STATS=1

python3 randopt.py \
  --dataset "$DATASET" \
  --train_data_path "$TRAIN_DATA_PATH" \
  --test_data_path "$TEST_DATA_PATH" \
  --model_name "$MODEL" \
  --num_engines "$NUM_ENGINES" \
  --tp "$TP" \
  --train_samples 200 \
  --precision bfloat16 \
  --population_size "$POPULATION" \
  --top_k_ratios "0.04,0.02,0.005" \
  --sigma_values "0.0005,0.001,0.002" \
  --max_tokens 2048 \
  --global_seed 42 \
  --experiment_dir "experiments/math500_qwen05b_baseline" \
  --cuda_devices "$CUDA_DEVICES" \
  --gpu_memory_utilization 0.92 \
  --max_num_seqs 512 \
  --max_num_batched_tokens 16384 \
  --max_model_len 3072
