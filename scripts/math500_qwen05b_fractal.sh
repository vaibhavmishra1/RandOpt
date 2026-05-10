#!/usr/bin/env bash
# Fractal RandOpt — iterated greedy + sigma annealing — on MATH-500 with
# Qwen2.5-0.5B-Instruct. Matches total budget of the baseline script.
set -euo pipefail
cd "$(dirname "$0")/.."

CUDA_DEVICES="${CUDA_DEVICES:-0}"
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
DATASET="math500"
TRAIN_DATA_PATH="data/math-500/test.jsonl"
TEST_DATA_PATH="data/math-500/test.jsonl"

# Fractal budget — total samples = DEPTH * SAMPLES_PER_LEVEL
DEPTH=4
SAMPLES_PER_LEVEL=64

# Geometric sigma annealing, anchored at the empirically-best sigma from
# the flat baseline (σ=0.0005 was the only productive scale for Qwen-0.5B).
# Schedule: 0.0005 -> 0.00035 -> 0.000245 -> 0.0001715
SIGMA_START=0.0005
SIGMA_DECAY=0.7

TP=1
NUM_GPUS="$(awk -F',' '{print NF}' <<< "$CUDA_DEVICES")"
NUM_ENGINES=$((NUM_GPUS / TP))
(( NUM_GPUS % TP == 0 )) || { echo "NUM_GPUS must be divisible by TP"; exit 1; }

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export HF_TOKEN="${HF_TOKEN:-}"
export VLLM_NO_USAGE_STATS=1

python3 fractal_randopt.py \
  --dataset "$DATASET" \
  --train_data_path "$TRAIN_DATA_PATH" \
  --test_data_path "$TEST_DATA_PATH" \
  --model_name "$MODEL" \
  --num_engines "$NUM_ENGINES" \
  --tp "$TP" \
  --train_samples 200 \
  --precision bfloat16 \
  --depth "$DEPTH" \
  --samples_per_level "$SAMPLES_PER_LEVEL" \
  --sigma_start "$SIGMA_START" \
  --sigma_decay "$SIGMA_DECAY" \
  --ensemble_top_k "1,5,10" \
  --density_thresholds "0.0,0.01,0.02,0.05,0.1" \
  --max_tokens 2048 \
  --global_seed 42 \
  --experiment_dir "experiments/math500_qwen05b_fractal" \
  --cuda_devices "$CUDA_DEVICES" \
  --gpu_memory_utilization 0.92 \
  --max_num_seqs 512 \
  --max_num_batched_tokens 16384 \
  --max_model_len 3072
