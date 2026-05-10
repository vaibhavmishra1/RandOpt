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

# Geometric sigma annealing: 0.002 -> 0.001 -> 0.0005 -> 0.00025
SIGMA_START=0.002
SIGMA_DECAY=0.5

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
  --cuda_devices "$CUDA_DEVICES"
