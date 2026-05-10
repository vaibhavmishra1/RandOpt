#!/usr/bin/env bash
# Fetch the MATH-500 test set into data/math-500/test.jsonl
# Source: HuggingFaceH4/MATH-500
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p data/math-500
OUT="data/math-500/test.jsonl"

if [[ -s "$OUT" ]]; then
  echo "Already exists: $OUT ($(wc -l <"$OUT") lines). Skipping."
  exit 0
fi

URL="https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl"
echo "Downloading $URL"
curl -L --fail -o "$OUT" "$URL"

echo "Wrote $OUT ($(wc -l <"$OUT") lines)"
