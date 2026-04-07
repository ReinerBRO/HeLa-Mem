#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/eval_longmemeval.sh <data_path> <mem_dir> [workers]"
  exit 1
fi

DATA_PATH="$1"
MEM_DIR="$2"
WORKERS="${3:-8}"

export HEBBIAN_MODEL="${HEBBIAN_MODEL:-gpt-4o-mini}"
export HEBBIAN_TAU="${HEBBIAN_TAU:-1e7}"
export HEBBIAN_LEARNING_RATE="${HEBBIAN_LEARNING_RATE:-0.02}"
export HEBBIAN_DECAY_RATE="${HEBBIAN_DECAY_RATE:-0.995}"
export HEBBIAN_ACTIVATION_ALPHA="${HEBBIAN_ACTIVATION_ALPHA:-0.1}"
export HEBBIAN_SPREADING_THRESHOLD="${HEBBIAN_SPREADING_THRESHOLD:-0.4}"
export HEBBIAN_MAX_FLIPPED="${HEBBIAN_MAX_FLIPPED:-1}"
export HEBBIAN_KEYWORD_WEIGHT="${HEBBIAN_KEYWORD_WEIGHT:-0.7}"

python -m hela_mem.eval_longmemeval \
  --data_path "$DATA_PATH" \
  --mem_dir "$MEM_DIR" \
  --workers "$WORKERS" \
  --top_k "${HEBBIAN_TOP_K:-20}" \
  --semantic_top_k "${HEBBIAN_SEMANTIC_TOP_K:-5}"
