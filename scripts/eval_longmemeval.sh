#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/eval_longmemeval.sh <data_path> <mem_dir> [workers]"
  exit 1
fi

DATA_PATH="$1"
MEM_DIR="$2"
WORKERS="${3:-8}"

export HELA_MEM_MODEL="${HELA_MEM_MODEL:-gpt-4o-mini}"
export HELA_MEM_TAU="${HELA_MEM_TAU:-1e7}"
export HELA_MEM_LEARNING_RATE="${HELA_MEM_LEARNING_RATE:-0.02}"
export HELA_MEM_DECAY_RATE="${HELA_MEM_DECAY_RATE:-0.995}"
export HELA_MEM_ACTIVATION_ALPHA="${HELA_MEM_ACTIVATION_ALPHA:-0.1}"
export HELA_MEM_SPREADING_THRESHOLD="${HELA_MEM_SPREADING_THRESHOLD:-0.4}"
export HELA_MEM_MAX_FLIPPED="${HELA_MEM_MAX_FLIPPED:-1}"
export HELA_MEM_KEYWORD_WEIGHT="${HELA_MEM_KEYWORD_WEIGHT:-0.7}"

python -m hela_mem.eval_longmemeval \
  --data_path "$DATA_PATH" \
  --mem_dir "$MEM_DIR" \
  --workers "$WORKERS" \
  --top_k "${HELA_MEM_TOP_K:-20}" \
  --semantic_top_k "${HELA_MEM_SEMANTIC_TOP_K:-5}"
