#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="${1:-data/longmemeval_s.json}"
MEM_DIR="${2:-results/longmemeval_mem_full}"
WORKERS="${3:-8}"

export HEBBIAN_MODEL="${HEBBIAN_MODEL:-gpt-4o-mini}"
export HEBBIAN_TAU="${HEBBIAN_TAU:-1e7}"
export HEBBIAN_LEARNING_RATE="${HEBBIAN_LEARNING_RATE:-0.02}"
export HEBBIAN_DECAY_RATE="${HEBBIAN_DECAY_RATE:-0.995}"
export HEBBIAN_ACTIVATION_ALPHA="${HEBBIAN_ACTIVATION_ALPHA:-0.1}"
export HEBBIAN_SPREADING_THRESHOLD="${HEBBIAN_SPREADING_THRESHOLD:-0.4}"
export HEBBIAN_MAX_FLIPPED="${HEBBIAN_MAX_FLIPPED:-3}"
export HEBBIAN_KEYWORD_WEIGHT="${HEBBIAN_KEYWORD_WEIGHT:-0.7}"
export HEBBIAN_TOP_K="${HEBBIAN_TOP_K:-15}"
export HEBBIAN_SEMANTIC_TOP_K="${HEBBIAN_SEMANTIC_TOP_K:-5}"

python -m hela_mem.eval_longmemeval \
  --data_path "$DATA_PATH" \
  --mem_dir "$MEM_DIR" \
  --workers "$WORKERS" \
  --top_k "${HEBBIAN_TOP_K}" \
  --semantic_top_k "${HEBBIAN_SEMANTIC_TOP_K}"
