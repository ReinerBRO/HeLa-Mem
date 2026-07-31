#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="${1:-data/locomo10.json}"
MEM_DIR="${2:-results/locomo_mem_full}"
WORKERS="${3:-5}"

export HEBBIAN_MODEL="${HEBBIAN_MODEL:-gpt-4o-mini}"
export HEBBIAN_TAU="${HEBBIAN_TAU:-5184000}"
export HEBBIAN_LEARNING_RATE="${HEBBIAN_LEARNING_RATE:-0.02}"
export HEBBIAN_DECAY_RATE="${HEBBIAN_DECAY_RATE:-0.995}"
export HEBBIAN_ACTIVATION_ALPHA="${HEBBIAN_ACTIVATION_ALPHA:-0.1}"
export HEBBIAN_SPREADING_THRESHOLD="${HEBBIAN_SPREADING_THRESHOLD:-0.6}"
export HEBBIAN_MAX_FLIPPED="${HEBBIAN_MAX_FLIPPED:-5}"
export HEBBIAN_KEYWORD_WEIGHT="${HEBBIAN_KEYWORD_WEIGHT:-0.5}"
export HEBBIAN_TOP_K="${HEBBIAN_TOP_K:-10}"
export HEBBIAN_KNOWLEDGE_TOP_K="${HEBBIAN_KNOWLEDGE_TOP_K:-5}"

python -m hela_mem.eval_locomo \
  --data_path "$DATA_PATH" \
  --mem_dir "$MEM_DIR" \
  --workers "$WORKERS" \
  --top_k "$HEBBIAN_TOP_K" \
  --knowledge_top_k "$HEBBIAN_KNOWLEDGE_TOP_K"
