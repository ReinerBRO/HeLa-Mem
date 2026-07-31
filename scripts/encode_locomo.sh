#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="${1:-data/locomo10.json}"
OUTPUT_DIR="${2:-results/locomo_mem_full}"
WORKERS="${3:-5}"

export HEBBIAN_MODEL="${HEBBIAN_MODEL:-gpt-4o-mini}"
export HEBBIAN_TAU="${HEBBIAN_TAU:-5184000}"
export HEBBIAN_LEARNING_RATE="${HEBBIAN_LEARNING_RATE:-0.02}"
export HEBBIAN_DECAY_RATE="${HEBBIAN_DECAY_RATE:-0.995}"
export HEBBIAN_ACTIVATION_ALPHA="${HEBBIAN_ACTIVATION_ALPHA:-0.1}"
export HEBBIAN_SPREADING_THRESHOLD="${HEBBIAN_SPREADING_THRESHOLD:-0.6}"
export HEBBIAN_MAX_FLIPPED="${HEBBIAN_MAX_FLIPPED:-5}"
export HEBBIAN_KNOWLEDGE_BUFFER_SIZE="${HEBBIAN_KNOWLEDGE_BUFFER_SIZE:-10}"

python -m hela_mem.encode_locomo \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --workers "$WORKERS"
