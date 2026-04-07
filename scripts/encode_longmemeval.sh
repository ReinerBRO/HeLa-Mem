#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="${1:-data/longmemeval_s.json}"
OUTPUT_DIR="${2:-results/longmemeval_mem_full}"
WORKERS="${3:-8}"

export HEBBIAN_MODEL="${HEBBIAN_MODEL:-gpt-4o-mini}"
export HEBBIAN_TAU="${HEBBIAN_TAU:-1e7}"
export HEBBIAN_LEARNING_RATE="${HEBBIAN_LEARNING_RATE:-0.02}"
export HEBBIAN_DECAY_RATE="${HEBBIAN_DECAY_RATE:-0.995}"
export HEBBIAN_ACTIVATION_ALPHA="${HEBBIAN_ACTIVATION_ALPHA:-0.1}"
export HEBBIAN_SPREADING_THRESHOLD="${HEBBIAN_SPREADING_THRESHOLD:-0.4}"
export HEBBIAN_MAX_FLIPPED="${HEBBIAN_MAX_FLIPPED:-3}"
export HEBBIAN_KNOWLEDGE_BUFFER_SIZE="${HEBBIAN_KNOWLEDGE_BUFFER_SIZE:-10}"

python -m hela_mem.encode_longmemeval \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --workers "$WORKERS"
