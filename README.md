# HeLa-Mem

Open-source release of the HeLa-Mem `gpt-4o-mini` LongMemEval experiment.

This repository is not a toy rewrite. The LongMemEval path is carried over from the original experiment code and then cleaned for release:

- removed hardcoded private API keys
- removed dependencies on unrelated project directories
- kept the original LongMemEval encode/eval logic
- kept the original Hebbian memory, retriever, knowledge memory, and GPT-judge evaluation flow

The repository intentionally excludes unrelated codepaths such as LoCoMo, MemoryOS, MemoryChain, paper assets, and old result directories.

## Included Code

```text
HeLa-Mem/
├── hela_mem/
│   ├── encode_longmemeval.py
│   ├── eval_longmemeval.py
│   ├── hebbian_knowledge_memory.py
│   ├── hebbian_memory.py
│   ├── hebbian_retriever.py
│   ├── profile_utils.py
│   ├── reranker.py
│   └── utils.py
├── scripts/
│   ├── encode_longmemeval.sh
│   └── eval_longmemeval.sh
├── pyproject.toml
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Configure your API access:

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

If you want multi-key rotation, provide:

```bash
export OPENAI_API_KEYS="key1,key2,key3"
```

or

```bash
export OPENAI_API_KEYS_FILE="/path/to/keys.txt"
```

The default model is `gpt-4o-mini`.

## Dataset Format

This repository does not bundle LongMemEval-S.

Expected fields per item:

- `question_id`
- `question`
- `answer`
- `question_type`
- `question_date`
- `haystack_dates`
- `haystack_sessions`

## Experiment Entry Points

The original LongMemEval experiment here is two-stage:

1. `encode_longmemeval.py`
2. `eval_longmemeval.py`

Encoding builds:

- `*_hebbian.json`
- `*_long_term.json`
- `*_long_term_kb_graph.json`

Evaluation does:

- episodic retrieval
- semantic retrieval
- answer generation
- GPT judge scoring
- per-item result saving
- summary aggregation

## Paper-Aligned Defaults

The included shell scripts keep the paper-side `gpt-4o-mini` defaults from the original run.

Encoding defaults:

- `HEBBIAN_MODEL=gpt-4o-mini`
- `HEBBIAN_TAU=1e7`
- `HEBBIAN_LEARNING_RATE=0.02`
- `HEBBIAN_DECAY_RATE=0.995`
- `HEBBIAN_ACTIVATION_ALPHA=0.1`
- `HEBBIAN_SPREADING_THRESHOLD=0.4`
- `HEBBIAN_MAX_FLIPPED=5`
- `HEBBIAN_KNOWLEDGE_BUFFER_SIZE=10`

Evaluation defaults:

- `HEBBIAN_MODEL=gpt-4o-mini`
- `HEBBIAN_TAU=1e7`
- `HEBBIAN_LEARNING_RATE=0.02`
- `HEBBIAN_DECAY_RATE=0.995`
- `HEBBIAN_ACTIVATION_ALPHA=0.1`
- `HEBBIAN_SPREADING_THRESHOLD=0.4`
- `HEBBIAN_MAX_FLIPPED=1`
- `HEBBIAN_TOP_K=20`
- `HEBBIAN_SEMANTIC_TOP_K=5`
- `HEBBIAN_KEYWORD_WEIGHT=0.7`

## Run Encoding

```bash
bash scripts/encode_longmemeval.sh /path/to/longmemeval_s_200.json results/longmemeval_mem_200
```

Or directly:

```bash
python -m hela_mem.encode_longmemeval \
  --data_path /path/to/longmemeval_s_200.json \
  --output_dir results/longmemeval_mem_200 \
  --workers 8
```

## Run Evaluation

```bash
bash scripts/eval_longmemeval.sh /path/to/longmemeval_s_200.json results/longmemeval_mem_200
```

Or directly:

```bash
python -m hela_mem.eval_longmemeval \
  --data_path /path/to/longmemeval_s_200.json \
  --mem_dir results/longmemeval_mem_200 \
  --workers 8 \
  --top_k 20 \
  --semantic_top_k 5
```

Outputs are written under:

- `results/.../eval_results/result_<question_id>.json`
- `results/.../eval_results/eval_summary.json`

## Notes

- This release keeps the original experiment-style environment variable names (`HEBBIAN_*`) so existing commands map cleanly.
- API-key rotation is still supported, but keys must now come from environment variables or a local keys file.
- The repository has been cleaned for release, but the LongMemEval path is kept source-aligned rather than simplified.
