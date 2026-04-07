# HeLa-Mem

Minimal open-source release of the HeLa-Mem LongMemEval experiment code.

This repository contains only the code needed to run the `gpt-4o-mini` LongMemEval experiment:

- LongMemEval encoding
- Hebbian episodic memory graph
- Hebbian knowledge memory
- LongMemEval evaluation with GPT judge

It intentionally does not include:

- LoCoMo code
- MemoryOS code
- MemoryChain code
- old result folders
- paper assets
- unrelated agent experiments

## Repository Layout

```text
HeLa-Mem/
├── hela_mem/
│   ├── encode_longmemeval.py
│   ├── eval_longmemeval.py
│   ├── hebbian_knowledge_memory.py
│   ├── hebbian_memory.py
│   ├── hebbian_retriever.py
│   ├── profile_extraction.py
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

Set your OpenAI-compatible credentials:

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

If you want request rotation across multiple keys:

```bash
export OPENAI_API_KEYS="key1,key2,key3"
```

The default model is `gpt-4o-mini`.

## Dataset

This repo does not ship LongMemEval-S.

Expected input is a JSON array where each item has:

- `question_id`
- `question`
- `answer`
- `question_type`
- `question_date`
- `haystack_dates`
- `haystack_sessions`

The original code was run with files such as:

- `eval/longmemeval/longmemeval_s_50.json`
- `eval/longmemeval/longmemeval_s_100.json`
- `eval/longmemeval/longmemeval_s_150.json`
- `eval/longmemeval/longmemeval_s_200.json`

## Run Encoding

The release keeps the paper-side defaults for the `gpt-4o-mini` LongMemEval run:

- `tau = 1e7`
- `learning_rate = 0.02`
- `decay_rate = 0.995`
- `activation_alpha = 0.1`
- `spreading_threshold = 0.4`
- `max_flipped = 5`

Example:

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

The release keeps the paper-side evaluation defaults for the `gpt-4o-mini` run:

- `top_k = 20`
- `semantic_top_k = 5`
- `max_flipped = 1`
- `keyword_weight = 0.7`

Example:

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

Evaluation writes:

- per-item outputs to `results/.../eval_results/result_<question_id>.json`
- overall summary to `results/.../eval_results/eval_summary.json`

## Environment Variables

Supported configuration uses `HELA_MEM_*` names and also accepts the old `HEBBIAN_*` names as fallback.

Common options:

- `HELA_MEM_MODEL`
- `HELA_MEM_TAU`
- `HELA_MEM_LEARNING_RATE`
- `HELA_MEM_DECAY_RATE`
- `HELA_MEM_ACTIVATION_ALPHA`
- `HELA_MEM_SPREADING_THRESHOLD`
- `HELA_MEM_MAX_FLIPPED`
- `HELA_MEM_KB_MAX_FLIPPED`
- `HELA_MEM_KEYWORD_WEIGHT`
- `HELA_MEM_USE_TIME_DECAY`
- `HELA_MEM_USE_KEYWORD_MATCH`
- `HELA_MEM_KNOWLEDGE_BUFFER_SIZE`

## Validation

The code was cleaned to remove hardcoded private API keys and old repo couplings.

For a no-credentials smoke test, encoding can run with:

```bash
python -m hela_mem.encode_longmemeval \
  --data_path /path/to/longmemeval_s_50.json \
  --output_dir results/smoke \
  --num_items 1 \
  --workers 1 \
  --skip_knowledge
```

In that mode, keyword extraction falls back to a local heuristic instead of LLM calls.
If `sentence-transformers` cannot be loaded, the code also falls back to a deterministic hashed embedding for smoke validation only. The actual paper experiment should use the normal `all-MiniLM-L6-v2` embedding path.
