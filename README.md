# HeLa-Mem

Code for HeLa-Mem on LongMemEval.

LoCoMo evaluation code will be released in the future.

## LongMemEval Result

The table below shows the target reproduce results of HeLa-Mem on LongMemEval-S on the full `500`-item benchmark:

| Method | Overall ACC |
| --- | ---: |
| LangMem | 37.20 |
| MemoryOS | 44.80 |
| Mem0 | 53.61 |
| FullText | 56.80 |
| NaiveRAG | 61.00 |
| A-MEM | 62.60 |
| **HeLa-Mem (Ours)** | **65.40** |

Best paper configuration:

- `HEBBIAN_MAX_FLIPPED=3`
- `HEBBIAN_KEYWORD_WEIGHT=0.7`
- `HEBBIAN_ACTIVATION_ALPHA=0.1`
- `HEBBIAN_SPREADING_THRESHOLD=0.4`
- `HEBBIAN_LEARNING_RATE=0.02`
- `HEBBIAN_DECAY_RATE=0.995`
- `HEBBIAN_TAU=1e7`
- `top_k=15` episodic
- `semantic_top_k=5`
- `gpt-4o-mini` backbone
- `gpt-4o-mini` LLM-as-judge

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

Expected fields per item:

- `question_id`
- `question`
- `answer`
- `question_type`
- `question_date`
- `haystack_dates`
- `haystack_sessions`

The complete `500`-item LongMemEval-S file is bundled in this repository:

- [data/longmemeval_s.json](/Users/h1syu1/PythonProjects/HeLa-Mem/data/longmemeval_s.json)

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

## Reproduce

Configure standard OpenAI credentials:

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

Then run the full `500`-item experiment.

### 1. Encode

```bash
bash scripts/encode_longmemeval.sh
```

Or directly:

```bash
python -m hela_mem.encode_longmemeval \
  --data_path data/longmemeval_s.json \
  --output_dir results/longmemeval_mem_full \
  --workers 8
```

### 2. Evaluate

```bash
bash scripts/eval_longmemeval.sh
```

Or directly:

```bash
python -m hela_mem.eval_longmemeval \
  --data_path data/longmemeval_s.json \
  --mem_dir results/longmemeval_mem_full \
  --workers 8 \
  --top_k 15 \
  --semantic_top_k 5
```

Outputs are written under:

- `results/.../eval_results/result_<question_id>.json`
- `results/.../eval_results/eval_summary.json`

If you want a smaller sanity-check run, keep the same dataset file and add `--num_items 100` or another cap to both encode and eval.

## Notes

- This release keeps the original experiment-style environment variable names (`HEBBIAN_*`) so existing commands map cleanly.
- API-key rotation is still supported, but keys must now come from environment variables or a local keys file.
- The code uses the standard OpenAI Python SDK request pattern (`client.chat.completions.create`) with `OPENAI_API_KEY` and the official OpenAI base URL by default.
- The repository has been cleaned for release, but the LongMemEval path is kept source-aligned rather than simplified.
