# HeLa-Mem

Code for **HeLa-Mem: Hebbian Learning and Associative Memory for LLM Agents** — accepted by **ACL 2026 (main)**.

Paper: [arXiv:2604.16839](https://arxiv.org/abs/2604.16839)

![Framework](https://arxiv.org/html/2604.16839/x2.png)

Code for HeLa-Mem on LongMemEval-S and LoCoMo.

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

## Included Code

```text
HeLa-Mem/
├── hela_mem/
│   ├── encode_longmemeval.py
│   ├── encode_locomo.py
│   ├── eval_longmemeval.py
│   ├── eval_locomo.py
│   ├── hebbian_knowledge_memory.py
│   ├── hebbian_memory.py
│   ├── hebbian_retriever.py
│   ├── profile_utils.py
│   ├── reranker.py
│   └── utils.py
├── scripts/
│   ├── encode_longmemeval.sh
│   ├── encode_locomo.sh
│   ├── eval_longmemeval.sh
│   └── eval_locomo.sh
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

### LongMemEval-S

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

### LoCoMo

Expected fields per sample:

- `sample_id`
- `conversation`
- `conversation.speaker_a`
- `conversation.speaker_b`
- `conversation.session_<n>`
- `conversation.session_<n>_date_time`
- `qa`

Expected fields per QA:

- `question`
- `answer` or `adversarial_answer`
- `category`
- `evidence`

Place the LoCoMo file at `data/locomo10.json`, or pass its path with `--data_path`.

## Experiment Entry Points

Both benchmarks use a two-stage memory workflow.

LongMemEval-S:

1. `encode_longmemeval.py`
2. `eval_longmemeval.py`

LoCoMo:

1. `encode_locomo.py`
2. `eval_locomo.py`

Encoding builds:

- `*_hebbian.json`
- `*_long_term.json`
- `*_long_term_kb_graph.json`

Evaluation does:

- episodic retrieval
- semantic retrieval
- answer generation
- benchmark-specific scoring
- per-item result saving
- summary aggregation

## Reproduce

Configure standard OpenAI credentials:

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

### LongMemEval-S

Run the full `500`-item experiment.

#### 1. Encode

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

#### 2. Evaluate

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

### LoCoMo

Put `locomo10.json` at `data/locomo10.json`, or pass its path as the first argument to the shell scripts.

#### 1. Encode

```bash
bash scripts/encode_locomo.sh
```

Or directly:

```bash
python -m hela_mem.encode_locomo \
  --data_path data/locomo10.json \
  --output_dir results/locomo_mem_full \
  --workers 5
```

#### 2. Evaluate

```bash
bash scripts/eval_locomo.sh
```

Or directly:

```bash
python -m hela_mem.eval_locomo \
  --data_path data/locomo10.json \
  --mem_dir results/locomo_mem_full \
  --workers 5 \
  --top_k 10 \
  --knowledge_top_k 5
```

Outputs are written under:

- `results/.../eval_results/result_<sample_id>.json`
- `results/.../eval_results/eval_summary.json`

LoCoMo evaluation reports F1 and BLEU-1 overall, by sample, and by question category. For a smaller sanity-check run, add `--num_samples 1` and/or `--max_qa_per_sample 10`.

## Notes

- This release keeps the original experiment-style environment variable names (`HEBBIAN_*`) so existing commands map cleanly.
- API-key rotation is still supported, but keys must now come from environment variables or a local keys file.
- The code uses the standard OpenAI Python SDK request pattern (`client.chat.completions.create`) with `OPENAI_API_KEY` and the official OpenAI base URL by default.
- The repository has been cleaned for release, but the benchmark paths are kept source-aligned rather than simplified.

## Citation

```bibtex
@article{zhu2026hela,
  title={HeLa-Mem: Hebbian Learning and Associative Memory for LLM Agents},
  author={Zhu, Jinchang and Li, Jindong and Zhang, Cheng and Liu, Jiahong and Yang, Menglin},
  journal={arXiv preprint arXiv:2604.16839},
  year={2026}
}
```
