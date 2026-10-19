# evalh — a minimal LLM eval harness

Based on [Building a Custom LLM Evaluation Harness in Python from Scratch](https://machinelearningmastery.com/building-a-custom-llm-evaluation-harness-in-python-from-scratch/).

```
gold.jsonl ──▶ inference (any model via LiteLLM) ──▶ judge ──▶ results.jsonl ──▶ pass rate
```

## Setup

Needs [uv](https://docs.astral.sh/uv/). Dependencies (`litellm`, `pydantic`) install on first run.

```bash
export ANTHROPIC_API_KEY=...      # or OPENAI_API_KEY, etc.
```

## Commands

```bash
# 1. read gold standard, run inference, judge
uv run evalh.py run gold.jsonl --model anthropic/claude-haiku-5-5

# swap the model under test — same dataset, same judge
uv run evalh.py run gold.jsonl --model openai/gpt-4o --out gpt4o.jsonl

# re-judge saved outputs with a different judge, without new inference calls
uv run evalh.py rescore results.jsonl --judge llm

# aggregate numbers
uv run evalh.py summary results.jsonl

# demo with no API key: all LLM calls are faked
uv run evalh.py run gold.jsonl --mock
```

## Gold standard format

One JSON object per line:

```json
{"id": "capital_001", "input": "What is the capital of Australia?", "expected_concept": "Canberra"}
```

Extra fields such as `category` are carried through to the results.

## Judges (`--judge`)

| mode | how it works | trade-off |
|---|---|---|
| `deterministic` | case-insensitive substring match on `expected_concept` | free, reproducible, but brittle (misses paraphrases) |
| `llm` | a judge model returns `{"passed": bool, "reason": str}` | grades meaning, but costs money and can be wrong |
| `hybrid` (default) | substring match first, sends only failures to the LLM judge | cheap and still catches paraphrases |

Results are appended to a JSONL file. Each record holds the raw output, latency, verdict, judge, and reason, so you can diff runs over time.
