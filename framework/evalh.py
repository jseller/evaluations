#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["litellm>=1.50", "pydantic>=2"]
# ///
"""evalh — a minimal LLM evaluation harness.

Pipeline:  gold dataset  ->  inference  ->  judge  ->  JSONL log  ->  summary

  uv run evalh.py run gold.jsonl --model anthropic/claude-haiku-5-5
  uv run evalh.py rescore results.jsonl --judge llm
  uv run evalh.py summary results.jsonl
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import litellm
from pydantic import BaseModel

litellm.suppress_debug_info = True
litellm.drop_params = True  # e.g. newer Claude models reject temperature=0; drop instead of failing

DEFAULT_MODEL = "anthropic/claude-haiku-5-5"
DEFAULT_JUDGE_MODEL = "anthropic/claude-sonnet-5-5"


# ---------------------------------------------------------------- 1. gold data

def load_gold(path: str) -> list[dict]:
    """Each line: {"id": ..., "input": ..., "expected_concept": ...}"""
    cases = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        case = json.loads(line)
        missing = {"id", "input", "expected_concept"} - case.keys()
        if missing:
            sys.exit(f"{path}:{n}: missing fields {sorted(missing)}")
        cases.append(case)
    return cases


# ---------------------------------------------------------------- 2. inference

async def run_case(case: dict, model: str, sem: asyncio.Semaphore, mock: bool) -> dict:
    async with sem:
        start = time.perf_counter()
        try:
            response = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": case["input"]}],
                num_retries=3,
                timeout=30,
                **({"mock_response": _mock_answer(case)} if mock else {}),
            )
            output, error = response.choices[0].message.content, None
        except Exception as e:  # keep going; one bad call shouldn't sink the run
            output, error = "", f"{type(e).__name__}: {e}"
        latency_ms = round((time.perf_counter() - start) * 1000)
    return {**case, "model": model, "raw_output": output, "latency_ms": latency_ms, "error": error}


def _mock_answer(case: dict) -> str:
    # alternate hits and misses so --mock shows both judge paths
    return f"mock answer about {case['expected_concept']}" if hash(case["id"]) % 2 else "mock answer, paraphrased"


# ---------------------------------------------------------------- 3. judges

def deterministic_judge(result: dict) -> dict:
    """Cheap and reproducible: case-insensitive substring match."""
    passed = result["expected_concept"].lower() in (result["raw_output"] or "").lower()
    return {**result, "passed": passed, "judge": "deterministic", "reason": None}


class JudgeScore(BaseModel):
    passed: bool
    reason: str


JUDGE_PROMPT = """You are grading another model's response.

Input: {input}
Response: {raw_output}
Criteria: the response must convey the concept "{expected_concept}".

Did the response meet the criteria? Reply only with JSON: {{"passed": true|false, "reason": "<one sentence>"}}"""


async def llm_judge(result: dict, judge_model: str, sem: asyncio.Semaphore, mock: bool) -> dict:
    """Semantic grading: catches correct answers that don't use the exact words."""
    if result.get("error"):  # inference failed; nothing to grade
        return {**result, "passed": False, "judge": "llm", "reason": "inference error"}
    async with sem:
        try:
            response = await litellm.acompletion(
                model=judge_model,
                messages=[{"role": "user", "content": JUDGE_PROMPT.format(**result)}],
                temperature=0,
                num_retries=3,
                timeout=30,
                **({"mock_response": '{"passed": true, "reason": "mock judge: semantically equivalent"}'} if mock else {}),
            )
            # parse with pydantic rather than provider-specific response_format, so any judge model works
            score = JudgeScore.model_validate_json(_extract_json(response.choices[0].message.content))
            passed, reason = score.passed, score.reason
        except Exception as e:
            passed, reason = False, f"judge error: {type(e).__name__}: {e}"
    return {**result, "passed": passed, "judge": "llm", "judge_model": judge_model, "reason": reason}


def _extract_json(text: str) -> str:
    # some providers wrap JSON in prose or ``` fences
    return text[text.find("{"): text.rfind("}") + 1]


async def judge_all(results, mode, judge_model, sem, mock):
    if mode == "deterministic":
        return [deterministic_judge(r) for r in results]
    if mode == "llm":
        return await asyncio.gather(*(llm_judge(r, judge_model, sem, mock) for r in results))
    # hybrid: substring match first, only escalate failures to the (costlier) LLM judge
    first = [deterministic_judge(r) for r in results]
    escalate = [r for r in first if not r["passed"]]
    rejudged = {r["id"]: r for r in await asyncio.gather(*(llm_judge(r, judge_model, sem, mock) for r in escalate))}
    return [rejudged.get(r["id"], r) for r in first]


# ---------------------------------------------------------------- 4. logging + report

def log_results(results: list[dict], path: str) -> None:
    """Append-only JSONL: raw outputs are kept so you can re-judge without re-running inference."""
    ts = datetime.now(timezone.utc).isoformat()
    with open(path, "a") as f:
        for r in results:
            f.write(json.dumps({**r, "timestamp": ts}) + "\n")


def summarize(results: list[dict]) -> dict:
    total = len(results)
    passed = sum(r["passed"] for r in results)
    lat = sorted(r["latency_ms"] for r in results if r.get("latency_ms") is not None)
    return {
        "total": total,
        "passed": passed,
        "pass_rate": passed / total if total else 0.0,
        "errors": sum(1 for r in results if r.get("error")),
        "p50_latency_ms": lat[len(lat) // 2] if lat else None,
    }


def print_report(results: list[dict]) -> None:
    for r in results:
        mark = "\033[32mPASS\033[0m" if r["passed"] else "\033[31mFAIL\033[0m"
        print(f"{mark}  {r['id']:<18} [{r['judge']}]", end="")
        if r.get("error"):
            print(f"  error: {r['error'][:80]}", end="")
        elif r.get("reason"):
            print(f"  {r['reason'][:90]}", end="")
        print()
    s = summarize(results)
    print(f"\n{s['passed']}/{s['total']} passed  ({s['pass_rate']:.0%})"
          f"  errors={s['errors']}  p50_latency={s['p50_latency_ms']}ms")


def read_jsonl(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


# ---------------------------------------------------------------- CLI

async def cmd_run(a):
    cases = load_gold(a.gold)
    if a.limit:
        cases = cases[: a.limit]
    sem = asyncio.Semaphore(a.concurrency)
    print(f"Running {len(cases)} cases on {a.model} (judge: {a.judge})...\n", file=sys.stderr)
    raw = await asyncio.gather(*(run_case(c, a.model, sem, a.mock) for c in cases))
    judged = await judge_all(raw, a.judge, a.judge_model, sem, a.mock)
    log_results(judged, a.out)
    print_report(judged)
    print(f"\nlogged to {a.out}", file=sys.stderr)
    return judged


async def cmd_rescore(a):
    # re-judge stored outputs with a different judge -- no new inference calls
    results = [{k: v for k, v in r.items() if k not in ("passed", "judge", "reason", "judge_model", "timestamp")}
               for r in read_jsonl(a.results)]
    judged = await judge_all(results, a.judge, a.judge_model, asyncio.Semaphore(a.concurrency), a.mock)
    if a.out:
        log_results(judged, a.out)
    print_report(judged)
    return judged


def main():
    p = argparse.ArgumentParser(prog="evalh", description="Minimal LLM eval harness: gold -> inference -> judge")
    sub = p.add_subparsers(dest="cmd", required=True)

    def judge_opts(sp):
        sp.add_argument("--judge", choices=["deterministic", "llm", "hybrid"], default="hybrid",
                        help="hybrid = substring match, escalate failures to LLM judge (default)")
        sp.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
        sp.add_argument("--concurrency", type=int, default=10)
        sp.add_argument("--mock", action="store_true", help="fake all LLM calls (no API key needed)")

    r = sub.add_parser("run", help="run inference on a gold dataset and judge it")
    r.add_argument("gold", help="gold dataset (.jsonl)")
    r.add_argument("--model", default=DEFAULT_MODEL, help="any LiteLLM model string, e.g. openai/gpt-4o, ollama/llama3")
    r.add_argument("--out", default="results.jsonl")
    r.add_argument("--limit", type=int, help="only run the first N cases")
    judge_opts(r)

    rs = sub.add_parser("rescore", help="re-judge saved outputs without re-running inference")
    rs.add_argument("results", help="results .jsonl from a previous run")
    rs.add_argument("--out", help="append re-judged records here")
    judge_opts(rs)

    s = sub.add_parser("summary", help="print pass rate for a results file")
    s.add_argument("results")

    a = p.parse_args()
    if a.cmd == "run":
        asyncio.run(cmd_run(a))
    elif a.cmd == "rescore":
        asyncio.run(cmd_rescore(a))
    else:
        print(json.dumps(summarize(read_jsonl(a.results)), indent=2))


if __name__ == "__main__":
    main()
