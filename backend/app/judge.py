"""Answer evaluation: a second model grades every generated answer against the sources it was given.

    python -m app.judge --mode rag      one search, then the answer (baseline)
    python -m app.judge --mode agent    the agent collects evidence, then the answer

Two kinds of checks:
- code checks, free and exact: are there citations at all, does every [n] point to a real source,
  is a page that answers the question among the cited ones;
- a judge model (a stronger model than the one answering) for what code cannot check:
  does the answer address the question, is every claim supported by the sources, do citations back their claims.

The judge sees only the question, the sources and the answer — not the expected page — so it grades
faithfulness to the evidence, not agreement with my golden set. Every answer and verdict is saved to
evals/runs/ so the judge can be checked against hand grading (`--sample` prints a few to grade by hand).
"""
import argparse
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from .agent import Outcome, Tools, pick_planner, run_agent
from .agent_evals import RUNS
from .evals import load_golden
from .llm import ClaudeCodeGenerator
from .search import Hit, read_document, vector_search

JUDGE_MODEL = "claude-sonnet-5-5"  # newer than the "sonnet" alias that writes the answers: the grader should not be the answering model

JUDGE_PROMPT = """You grade answers of a documentation support assistant. You get a user question, the numbered
sources the assistant was given, and its answer. Judge only against these sources, not your own knowledge.

Return exactly one JSON object:
{"answers_question": true|false,   // does it address what was asked (a correct "the docs do not cover this" counts as false here)
 "grounded": true|false,           // every factual claim and code line is supported by the sources
 "citations_ok": true|false,       // each [n] supports the claim it is attached to
 "unsupported": ["..."],           // claims not supported by the sources, quoted briefly; [] if none
 "reason": "one sentence"}"""

CITATION = re.compile(r"\[(\d+)\]")
REFUSAL = re.compile(r"(does not|doesn't|do not|don't) (cover|contain|mention)|not covered", re.I)


def answer(case: dict, mode: str) -> tuple[list[Hit], str, int]:
    if mode == "agent":
        outcome = next(x for x in run_agent(case["question"], pick_planner("claude-code"),
                                            Tools(vector_search, read_document)) if isinstance(x, Outcome))
        evidence, steps = outcome.evidence, len(outcome.steps)
    else:
        evidence, steps = vector_search(case["question"], 6), 1
    text = "".join(ClaudeCodeGenerator().stream(case["question"], evidence))
    return evidence, text, steps


def code_checks(text: str, evidence: list[Hit], expected: list[str]) -> dict:
    cited = sorted({int(n) for n in CITATION.findall(text)})
    valid = [n for n in cited if 1 <= n <= len(evidence)]
    return {"cited": cited,
            "has_citations": bool(cited),
            "bad_citation_numbers": [n for n in cited if n not in valid],
            "cites_expected_page": any(evidence[n - 1].path in expected for n in valid),
            "refused": bool(REFUSAL.search(text))}


def judge(question: str, evidence: list[Hit], text: str) -> dict:
    sources = "\n\n".join(f'<source n="{n}" path="{h.path}">\n{h.content}\n</source>'
                          for n, h in enumerate(evidence, start=1))
    prompt = f"Question: {question}\n\n{sources}\n\n<answer>\n{text}\n</answer>"
    proc = subprocess.run(["claude", "-p", prompt, "--system-prompt", JUDGE_PROMPT, "--model", JUDGE_MODEL,
                           "--tools", ""], capture_output=True, text=True, timeout=240)
    if proc.returncode:
        raise RuntimeError(f"judge failed: {(proc.stderr or proc.stdout)[-300:]}")
    return parse_verdict(proc.stdout)


def parse_verdict(raw: str) -> dict:
    """The first JSON object in the reply (the judge may add prose around it)."""
    start = raw.find("{")
    try:
        data, _ = json.JSONDecoder().raw_decode(raw[start:]) if start >= 0 else (None, 0)
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict):
        return {"judge_error": raw[:200]}
    return {k: data.get(k) for k in ("answers_question", "grounded", "citations_ok", "unsupported", "reason")}


def run_case(case: dict, mode: str) -> dict:
    started = time.perf_counter()
    try:
        evidence, text, steps = answer(case, mode)
        row = {"id": case["id"], "question": case["question"], "answer": text, "steps": steps,
               "sources": [h.path for h in evidence],
               **code_checks(text, evidence, case["expected"]), **judge(case["question"], evidence, text)}
    except Exception as e:
        row = {"id": case["id"], "error": repr(e)[:300]}
    row["ms"] = int((time.perf_counter() - started) * 1000)
    return row


def rate(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if isinstance(r.get(key), bool)]
    return round(sum(vals) / len(vals), 3) if vals else None


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r and "judge_error" not in r]
    passed = [r for r in ok if r["answers_question"] and r["grounded"] and r["citations_ok"]
              and r["has_citations"] and not r["bad_citation_numbers"]]
    return {"questions": len(rows), "errors": len(rows) - len(ok),
            "pass": round(len(passed) / len(ok), 3) if ok else None,
            **{k: rate(ok, k) for k in ("answers_question", "grounded", "citations_ok",
                                         "has_citations", "cites_expected_page", "refused")},
            "bad_citation_numbers": sum(bool(r["bad_citation_numbers"]) for r in ok),
            "failed": {r["id"]: r.get("reason", "") for r in ok if r not in passed}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rag", "agent"], default="rag")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()

    # out-of-scope questions are graded by the hand-off metrics in app.agent_evals, not here
    cases = [c for c in load_golden() if c["expected"] and (not args.only or c["id"] in args.only)]
    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(lambda c: run_case(c, args.mode), cases))

    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / f"{datetime.now():%Y-%m-%d_%H%M}_judge_{args.mode}.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(json.dumps({"run": out.name, **summarize(rows)}, indent=2, ensure_ascii=False))
