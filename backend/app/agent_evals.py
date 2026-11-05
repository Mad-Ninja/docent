"""Agent evaluation: does the agent collect the right evidence, and hand off only when it should?

    python -m app.agent_evals --planner search-once     the plain-RAG baseline (no model, seconds)
    python -m app.agent_evals --planner claude-code     the agent, through the local `claude -p`
    python -m app.agent_evals --planner anthropic       the agent, through the Anthropic API

Every run writes one JSON line per question to evals/runs/ so two runs can be diffed question by question.

Metrics:
- evidence hit: an expected page is among the chunks the agent hands to the answer step;
- hand-off precision / recall: out-of-scope questions should be handed off, documented ones should not;
- steps and latency per question: what the extra quality costs.
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from .agent import Outcome, pick_planner, run_agent, Tools
from .evals import load_golden
from .search import read_document, vector_search

RUNS = Path(__file__).resolve().parents[2] / "evals" / "runs"


def run_case(case: dict, planner_name: str) -> dict:
    started = time.perf_counter()
    try:
        outcome = next(x for x in run_agent(case["question"], pick_planner(planner_name),
                                            Tools(vector_search, read_document)) if isinstance(x, Outcome))
    except Exception as e:  # one broken question must not kill a 20-minute run
        return {"id": case["id"], "error": repr(e)[:300], "ms": int((time.perf_counter() - started) * 1000)}
    pages = list(dict.fromkeys(h.path for h in outcome.evidence))
    return {"id": case["id"], "kind": case.get("kind", "natural"),
            "should_handoff": bool(case.get("handoff")), "outcome": outcome.kind,
            "evidence_hit": any(p in case["expected"] for p in pages) if case["expected"] else None,
            "pages": pages[:8], "handoff_reason": outcome.handoff_reason,
            "steps": [{"tool": s.tool, "input": s.input, "error": s.error} for s in outcome.steps],
            "ms": int((time.perf_counter() - started) * 1000)}


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    scoped = [r for r in ok if not r["should_handoff"]]
    handed = [r for r in ok if r["outcome"] == "handoff"]
    should = [r for r in ok if r["should_handoff"]]
    true_handoffs = sum(r["should_handoff"] for r in handed)
    ms = sorted(r["ms"] for r in ok)
    return {
        "questions": len(rows), "errors": len(rows) - len(ok),
        "evidence_hit": round(sum(bool(r["evidence_hit"]) for r in scoped) / max(len(scoped), 1), 3),
        "handoff_precision": round(true_handoffs / len(handed), 3) if handed else None,
        "handoff_recall": round(true_handoffs / len(should), 3) if should else None,
        "step_limit_hits": sum(r["outcome"] == "limit" for r in ok),
        "avg_steps": round(sum(len(r["steps"]) for r in ok) / max(len(ok), 1), 2),
        "p50_ms": ms[len(ms) // 2] if ms else None,
        "p90_ms": ms[int(len(ms) * 0.9)] if ms else None,
        "missed": [r["id"] for r in scoped if not r["evidence_hit"]],
        "wrong_handoffs": [r["id"] for r in handed if not r["should_handoff"]],
        "missed_handoffs": [r["id"] for r in should if r["outcome"] != "handoff"],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", choices=["search-once", "claude-code", "anthropic"], default="search-once")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", nargs="*", help="question ids")
    args = ap.parse_args()

    cases = [c for c in load_golden() if not args.only or c["id"] in args.only]
    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(lambda c: run_case(c, args.planner), cases))

    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / f"{datetime.now():%Y-%m-%d_%H%M}_{args.planner}.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(json.dumps({"run": out.name, **summarize(rows)}, indent=2, ensure_ascii=False))
