"""/chat: answers as a stream of Server-Sent Events.

Events, in order:
  {"type": "step", ...}                       agent mode only: one per tool call, as it finishes
  {"type": "sources", "items": [...], "backend": "..."}
  {"type": "token", "text": "..."}            answer chunks
  {"type": "handoff", "ticket": 12, "reason": "..."}   instead of tokens, when a person takes over
  {"type": "done"} | {"type": "error", "message": "..."}
"""
import json
import time
from collections.abc import Iterator

from psycopg.types.json import Jsonb

from .agent import Outcome, Step, Tools, pick_planner, run_agent, step_dict
from .db import connect
from .llm import pick_generator
from .search import Hit, read_document, vector_search

HANDOFF_TEXT = "I've passed your question to our support team — a person will reply here."


def sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def sources_event(hits: list[Hit], backend: str) -> str:
    return sse({"type": "sources", "backend": backend, "items": [
        {"n": n, "path": h.path, "heading": h.heading, "score": round(h.score, 3)} for n, h in enumerate(hits, start=1)]})


def stream_answer(question: str, k: int, mode: str = "agent") -> Iterator[str]:
    started = time.perf_counter()
    generator = pick_generator()
    try:
        if mode == "agent":
            outcome = None
            for item in run_agent(question, pick_planner(generator.name), Tools(vector_search, read_document)):
                if isinstance(item, Step):
                    yield sse({"type": "step", **step_dict(item)})
                else:
                    outcome = item
        else:
            outcome = Outcome("answer", vector_search(question, k))

        if outcome.kind == "handoff":
            ticket = save_ticket(question, outcome.handoff_reason)
            yield sse({"type": "handoff", "ticket": ticket, "reason": outcome.handoff_reason})
            yield sse({"type": "token", "text": HANDOFF_TEXT})
        else:
            yield sources_event(outcome.evidence, generator.name)
            for text in generator.stream(question, outcome.evidence):
                if text:
                    yield sse({"type": "token", "text": text})
    except Exception as e:  # the stream has started: report in-band instead of breaking the HTTP response
        yield sse({"type": "error", "message": str(e)[:300]})
        return
    save_trace(question, mode, generator.name, outcome, int((time.perf_counter() - started) * 1000))
    yield sse({"type": "done"})


def save_ticket(question: str, reason: str) -> int:
    with connect() as conn:
        return conn.execute("INSERT INTO tickets (question, reason) VALUES (%s, %s) RETURNING id",
                            (question, reason)).fetchone()[0]


def save_trace(question: str, mode: str, backend: str, outcome: Outcome, total_ms: int) -> None:
    planner = backend if mode == "agent" else "rag"
    try:
        with connect() as conn:
            conn.execute(
                "INSERT INTO traces (question, planner, outcome, steps, sources, total_ms) VALUES (%s, %s, %s, %s, %s, %s)",
                (question, planner, outcome.kind, Jsonb([step_dict(s) for s in outcome.steps]),
                 Jsonb([h.path for h in outcome.evidence]), total_ms))
    except Exception:
        pass  # tracing must never fail an answer the user already received
