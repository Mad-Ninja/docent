"""/chat: answers as a stream of Server-Sent Events.

Events: {"type": "sources", "items": [...]}, then {"type": "token", "text": "..."} chunks, then {"type": "done"}.
Until the LLM is wired in (day 3) the answer is extractive: the best passages with their citations. The event format
stays the same, so the UI does not change when generation arrives."""
import json
import time
from collections.abc import Iterator

from .search import Hit, vector_search


def sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def extractive_answer(hits: list[Hit]) -> str:
    if not hits:
        return "I could not find this in the documentation."
    parts = [f"[{n}] {h.heading}\n{h.content[:400].strip()}…" for n, h in enumerate(hits[:3], start=1)]
    return "Most relevant passages from the documentation:\n\n" + "\n\n".join(parts)


def stream_answer(question: str, k: int) -> Iterator[str]:
    hits = vector_search(question, k)
    yield sse({"type": "sources", "items": [
        {"n": n, "path": h.path, "heading": h.heading, "score": round(h.score, 3)} for n, h in enumerate(hits, start=1)]})
    text = extractive_answer(hits)
    for i in range(0, len(text), 24):  # small chunks, like model tokens
        yield sse({"type": "token", "text": text[i:i + 24]})
        time.sleep(0.01)
    yield sse({"type": "done"})
