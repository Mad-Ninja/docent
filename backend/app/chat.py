"""/chat: answers as a stream of Server-Sent Events.

Events: {"type": "sources", "items": [...], "backend": "..."}, then {"type": "token", "text": "..."} chunks,
then {"type": "done"}, or {"type": "error", "message": "..."} if generation fails midway."""
import json
from collections.abc import Iterator

from .llm import pick_generator
from .search import vector_search


def sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def stream_answer(question: str, k: int) -> Iterator[str]:
    hits = vector_search(question, k)
    generator = pick_generator()
    yield sse({"type": "sources", "backend": generator.name, "items": [
        {"n": n, "path": h.path, "heading": h.heading, "score": round(h.score, 3)} for n, h in enumerate(hits, start=1)]})
    try:
        for text in generator.stream(question, hits):
            if text:
                yield sse({"type": "token", "text": text})
    except Exception as e:  # the stream has started: report in-band instead of breaking the HTTP response
        yield sse({"type": "error", "message": str(e)[:300]})
        return
    yield sse({"type": "done"})
