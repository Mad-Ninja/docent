"""Retrieval evaluation on the golden set: does search put the right page near the top?

    python -m app.evals                      vector search, k=6
    python -m app.evals --search hybrid      vector + full-text (RRF); `text` = full-text only
    python -m app.evals --kind keyword       only questions with exact code names
    python -m app.evals --per-page 2         at most 2 chunks from one page (diversity)

Metrics (questions with an expected page only; out-of-scope ones are for the agent eval):
- hit@k: share of questions where any expected page is among the top k chunks;
- MRR: mean of 1 / rank of the first expected page (1.0 = always first, 0 = never found);
- distinct pages: how many different pages the top k covers — duplicates of one page waste the answer's context.
"""
import argparse
import json
import time
from pathlib import Path

from .search import Hit, hybrid_search, text_search, vector_search

SEARCHES = {"vector": vector_search, "text": text_search, "hybrid": hybrid_search}

GOLDEN = Path(__file__).resolve().parents[2] / "evals" / "golden.jsonl"


def load_golden() -> list[dict]:
    return [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]


def first_rank(hits: list[Hit], expected: list[str]) -> int | None:
    return next((i for i, h in enumerate(hits, start=1) if h.path in expected), None)


def evaluate(search, k: int, kind: str | None = None) -> dict:
    """`kind`: "keyword" = questions with exact code names, "natural" = the rest, None = all."""
    cases = [c for c in load_golden() if c["expected"]
             and (kind is None or c.get("kind", "natural") == kind)]
    ranks, pages, misses = [], [], []
    started = time.perf_counter()
    for c in cases:
        hits = search(c["question"], k)
        rank = first_rank(hits, c["expected"])
        ranks.append(rank)
        pages.append(len({h.path for h in hits}))
        if rank is None:
            misses.append(f"{c['id']}: got {', '.join(dict.fromkeys(h.path for h in hits[:3]))}")
    n = len(cases)
    return {"questions": n, "k": k,
            f"hit@{k}": round(sum(r is not None for r in ranks) / n, 3),
            "hit@1": round(sum(r == 1 for r in ranks) / n, 3),
            "mrr": round(sum(1 / r for r in ranks if r) / n, 3),
            "distinct_pages": round(sum(pages) / n, 1),
            "ms_per_query": int((time.perf_counter() - started) * 1000 / n),
            "misses": misses}


def diversified(base, per_page: int):
    """Over-fetch, then keep at most `per_page` chunks of each page, preserving order."""
    def search(q: str, k: int) -> list[Hit]:
        out, seen = [], {}
        for h in base(q, k * 4):
            if seen.get(h.path, 0) < per_page:
                seen[h.path] = seen.get(h.path, 0) + 1
                out.append(h)
            if len(out) == k:
                break
        return out
    return search


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, default=6)
    ap.add_argument("--per-page", type=int, default=0)
    ap.add_argument("--search", choices=SEARCHES, default="vector")
    ap.add_argument("--kind", choices=["natural", "keyword"], default=None)
    args = ap.parse_args()
    base = SEARCHES[args.search]
    search = diversified(base, args.per_page) if args.per_page else base
    print(json.dumps(evaluate(search, args.k, args.kind), indent=2, ensure_ascii=False))
