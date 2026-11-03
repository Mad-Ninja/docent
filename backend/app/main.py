from fastapi import FastAPI, Query

from .search import as_dicts, vector_search

app = FastAPI(title="kb-agent", description="Support agent over a documentation knowledge base")


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/search")
def search(q: str = Query(min_length=2), k: int = Query(6, ge=1, le=20)) -> list[dict]:
    return as_dicts(vector_search(q, k))
