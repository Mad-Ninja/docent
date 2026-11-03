# Docent

**AI support agent over documentation — RAG, tools, evals.** It answers questions with citations, decides which tools to call, and every change is measured by an evaluation suite.

Built with Claude Code as a coding agent: I design the architecture, review every change and own the result.

The knowledge base is the [FastAPI documentation](https://github.com/fastapi/fastapi/tree/master/docs/en/docs) (MIT, see `corpus/fastapi/LICENSE`).

**Status:** day 1 of 7 — ingestion and vector search.

| Part | Status |
|---|---|
| Ingestion: markdown → chunks with heading paths → embeddings → Postgres + pgvector | ✅ |
| Vector search `/search` | ✅ |
| Hybrid search (pgvector + full-text, RRF) | ⏳ |
| Answers with citations (Claude) | ⏳ |
| Agent with tools | ⏳ |
| Evaluation: retrieval metrics, LLM-as-judge, agent metrics | ⏳ |
| React UI | ⏳ |

## Run

```bash
docker compose up -d                      # Postgres 16 + pgvector on :5435
cd backend
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m app.ingest            # ~6 min on CPU: 156 pages → ~2400 chunks
.venv/bin/uvicorn app.main:app --reload   # http://127.0.0.1:8000/docs
.venv/bin/python -m pytest tests
```

## Decisions

- **Chunks keep their heading path** ("Security › OAuth2 › Scopes"): it is embedded together with the text and shown as the citation.
- **Embeddings:** `BAAI/bge-small-en-v1.5` via fastembed (ONNX, CPU, 384 dims) — free, local, good enough for an English corpus; queries get the BGE search instruction, passages don't.
- **Index:** HNSW on cosine distance + a generated `tsvector` column ready for the full-text half of hybrid search.
