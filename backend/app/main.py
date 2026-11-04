from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .chat import stream_answer
from .search import as_dicts, vector_search

app = FastAPI(title="Docent", description="AI support agent over documentation: RAG, tools, evals")


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    k: int = Field(6, ge=1, le=20)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/search")
def search(q: str = Query(min_length=2), k: int = Query(6, ge=1, le=20)) -> list[dict]:
    return as_dicts(vector_search(q, k))


@app.post("/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    return StreamingResponse(stream_answer(req.question, req.k), media_type="text/event-stream")
