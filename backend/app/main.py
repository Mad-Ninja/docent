from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .chat import stream_answer
from .db import connect, init_schema
from .search import as_dicts, vector_search


@asynccontextmanager
async def lifespan(_: FastAPI):
    with connect() as conn:
        init_schema(conn)  # idempotent: adds tables introduced after the last ingest
    yield


app = FastAPI(title="Docent", description="AI support agent over documentation: RAG, tools, evals", lifespan=lifespan)


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    k: int = Field(6, ge=1, le=20)
    # agent: the model picks tools; rag: one vector search, then the answer (the baseline)
    mode: Literal["agent", "rag"] = "agent"


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/search")
def search(q: str = Query(min_length=2), k: int = Query(6, ge=1, le=20)) -> list[dict]:
    return as_dicts(vector_search(q, k))


@app.post("/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    return StreamingResponse(stream_answer(req.question, req.k, req.mode), media_type="text/event-stream")
