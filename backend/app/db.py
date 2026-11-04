import psycopg
from pgvector.psycopg import register_vector

from .config import settings

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id    serial PRIMARY KEY,
    path  text UNIQUE NOT NULL,
    title text NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id          serial PRIMARY KEY,
    document_id int NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ord         int NOT NULL,
    heading     text NOT NULL,
    content     text NOT NULL,
    embedding   vector({settings.embed_dim}) NOT NULL,
    -- full-text side of the hybrid search; the heading counts too
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', heading || ' ' || content)) STORED,
    UNIQUE (document_id, ord)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);

-- one row per answered question: what the agent did, for debugging and evals
CREATE TABLE IF NOT EXISTS traces (
    id         serial PRIMARY KEY,
    created_at timestamptz NOT NULL DEFAULT now(),
    question   text NOT NULL,
    planner    text NOT NULL,
    outcome    text NOT NULL,
    steps      jsonb NOT NULL,
    sources    jsonb NOT NULL,
    total_ms   int NOT NULL
);

-- questions the agent passed to a person
CREATE TABLE IF NOT EXISTS tickets (
    id         serial PRIMARY KEY,
    created_at timestamptz NOT NULL DEFAULT now(),
    question   text NOT NULL,
    reason     text NOT NULL,
    status     text NOT NULL DEFAULT 'open'
);
"""


def connect() -> psycopg.Connection:
    conn = psycopg.connect(settings.database_url, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def init_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA)
