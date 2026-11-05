from dataclasses import asdict, dataclass

import numpy as np

from .db import connect
from .embeddings import embed_query


@dataclass
class Hit:
    chunk_id: int
    path: str
    heading: str
    content: str
    score: float


def vector_search(query: str, k: int = 6) -> list[Hit]:
    """Nearest chunks by cosine distance (pgvector `<=>`); score = 1 - distance."""
    with connect() as conn:
        rows = conn.execute(
            """SELECT c.id, d.path, c.heading, c.content, 1 - (c.embedding <=> %s) AS score
               FROM chunks c JOIN documents d ON d.id = c.document_id
               ORDER BY c.embedding <=> %s LIMIT %s""",
            (np.array(embed_query(query)),) * 2 + (k,)).fetchall()
    return [Hit(*r) for r in rows]


# Natural questions rarely have every word in one chunk, so the terms are OR-ed
# (plainto_tsquery ANDs them) and ts_rank_cd puts chunks with more and closer terms first.
_OR_QUERY = "CAST(replace(plainto_tsquery('english', %s)::text, '&', '|') AS tsquery)"


def text_search(query: str, k: int = 6) -> list[Hit]:
    """Full-text search over the generated `tsv` column (stemmed words, heading included)."""
    with connect() as conn:
        rows = conn.execute(
            f"""SELECT c.id, d.path, c.heading, c.content, ts_rank_cd(c.tsv, q)::float8 AS score
                FROM chunks c JOIN documents d ON d.id = c.document_id, {_OR_QUERY} AS q
                WHERE c.tsv @@ q ORDER BY score DESC LIMIT %s""",
            (query, k)).fetchall()
    return [Hit(*r) for r in rows]


TEXT_WEIGHT = 1.0
RRF_K = 60  # the usual constant from the RRF paper: dampens the gap between rank 1 and rank 2


def hybrid_search(query: str, k: int = 6, pool: int = 30, text_weight: float = TEXT_WEIGHT) -> list[Hit]:
    """Vector + full-text, merged by Reciprocal Rank Fusion: score = Σ 1 / (RRF_K + rank).

    RRF uses only ranks, so the two scores (cosine similarity and ts_rank) never have to be
    put on one scale. A chunk found by both lists beats one found by a single list.
    """
    with connect() as conn:
        rows = conn.execute(
            f"""WITH v AS (
                    SELECT c.id, row_number() OVER (ORDER BY c.embedding <=> %(emb)s) AS r, 1.0 AS w
                    FROM chunks c ORDER BY c.embedding <=> %(emb)s LIMIT %(pool)s),
                t AS (
                    SELECT c.id, row_number() OVER (ORDER BY ts_rank_cd(c.tsv, q) DESC) AS r, %(tw)s AS w
                    FROM chunks c, {_OR_QUERY.replace('%s', '%(q)s')} AS q
                    WHERE c.tsv @@ q ORDER BY ts_rank_cd(c.tsv, q) DESC LIMIT %(pool)s),
                fused AS (
                    SELECT id, sum(w / (%(rrf)s + r))::float8 AS score
                    FROM (SELECT * FROM v UNION ALL SELECT * FROM t) both_lists GROUP BY id)
                SELECT c.id, d.path, c.heading, c.content, f.score
                FROM fused f JOIN chunks c ON c.id = f.id JOIN documents d ON d.id = c.document_id
                ORDER BY f.score DESC LIMIT %(k)s""",
            {"emb": np.array(embed_query(query)), "q": query, "pool": pool, "rrf": RRF_K, "tw": text_weight, "k": k}).fetchall()
    return [Hit(*r) for r in rows]


def read_document(path: str, limit: int = 6) -> list[Hit]:
    """The first chunks of one page in reading order; score 1.0 because the agent asked for exactly this page."""
    with connect() as conn:
        rows = conn.execute(
            """SELECT c.id, d.path, c.heading, c.content, 1.0::float8
               FROM chunks c JOIN documents d ON d.id = c.document_id
               WHERE d.path = %s ORDER BY c.ord LIMIT %s""", (path, limit)).fetchall()
    return [Hit(*r) for r in rows]


def as_dicts(hits: list[Hit]) -> list[dict]:
    return [asdict(h) for h in hits]
