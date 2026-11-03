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


def as_dicts(hits: list[Hit]) -> list[dict]:
    return [asdict(h) for h in hits]
