"""Load the corpus into Postgres: markdown → chunks → embeddings.

  python -m app.ingest            rebuild the whole index
"""
import time

import numpy as np

from .chunking import chunk_document
from .config import settings
from .db import connect, init_schema
from .embeddings import embed_passages


def run() -> None:
    started = time.time()
    conn = connect()
    init_schema(conn)
    conn.execute("TRUNCATE documents, chunks RESTART IDENTITY CASCADE")
    files = sorted(settings.corpus_dir.rglob("*.md"))
    total = 0
    for f in files:
        rel = f.relative_to(settings.corpus_dir).as_posix()
        chunks = chunk_document(rel, f.read_text(encoding="utf-8"), settings.chunk_chars)
        if not chunks:
            continue
        # The heading goes into the embedded text: "Security › OAuth2" says a lot about the paragraph below it.
        vectors = embed_passages([f"{c.heading}\n{c.text}" for c in chunks])
        doc_id = conn.execute("INSERT INTO documents (path, title) VALUES (%s, %s) RETURNING id",
                              (rel, chunks[0].title)).fetchone()[0]
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (document_id, ord, heading, content, embedding) VALUES (%s, %s, %s, %s, %s)",
                [(doc_id, c.ord, c.heading, c.text, np.array(v)) for c, v in zip(chunks, vectors)])
        total += len(chunks)
    print(f"indexed {len(files)} files, {total} chunks in {time.time() - started:.0f}s")


if __name__ == "__main__":
    run()
