from functools import lru_cache

from fastembed import TextEmbedding

from .config import settings

# BGE models expect this instruction in front of queries (not in front of passages).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def model() -> TextEmbedding:
    return TextEmbedding(settings.embed_model)


def embed_passages(texts: list[str]) -> list[list[float]]:
    return [v.tolist() for v in model().embed(texts, batch_size=32)]


def embed_query(text: str) -> list[float]:
    return next(iter(model().embed([QUERY_PREFIX + text]))).tolist()
