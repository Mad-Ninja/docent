from pathlib import Path

from pydantic_settings import BaseSettings

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    database_url: str = "postgresql://kb:kb@localhost:5435/kb"
    corpus_dir: Path = ROOT / "corpus" / "fastapi"
    # Small English model, 384 dimensions, runs on CPU through ONNX (no torch).
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384
    chunk_chars: int = 1200


settings = Settings()
