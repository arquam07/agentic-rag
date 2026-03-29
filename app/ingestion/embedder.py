from functools import lru_cache
from sentence_transformers import SentenceTransformer
import os

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    """Load once, reuse across requests."""
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_chunks(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of strings.
    Returns a list of float vectors (one per text).
    """
    model = _load_model()
    vectors = model.encode(texts, batch_size=32, show_progress_bar=False)
    return vectors.tolist()