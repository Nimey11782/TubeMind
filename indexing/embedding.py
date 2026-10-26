import os
from pathlib import Path
from threading import Lock

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_CACHE_PATH = Path(os.getenv("MODEL_CACHE_PATH", PROJECT_ROOT / "model_cache"))
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384

_embeddings_lock = Lock()


class LocalSentenceTransformerEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            cache_folder=str(MODEL_CACHE_PATH),
            device="cpu",
            token=False,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.tolist()


_embeddings: LocalSentenceTransformerEmbeddings | None = None


def get_embeddings() -> LocalSentenceTransformerEmbeddings:
    global _embeddings

    if _embeddings is None:
        with _embeddings_lock:
            if _embeddings is None:
                _embeddings = LocalSentenceTransformerEmbeddings()
    return _embeddings
