from _thread import LockType
from pathlib import Path
from threading import Lock
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX_PATH = PROJECT_ROOT / "faiss_index"
MODEL_CACHE_PATH = PROJECT_ROOT / "model_cache"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_embeddings_lock = Lock()
_index_locks: dict[Path, LockType] = {}
_index_locks_lock = Lock()


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


def get_user_index_path(user_id: int) -> Path:
    if user_id < 1:
        raise ValueError("user_id must be a positive integer")
    return DEFAULT_INDEX_PATH / f"user_{user_id}"


def _get_index_lock(index_path: Path):
    with _index_locks_lock:
        return _index_locks.setdefault(index_path, Lock())

def get_embeddings() -> LocalSentenceTransformerEmbeddings:
    global _embeddings

    if _embeddings is None:
        with _embeddings_lock:
            if _embeddings is None:
                _embeddings = LocalSentenceTransformerEmbeddings()
    return _embeddings


def store_embeddings(
    docs: list[Document],
    index_path: str | Path = DEFAULT_INDEX_PATH,
) -> FAISS:
    if not docs:
        raise ValueError("Cannot create a vector index without transcript chunks")

    index_path = Path(index_path).resolve()
    embeddings = get_embeddings()
    index_path.mkdir(parents=True, exist_ok=True)
    document_ids = [
        doc.metadata.get("chunk_id") or uuid4().hex
        for doc in docs
    ]

    with _get_index_lock(index_path):
        if (index_path / "index.faiss").is_file():
            vectorstore = FAISS.load_local(
                str(index_path),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            video_ids = {
                doc.metadata.get("video_id")
                for doc in docs
                if doc.metadata.get("video_id")
            }
            existing_ids = []
            for docstore_id in vectorstore.index_to_docstore_id.values():
                stored_doc = vectorstore.docstore.search(docstore_id)
                if (
                    isinstance(stored_doc, Document)
                    and stored_doc.metadata.get("video_id") in video_ids
                ):
                    existing_ids.append(docstore_id)

            if existing_ids:
                vectorstore.delete(ids=existing_ids)
            vectorstore.add_documents(docs, ids=document_ids)
        else:
            vectorstore = FAISS.from_documents(
                documents=docs,
                embedding=embeddings,
                ids=document_ids,
            )

        vectorstore.save_local(str(index_path))
    return vectorstore