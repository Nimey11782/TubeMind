import os
import re
from hashlib import sha256
from pathlib import Path
from threading import Lock

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langsmith import traceable
from rank_bm25 import BM25Okapi

from indexing.embedding import (
    DEFAULT_INDEX_PATH,
    MODEL_CACHE_PATH,
    get_embeddings,
)
from tracing_config import TRACING_ENABLED

_vectorstores: dict[Path, FAISS] = {}
_vectorstore_lock = Lock()
_bm25_indexes: dict[Path, tuple[int, BM25Okapi, list[Document]]] = {}
_bm25_lock = Lock()
_reranker = None
_reranker_lock = Lock()
_BM25_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
_RRF_CONSTANT = 60
_HYBRID_CANDIDATE_COUNT = 20


def _resolve_index_path(index_path: str | Path) -> Path:
    return Path(index_path).resolve()


def get_vectorstore(index_path: str | Path = DEFAULT_INDEX_PATH) -> FAISS:
    resolved_path = _resolve_index_path(index_path)
    vectorstore = _vectorstores.get(resolved_path)
    if vectorstore is None:
        with _vectorstore_lock:
            vectorstore = _vectorstores.get(resolved_path)
            if vectorstore is None:
                if not (resolved_path / "index.faiss").is_file():
                    raise FileNotFoundError(
                        f"FAISS index not found at {resolved_path}. Ingest a video first."
                    )

                vectorstore = FAISS.load_local(
                    str(resolved_path),
                    get_embeddings(),
                    allow_dangerous_deserialization=True,
                )
                _vectorstores[resolved_path] = vectorstore

    return vectorstore


def set_vectorstore(
    vectorstore: FAISS,
    index_path: str | Path = DEFAULT_INDEX_PATH,
) -> None:
    resolved_path = _resolve_index_path(index_path)
    with _vectorstore_lock:
        _vectorstores[resolved_path] = vectorstore
    with _bm25_lock:
        _bm25_indexes.pop(resolved_path, None)


def _get_reranker():
    global _reranker

    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                import onnxruntime as ort
                from flashrank import Ranker
                from flashrank.Config import model_file_map

                model_name = "ms-marco-MiniLM-L-12-v2"
                reranker = Ranker(
                    model_name=model_name,
                    cache_dir=str(MODEL_CACHE_PATH / "flashrank"),
                )
                session_options = ort.SessionOptions()
                session_options.intra_op_num_threads = min(
                    2,
                    max(1, os.cpu_count() or 1),
                )
                session_options.inter_op_num_threads = 1
                session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

                model_path = reranker.model_dir / model_file_map[model_name]
                del reranker.session
                reranker.session = ort.InferenceSession(
                    str(model_path),
                    sess_options=session_options,
                )
                _reranker = reranker
    return _reranker


@traceable(enabled=TRACING_ENABLED)
def rerank_documents(
    query: str,
    docs: list[Document],
    top_n: int = 4,
) -> list[Document]:
    if not docs or top_n < 1:
        return []

    from flashrank import RerankRequest

    passages = [
        {
            "id": _document_id(doc),
            "text": doc.page_content,
            "meta": doc.metadata,
        }
        for doc in docs
    ]
    ranked = _get_reranker().rerank(
        RerankRequest(query=query, passages=passages)
    )
    documents_by_id = {_document_id(doc): doc for doc in docs}
    return [
        documents_by_id[result["id"]]
        for result in ranked[:top_n]
        if result["id"] in documents_by_id
    ]


def _document_id(doc: Document) -> str:
    chunk_id = doc.metadata.get("chunk_id")
    if chunk_id:
        return str(chunk_id)
    return sha256(doc.page_content.encode("utf-8")).hexdigest()


def reciprocal_rank_fusion(
    dense_results: list[Document],
    sparse_results: list[Document],
    top_k: int = 20,
) -> list[Document]:
    scores = {}
    documents = {}

    for results in (dense_results, sparse_results):
        for rank, doc in enumerate(results, start=1):
            document_id = _document_id(doc)
            scores[document_id] = (
                scores.get(document_id, 0.0) + 1 / (_RRF_CONSTANT + rank)
            )
            documents.setdefault(document_id, doc)

    ranked_ids = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
    return [documents[document_id] for document_id in ranked_ids]


def bm25_search(
    query: str,
    vectorstore,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    top_k: int = 20,
) -> list[Document]:
    if not query.strip() or top_k < 1:
        return []

    resolved_path = _resolve_index_path(index_path)
    cached_index = _bm25_indexes.get(resolved_path)
    if cached_index is None or cached_index[0] != id(vectorstore):
        with _bm25_lock:
            cached_index = _bm25_indexes.get(resolved_path)
            if cached_index is None or cached_index[0] != id(vectorstore):
                documents = []
                for docstore_id in vectorstore.index_to_docstore_id.values():
                    doc = vectorstore.docstore.search(docstore_id)
                    if isinstance(doc, Document) and doc.page_content.strip():
                        documents.append(doc)

                if not documents:
                    _bm25_indexes.pop(resolved_path, None)
                    return []

                tokenized_corpus = [
                    _BM25_TOKEN_PATTERN.findall(doc.page_content.casefold())
                    for doc in documents
                ]
                if not any(tokenized_corpus):
                    return []

                cached_index = (
                    id(vectorstore),
                    BM25Okapi(tokenized_corpus),
                    documents,
                )
                _bm25_indexes[resolved_path] = cached_index

    tokens = _BM25_TOKEN_PATTERN.findall(query.casefold())
    if not tokens:
        return []

    scores = cached_index[1].get_scores(tokens)
    ranked = sorted(
        (
            (index, float(score))
            for index, score in enumerate(scores)
            if score > 0
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    return [cached_index[2][index] for index, _ in ranked[:top_k]]


@traceable(enabled=TRACING_ENABLED)
def retrieve_chunks(
    query: str,
    mode: str = "hybrid",
    k: int = 10,
    index_path: str | Path = DEFAULT_INDEX_PATH,
) -> list[Document]:
    if mode not in {"similarity", "mmr", "rerank", "hybrid"}:
        raise ValueError(f"Unknown retrieval mode: {mode}")
    if k < 1:
        raise ValueError("k must be greater than zero")
    if not query.strip():
        raise ValueError("query must not be empty")

    vectorstore = get_vectorstore(index_path)

    if mode == "similarity":
        return vectorstore.similarity_search(query, k=min(k, 4))

    if mode == "mmr":
        return vectorstore.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=max(k * 2, 20),
            lambda_mult=0.6,
        )

    if mode == "rerank":
        candidates = vectorstore.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=max(k * 2, 20),
            lambda_mult=0.6,
        )
        return rerank_documents(query, candidates, top_n=min(k, 4))

    candidate_count = max(k, _HYBRID_CANDIDATE_COUNT)
    dense_results = vectorstore.similarity_search(query, k=candidate_count)
    sparse_results = bm25_search(
        query,
        vectorstore,
        index_path=index_path,
        top_k=candidate_count,
    )
    fused_results = reciprocal_rank_fusion(
        dense_results,
        sparse_results,
        top_k=candidate_count,
    )
    return rerank_documents(query, fused_results, top_n=min(k, 4))
