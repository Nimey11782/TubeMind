import logging

from indexing.document_ingestion import ingest_youtube_transcript
from indexing.window_chunking import create_windowed_chunks
from indexing.embedding import DEFAULT_INDEX_PATH, store_embeddings
from retriever import set_vectorstore

logger = logging.getLogger(__name__)


def run_ingestion(url: str, index_path=DEFAULT_INDEX_PATH):
    logger.info("Fetching YouTube transcript")
    docs = ingest_youtube_transcript(url)

    windowed_docs = create_windowed_chunks(docs)
    if not windowed_docs:
        raise ValueError("The video transcript did not contain any usable captions")

    logger.info("Embedding %d transcript chunks", len(windowed_docs))
    vectorstore = store_embeddings(windowed_docs, index_path)
    set_vectorstore(vectorstore, index_path)

    logger.info("FAISS index created successfully")
    return vectorstore
if __name__ == "__main__":
    run_ingestion("https://www.youtube.com/watch?v=Gfr50f6ZBvo")