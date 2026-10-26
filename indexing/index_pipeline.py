import logging

from indexing.document_ingestion import ingest_youtube_transcript
from indexing.window_chunking import create_windowed_chunks
from indexing.vector_store import store_embeddings
from retriever import clear_chat_cache

logger = logging.getLogger(__name__)


def run_ingestion(url: str, user_id: int, chat_id: int):
    logger.info("Fetching YouTube transcript")
    docs = ingest_youtube_transcript(url)

    windowed_docs = create_windowed_chunks(docs)
    if not windowed_docs:
        raise ValueError("The video transcript did not contain any usable captions")

    logger.info("Embedding %d transcript chunks", len(windowed_docs))
    store_embeddings(windowed_docs, user_id, chat_id)
    clear_chat_cache(user_id, chat_id)

    logger.info("Transcript vectors stored successfully")
if __name__ == "__main__":
    run_ingestion("https://www.youtube.com/watch?v=Gfr50f6ZBvo", 1, 1)