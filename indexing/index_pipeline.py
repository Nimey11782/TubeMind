from indexing.document_ingestion import ingest_youtube_transcript
from indexing.window_chunking import create_windowed_chunks
from indexing.embedding import store_embeddings


def run_ingestion(url: str):

    print("📥 Ingesting transcript...")
    docs = ingest_youtube_transcript(url)

    print("✂️ Splitting documents...")
    windowed_docs = create_windowed_chunks(docs)

    print("🧠 Creating embeddings...")
    store_embeddings(windowed_docs)

    print("✅ FAISS index created successfully")
if __name__ == "__main__":
    run_ingestion("https://www.youtube.com/watch?v=Gfr50f6ZBvo")