from document_ingestion import ingest_youtube_transcript
from text_splitting import split_documents
from embedding import store_embeddings


if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=Gfr50f6ZBvo"

    print("📥 Ingesting transcript...")
    docs = ingest_youtube_transcript(url)

    print("✂️ Splitting documents...")
    split_docs = split_documents(docs)

    print("🧠 Creating embeddings...")
    store_embeddings(split_docs)

    print("✅ FAISS index created successfully")