# YouTube RAG Chatbot

A Retrieval-Augmented Generation (RAG) system that enables users to chat with YouTube videos using transcript-based semantic search. The system retrieves relevant transcript segments, reranks them for relevance, and generates source-grounded answers with timestamp citations.

## Features

* YouTube transcript ingestion
* Window-based chunking with overlap
* Semantic search using FAISS
* MMR (Max Marginal Relevance) retrieval
* CrossEncoder reranking
* Timestamp-grounded citations
* FastAPI backend
* LangSmith tracing and observability
* Benchmark query evaluation

---

## Architecture

```text
YouTube Video
      │
      ▼
Transcript Extraction
      │
      ▼
Window Chunking
      │
      ▼
MiniLM Embeddings
      │
      ▼
FAISS Vector Store
      │
      ▼
MMR Retrieval
      │
      ▼
CrossEncoder Reranking
      │
      ▼
Llama-3.3-70B (Groq)
      │
      ▼
Answer + Citations
```

---

## Tech Stack

### Retrieval

* LangChain
* FAISS
* Sentence Transformers
* MMR Retrieval
* CrossEncoder Reranking

### LLM

* Groq
* Llama-3.3-70B-Versatile

### Backend

* FastAPI
* Pydantic

### Monitoring

* LangSmith

---

## API Endpoints

### Ingest a Video

```http
POST /ingest
```

Request:

```json
{
  "url": "https://youtube.com/watch?v=..."
}
```

### Ask Questions

```http
POST /chat
```

Request:

```json
{
  "question": "What are AI concerns?",
  "chat_history": []
}
```

### Health Check

```http
GET /health
```

---

## Example Questions

* What are AI concerns?
* What does the speaker say about open-source AI?
* Summarize the video.
* What future does the speaker envision for AI?
* What benefits and risks of AI are discussed?

---

## Key Improvements

Compared to the initial version, the system was upgraded with:

* Window-based chunking for better context preservation
* MMR retrieval for more diverse results
* CrossEncoder reranking for improved precision
* FastAPI deployment
* LangSmith tracing for debugging and monitoring
* Timestamp-grounded source citations

## Retrieval Evaluation

A qualitative benchmark was conducted to compare different retrieval strategies on representative queries.

### Observations

- CrossEncoder reranking improved retrieval precision for entity-focused questions such as questions about Meta and Mark Zuckerberg.
- MMR retrieval produced more diverse context and performed better on broad summarization-style queries.
- Similarity search often retrieved semantically related content but was less effective at prioritizing the most relevant passages.
- CrossEncoder reranking introduced approximately 0.5 seconds of additional retrieval latency compared to similarity search and MMR retrieval.

### Example Findings

| Query Type | Best Performing Method |
|------------|-----------------------|
| Entity-specific questions | MMR + CrossEncoder |
| Broad summarization | MMR |
| General factual retrieval | MMR / CrossEncoder |

### Latency Comparison

| Retrieval Method | Average Latency |
|------------------|-----------------|
| Similarity Search | ~0.02s |
| MMR Retrieval | ~0.03s |
| MMR + CrossEncoder Reranking | ~0.50s |

### Key Insight

The evaluation revealed a tradeoff between retrieval quality and latency. MMR retrieval provided broader contextual coverage for summarization tasks, while CrossEncoder reranking improved precision for highly specific queries at the cost of additional retrieval time.
---

## Future Improvements

* Multi-video retrieval and reasoning
* Hybrid retrieval (BM25 + Dense Retrieval)
* Conversational memory
* Docker deployment
* Multimodal RAG using video frames and transcripts
* Automated evaluation with RAGAS

---

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API server:

```bash
uvicorn main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

to access the FastAPI Swagger UI.
