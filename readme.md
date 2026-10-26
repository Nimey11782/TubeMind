# TubeMind

TubeMind lets users ask questions about YouTube videos using transcript-based retrieval. It combines timestamped transcript chunks, pgvector semantic search, BM25 keyword search, reciprocal-rank fusion, FlashRank reranking, and Groq answer generation.

## Architecture

- **Express API** handles registration, login, JWT authentication, chat and message storage, and streams chat responses.
- **Python RAG service** handles transcript ingestion, embeddings, retrieval, reranking, and answer generation. It is an internal service used by Express.
- **React** provides a small chat interface with a separate chat list.
- **PostgreSQL + pgvector** store accounts, chats, messages, and transcript embeddings. An HNSW cosine-distance index supports vector search; each search is filtered to the authenticated user's chat.
- **BM25 and FlashRank** continue to provide keyword retrieval, result fusion, and reranking alongside semantic vector search.

Every chat and its messages belong to the authenticated user. Chat history, citations, and the associated video remain available after signing out and back in.

The PostgreSQL database starts with a fresh schema and does not import previous records. Each new chat's transcript chunks and vectors are stored in PostgreSQL when a video is ingested.

### Project layout

```text
backend/
  controllers/    HTTP request and response handling
  errors/         Shared HTTP error type
  middleware/     Authentication and error handling
  repositories/   PostgreSQL queries
  routes/         Express route definitions
  services/       Authentication, chat rules, and Python RAG client
  db.js           PostgreSQL connection and schema initialization
  app.js          Express dependency wiring and middleware
  server.js       Environment setup and server lifecycle

frontend/src/
  components/     Auth form, chat sidebar, messages, and chat panel
  services/       API, authentication, and chat requests
  App.jsx         Screen composition and UI action handlers
  main.jsx        React entry point
```

The Express layers use direct module-level functions rather than service/controller factories, so each request path can be followed from route to controller, service, and repository.

## Features

- Register and log in with JWT authentication
- Create, select, and delete separate chats
- Add one YouTube video to each chat
- Keep each chat's transcript index and message history separate
- Stream answers with timestamp-linked citations
- Similarity, MMR, rerank, and hybrid retrieval modes

## API

All routes except registration, login, and health require `Authorization: Bearer <access_token>`.

| Method | Route | Purpose |
| --- | --- | --- |
| `POST` | `/register` | Create an account |
| `POST` | `/login` | Get an access token |
| `GET` | `/health` | API health check |
| `GET` | `/chats` | List the current user's chats |
| `POST` | `/chats` | Create an empty chat |
| `GET` | `/chats/:chatId/messages` | Load a chat and its messages |
| `DELETE` | `/chats/:chatId` | Delete a chat and its messages |
| `POST` | `/chats/:chatId/ingest` | Add that chat's video (`{"url":"..."}`) |
| `POST` | `/chats/:chatId/chat/stream` | Stream an answer (`{"question":"..."}`) |

The streaming endpoint sends server-sent events: `token` events contain answer text, and the final `done` event contains citations. The optional `retrieval_mode` accepts `hybrid` (default), `similarity`, `mmr`, or `rerank`.

## Run locally (PowerShell)

Create and configure `.env` in the project root:

```env
GROQ_API_KEY=your_key
JWT_SECRET_KEY=generate_a_long_random_value
GROQ_MODEL=openai/gpt-oss-120b
POSTGRES_PASSWORD=choose_a_url_safe_password
DATABASE_URL=postgresql://tubemind:choose_a_url_safe_password@127.0.0.1:5433/tubemind

# Optional tracing
ENABLE_LANGSMITH_TRACING=false
LANGCHAIN_API_KEY=your_key
LANGCHAIN_PROJECT=youtube-rag
```

Install Python dependencies:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

In separate terminals, start the API first (it initializes the PostgreSQL schema), then the RAG service, and then the frontend:

```powershell
Set-Location backend
npm install
npm start
```

```powershell
.\venv\Scripts\python.exe -m uvicorn rag_service:app --host 127.0.0.1 --port 8001
```

```powershell
Set-Location frontend
npm install
npm run dev
```

Start PostgreSQL with pgvector installed and create an empty `tubemind` database before running the services. Open the Vite URL shown in the frontend terminal (usually `http://localhost:5173`). The API listens on port `8000`; the RAG service listens on port `8001`. The first ingestion downloads/loads the local embedding model, and the first hybrid query may load the reranker.

## Run with Docker Compose

With a `.env` containing a URL-safe `POSTGRES_PASSWORD`, `GROQ_API_KEY`, and `JWT_SECRET_KEY`:

```powershell
docker compose up --build
```

Open `http://localhost:8501`. The Express API is available on port `8000`; the Python RAG service is only available to the internal Compose network. PostgreSQL is available to host tools such as pgAdmin on port `5433` (port `5432` inside the Compose network); PostgreSQL data and downloaded models persist in `postgres_data` and `model_cache`.

PostgreSQL data persists in the `postgres_data` Docker volume. To run the API and RAG service outside Docker, set `DATABASE_URL` to a PostgreSQL database with the pgvector extension installed.

## Retrieval evaluation

Run `python retrieval_benchmark.py` from the project root to benchmark retrieval for user `1`, chat `1`. The RAG pipeline retains the current Sentence Transformers model, BM25, and FlashRank components, with pgvector providing semantic vector search.
