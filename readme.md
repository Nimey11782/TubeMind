# TubeMind

TubeMind lets users ask questions about YouTube videos using transcript-based retrieval. It combines timestamped transcript chunks, FAISS semantic search, BM25 keyword search, reciprocal-rank fusion, FlashRank reranking, and Groq answer generation.

## Architecture

- **Express API** handles registration, login, JWT authentication, chat and message storage, and streams chat responses.
- **Python RAG service** retains the existing transcript, embedding, FAISS, retrieval, reranking, and answer-generation code. It is an internal service used by Express.
- **React** provides a small chat interface with a separate chat list.
- **SQLite** stores accounts, chats, and messages. Existing accounts and messages are retained; old messages are grouped under a chat named “Previous chat”.
- **FAISS** stores a separate index for each chat. Each chat can have one video; create a new chat to use a different video.

Every chat and its messages belong to the authenticated user. Chat history, citations, and the associated video remain available after signing out and back in.

Existing SQLite accounts and messages are migrated automatically on startup. Old user-level FAISS indexes are not assigned to chats automatically; ingest the video into a chat to build its chat-specific index.

### Project layout

```text
backend/
  controllers/    HTTP request and response handling
  errors/         Shared HTTP error type
  middleware/     Authentication and error handling
  repositories/   SQLite queries
  routes/         Express route definitions
  services/       Authentication, chat rules, and Python RAG client
  db.js           SQLite connection and schema migration
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

In separate terminals from the project root, start the RAG service, API, and frontend:

```powershell
.\venv\Scripts\python.exe -m uvicorn rag_service:app --host 127.0.0.1 --port 8001
```

```powershell
Set-Location backend
npm install
npm start
```

```powershell
Set-Location frontend
npm install
npm run dev
```

Open the Vite URL shown in the frontend terminal (usually `http://localhost:5173`). The API listens on port `8000`; the RAG service listens on port `8001`. The first ingestion downloads/loads the local embedding model, and the first hybrid query may load the reranker.

## Run with Docker Compose

With a valid `.env` containing `GROQ_API_KEY` and `JWT_SECRET_KEY`:

```powershell
docker compose up --build
```

Open `http://localhost:8501`. The Express API is available on port `8000`; the Python RAG service is only available to the internal Compose network. SQLite data, FAISS indexes, and downloaded models persist in `data`, `faiss_index`, and `model_cache`.

## Retrieval evaluation

Run `python retrieval_benchmark.py` from the project root to benchmark the existing retrieval pipeline against the default index. The RAG pipeline retains the current Sentence Transformers model, FAISS, BM25, and FlashRank components.
