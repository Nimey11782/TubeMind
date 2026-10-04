# main.py
import json
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from database.auth_routes import router as auth_router
from database.chat_store import get_chat_history, save_message
from database.dependencies import get_current_user
from database.history import load_chat_history
from database.schema import init_db
from generation import RAGResponse, ask, stream_answer
from indexing.document_ingestion import extract_video_id
from indexing.embedding import get_user_index_path
from indexing.index_pipeline import run_ingestion

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth_router)

class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    retrieval_mode: Literal["similarity", "mmr", "rerank", "hybrid"] = "hybrid"

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, question: str) -> str:
        if not question.strip():
            raise ValueError("Question must not be blank")
        return question.strip()

class IngestRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)

    @field_validator("url")
    @classmethod
    def url_must_identify_youtube_video(cls, url: str) -> str:
        extract_video_id(url)
        return url.strip()

@app.get("/history")
def history(
    current_user=Depends(get_current_user)
):

    rows = get_chat_history(
        current_user["user_id"],
        limit=50
    )

    return rows

@app.post("/chat", response_model=RAGResponse)
def chat(request: QueryRequest, current_user=Depends(get_current_user)):

    history = load_chat_history(
        current_user["user_id"]
    )

    index_path = get_user_index_path(current_user["user_id"])
    try:
        response = ask(
            request.question,
            history,
            retrieval_mode=request.retrieval_mode,
            index_path=index_path,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=409,
            detail="Ingest a video before asking a question.",
        ) from exc

    save_message(
        current_user["user_id"],
        "user",
        request.question
    )

    save_message(
        current_user["user_id"],
        "assistant",
        response.answer
    )

    return response


@app.post("/chat/stream")
def chat_stream(request: QueryRequest, current_user=Depends(get_current_user)):
    user_id = current_user["user_id"]
    index_path = get_user_index_path(user_id)
    if not (index_path / "index.faiss").is_file():
        raise HTTPException(
            status_code=409,
            detail="Ingest a video before asking a question.",
        )

    history = load_chat_history(user_id)

    async def events():
        answer_parts = []
        async for event in stream_answer(
            request.question,
            history,
            retrieval_mode=request.retrieval_mode,
            index_path=index_path,
        ):
            if event["type"] == "token":
                answer_parts.append(event["content"])
            elif event["type"] == "done":
                answer = "".join(answer_parts)
                await run_in_threadpool(
                    save_message,
                    user_id,
                    "user",
                    request.question,
                )
                await run_in_threadpool(
                    save_message,
                    user_id,
                    "assistant",
                    answer,
                )
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/ingest")
def ingest(request: IngestRequest, current_user=Depends(get_current_user)):
    index_path = get_user_index_path(current_user["user_id"])
    run_ingestion(request.url, index_path=index_path)
    return {"status": "ok"}

@app.get("/health")
async def health():
    return {"status": "alive"}