from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import json
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from indexing.document_ingestion import extract_video_id
from indexing.index_pipeline import run_ingestion
from indexing.vector_store import has_chat_vectors, verify_vector_database
from generation import ask, stream_answer


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class IngestRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)
    user_id: int = Field(gt=0)
    chat_id: int = Field(gt=0)

    @field_validator("url")
    @classmethod
    def validate_youtube_url(cls, url: str) -> str:
        extract_video_id(url)
        return url.strip()


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    retrieval_mode: Literal["similarity", "mmr", "rerank", "hybrid"] = "hybrid"
    user_id: int = Field(gt=0)
    chat_id: int = Field(gt=0)
    history: list[ChatMessage] = Field(default_factory=list, max_length=8)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, question: str) -> str:
        if not question.strip():
            raise ValueError("Question must not be blank")
        return question.strip()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await run_in_threadpool(verify_vector_database)
    yield


app = FastAPI(title="TubeMind RAG service", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "alive"}


@app.post("/ingest")
def ingest(request: IngestRequest):
    try:
        run_ingestion(request.url, request.user_id, request.chat_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"status": "ok", "video_id": extract_video_id(request.url)}


def langchain_history(history: list[ChatMessage]):
    return [
        HumanMessage(content=message.content)
        if message.role == "user"
        else AIMessage(content=message.content)
        for message in history
    ]


@app.post("/chat")
def chat(request: ChatRequest):
    if not has_chat_vectors(request.user_id, request.chat_id):
        raise HTTPException(status_code=409, detail="Ingest a video before chatting")
    return ask(
        request.question,
        langchain_history(request.history),
        retrieval_mode=request.retrieval_mode,
        user_id=request.user_id,
        chat_id=request.chat_id,
    )


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    if not has_chat_vectors(request.user_id, request.chat_id):
        raise HTTPException(status_code=409, detail="Ingest a video before chatting")

    async def events() -> AsyncIterator[str]:
        async for event in stream_answer(
            request.question,
            langchain_history(request.history),
            retrieval_mode=request.retrieval_mode,
            user_id=request.user_id,
            chat_id=request.chat_id,
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
