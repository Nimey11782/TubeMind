from collections.abc import AsyncIterator
import json
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field, field_validator

from indexing.document_ingestion import extract_video_id
from indexing.embedding import get_chat_index_path
from indexing.index_pipeline import run_ingestion
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


app = FastAPI(title="TubeMind RAG service")


@app.get("/health")
async def health():
    return {"status": "alive"}


@app.post("/ingest")
def ingest(request: IngestRequest):
    index_path = get_chat_index_path(request.user_id, request.chat_id)
    try:
        run_ingestion(request.url, index_path=index_path)
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
    index_path = get_chat_index_path(request.user_id, request.chat_id)
    try:
        response = ask(
            request.question,
            langchain_history(request.history),
            retrieval_mode=request.retrieval_mode,
            index_path=index_path,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail="Ingest a video before chatting") from exc
    return response


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    index_path = get_chat_index_path(request.user_id, request.chat_id)
    if not (index_path / "index.faiss").is_file():
        raise HTTPException(status_code=409, detail="Ingest a video before chatting")

    async def events() -> AsyncIterator[str]:
        async for event in stream_answer(
            request.question,
            langchain_history(request.history),
            retrieval_mode=request.retrieval_mode,
            index_path=index_path,
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
