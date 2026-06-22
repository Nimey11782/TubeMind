# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from generation import ask, RAGResponse
from indexing.index_pipeline import run_ingestion
from database.auth_routes import router as auth_router
from fastapi import Depends
from database.dependencies import get_current_user
from database.history import load_chat_history
from database.chat_store import save_message
import database.schema

app = FastAPI()
app.include_router(auth_router)

class QueryRequest(BaseModel):
    question: str

class IngestRequest(BaseModel):
    url: str

from langchain_core.messages import (HumanMessage,AIMessage)
from database.chat_store import (save_message,get_chat_history)
@app.get("/history")
async def history(
    current_user=Depends(get_current_user)
):

    rows = get_chat_history(
        current_user["user_id"],
        limit=50
    )

    return rows
def load_chat_history(user_id):

    rows = get_chat_history(user_id)

    history = []

    for row in rows:

        if row["role"] == "user":
            history.append(
                HumanMessage(content=row["content"])
            )

        else:
            history.append(
                AIMessage(content=row["content"])
            )

    return history

@app.post("/chat", response_model=RAGResponse)
async def chat(request: QueryRequest,current_user=Depends(get_current_user)):

    history = load_chat_history(
        current_user["user_id"]
    )

    response = ask(
        request.question,
        history
    )

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

@app.post("/ingest")
async def ingest(request: IngestRequest,current_user=Depends(get_current_user)):
    run_ingestion(request.url)
    return {"status": "ok"}

@app.get("/health")
async def health():
    return {"status": "alive"}