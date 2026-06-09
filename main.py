# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from generation import ask, RAGResponse
from indexing.index_pipeline import run_ingestion

app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    chat_history: list = []

class IngestRequest(BaseModel):
    url: str

@app.post("/chat", response_model=RAGResponse)
async def chat(request: QueryRequest):
    return ask(request.question, request.chat_history)

@app.post("/ingest")
async def ingest(request: IngestRequest):
    run_ingestion(request.url)
    return {"status": "ok"}

@app.get("/health")
async def health():
    return {"status": "alive"}