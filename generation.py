import os
from collections.abc import AsyncIterator

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from indexing.embedding import DEFAULT_INDEX_PATH
from retriever import retrieve_chunks
from prompt import rag_prompt

from langsmith import traceable
from tracing_config import TRACING_ENABLED

# ─────────────────────────────────────────
# 1. SCHEMA
# ─────────────────────────────────────────

class Citation(BaseModel):
    source_number: int
    timestamp: str
    video_id: str
    youtube_link: str
    snippet: str

class RAGResponse(BaseModel):
    answer: str
    citations: list[Citation]

# ─────────────────────────────────────────
# 2. LLM
# ─────────────────────────────────────────

llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
    temperature=0,
    max_tokens=1024,
)

# ─────────────────────────────────────────
# 3. CONTEXT BUILDER
# ─────────────────────────────────────────
@traceable(enabled=TRACING_ENABLED)
def build_context(docs: list[Document]) -> str:
    if not docs:
        return "No relevant context found."
    
    formatted_chunks = []
    for idx, doc in enumerate(docs, start=1):
        start_time = round(doc.metadata["start"], 2)
        end_time = round(doc.metadata["end"], 2)

        chunk_text = f"""
[SOURCE {idx}] Transcript ({start_time}s - {end_time}s):
{doc.page_content}
"""
        formatted_chunks.append(chunk_text)

    return "\n\n".join(formatted_chunks)

# ─────────────────────────────────────────
# 4. CITATION BUILDER
# ─────────────────────────────────────────
def seconds_to_timestamp(seconds: float) -> str:

    total_seconds = int(seconds)

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if hours > 0:
        return f"{hours:02}:{minutes:02}:{secs:02}"

    return f"{minutes:02}:{secs:02}"
@traceable(enabled=TRACING_ENABLED)
def build_citations(docs: list[Document]) -> list[Citation]:
    citations = []
    for source_number, doc in enumerate(docs, start=1):
        start = round(doc.metadata["start"], 2)
        end = round(doc.metadata["end"], 2)
        video_id = doc.metadata["video_id"]

        citations.append(Citation(
            source_number=source_number,
            timestamp=f"{seconds_to_timestamp(start)} - {seconds_to_timestamp(end)}",
            video_id=video_id,
            youtube_link=f"https://youtube.com/watch?v={video_id}&t={int(start)}",
            snippet=doc.page_content[:150]
        ))
    return citations


# ─────────────────────────────────────────
# 5. RETRIEVE AND FORMAT
# ─────────────────────────────────────────

@traceable(enabled=TRACING_ENABLED)
def get_context_and_docs(input_dict: dict) -> tuple[str, list[Document]]:
    question = input_dict["question"]
    retrieval_mode = input_dict.get("retrieval_mode", "hybrid")
    question_for_retrieval = question
    retrieval_k = 4 if retrieval_mode == "similarity" else 10

    if input_dict.get("chat_history"):
        last_user_question = ""
        for msg in reversed(input_dict["chat_history"]):
            if isinstance(msg, HumanMessage):
                last_user_question = msg.content
                break
        question_for_retrieval = last_user_question + " " + question

    docs = retrieve_chunks(
        question_for_retrieval,
        mode=retrieval_mode,
        k=retrieval_k,
        index_path=input_dict.get("index_path", DEFAULT_INDEX_PATH),
    )
    
    context = build_context(docs)
    return context, docs

# ─────────────────────────────────────────
# 6. CHAIN
# ─────────────────────────────────────────

main_chain = rag_prompt | llm | StrOutputParser()

# ─────────────────────────────────────────
# 7. PUBLIC FUNCTION
# ─────────────────────────────────────────
@traceable(enabled=TRACING_ENABLED)
def ask(
    question: str,
    chat_history=None,
    retrieval_mode="hybrid",
    index_path=DEFAULT_INDEX_PATH,
) -> RAGResponse:
    if chat_history is None:
        chat_history = []
        
    input_dict = {
        "question": question,
        "chat_history": chat_history,
        "retrieval_mode": retrieval_mode,
        "index_path": index_path,
    }
    
    context, docs = get_context_and_docs(input_dict)
    citations = build_citations(docs)
    if not docs:
        return RAGResponse(
            answer="This wasn't covered in the video.",
            citations=[],
        )
    
    answer = main_chain.invoke({
        "question": question,
        "chat_history": chat_history,
        "context": context
    })

    return RAGResponse(
        answer=answer,
        citations=citations,
    )


async def stream_answer(
    question: str,
    chat_history=None,
    retrieval_mode="hybrid",
    index_path=DEFAULT_INDEX_PATH,
) -> AsyncIterator[dict]:
    if chat_history is None:
        chat_history = []

    context, docs = await run_in_threadpool(
        get_context_and_docs,
        {
            "question": question,
            "chat_history": chat_history,
            "retrieval_mode": retrieval_mode,
            "index_path": index_path,
        },
    )
    if not docs:
        yield {"type": "token", "content": "This wasn't covered in the video."}
        yield {"type": "done", "citations": []}
        return

    chain_input = {
        "question": question,
        "chat_history": chat_history,
        "context": context,
    }
    async for token in main_chain.astream(chain_input):
        if token:
            yield {"type": "token", "content": token}

    yield {
        "type": "done",
        "citations": [
            citation.model_dump()
            for citation in build_citations(docs)
        ],
    }
