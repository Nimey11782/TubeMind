# generation.py

from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel
from operator import itemgetter
from dotenv import load_dotenv

from retriever import retrieve_chunks
from prompt import rag_prompt

from langsmith import traceable

load_dotenv()

# ─────────────────────────────────────────
# 1. SCHEMA
# ─────────────────────────────────────────

class Citation(BaseModel):
    timestamp:str
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
    model="llama-3.3-70b-versatile",
    temperature=0
)

# ─────────────────────────────────────────
# 3. CONTEXT BUILDER
# ─────────────────────────────────────────
@traceable
def build_context(docs: list) -> str:
    if not docs:
        return "No relevant context found."
    
    formatted_chunks = []
    for idx, doc in enumerate(docs, start=1):
        start_time = round(doc.metadata["start"], 2)
        end_time = round(doc.metadata["end"], 2)

        chunk_text = f"""
[CHUNK {idx}]
Timestamp: {start_time}s - {end_time}s
Content:
{doc.page_content}
"""
        formatted_chunks.append(chunk_text)

    return "\n\n".join(formatted_chunks)

# ─────────────────────────────────────────
# 4. CITATION BUILDER
# ─────────────────────────────────────────
def seconds_to_timestamp(seconds: float):

    total_seconds = int(seconds)

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if hours > 0:
        return f"{hours:02}:{minutes:02}:{secs:02}"

    return f"{minutes:02}:{secs:02}"
@traceable
def build_citations(docs: list) -> list[Citation]:
    citations = []
    for doc in docs:
        start = round(doc.metadata["start"], 2)
        end = round(doc.metadata["end"], 2)
        video_id = doc.metadata["video_id"]

        citations.append(Citation(
            timestamp=f"{seconds_to_timestamp(start)} - {seconds_to_timestamp(end)}",
            video_id=video_id,
            youtube_link=f"https://youtube.com/watch?v={video_id}&t={int(start)}",
            snippet=doc.page_content[:150]
        ))
    return citations


# ─────────────────────────────────────────
# 5. RETRIEVE AND FORMAT
# ─────────────────────────────────────────

# global to preserve docs after chain runs
_last_retrieved_docs = []
@traceable
def retrieve_and_format(input: dict) -> dict:
    global _last_retrieved_docs
    question = input["question"]

    # single retrieval — no double fetch
    docs = retrieve_chunks(question)
    _last_retrieved_docs = docs

    context = build_context(docs)

    return {
        "context": context,
        "question": question,
        "chat_history": input["chat_history"]
    }

# ─────────────────────────────────────────
# 6. CHAIN
# ─────────────────────────────────────────

main_chain = RunnableLambda(retrieve_and_format) | rag_prompt | llm | StrOutputParser()

# ─────────────────────────────────────────
# 7. PUBLIC FUNCTION
# ─────────────────────────────────────────
@traceable
def ask(question: str, chat_history=None) -> RAGResponse:
    if chat_history is None:
        chat_history = []
    answer = main_chain.invoke({
        "question": question,
        "chat_history": chat_history
    })

    citations = build_citations(_last_retrieved_docs)

    return RAGResponse(
        answer=answer,
        citations=citations,
    )

# ─────────────────────────────────────────
# 8. TEST
# ─────────────────────────────────────────

# if __name__ == "__main__":
#     result = ask("what does he say about nuclear fusion")
    
#     print("ANSWER:")
#     print(result.answer)
    
#     print("\nCITATIONS:")
#     for c in result.citations:
#         print(f"[{c.start}s - {c.end}s] → {c.youtube_link}")
#         print(f"  {c.snippet}\n")
