# rag_chain

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel,RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from operator import itemgetter

from dotenv import load_dotenv

from retriever import retrieve_chunks
from prompt import rag_prompt

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0
)

def build_context(docs):
    if not docs:
        return "No relevant context found."
    """
    Converts retrieved Documents into a single context string
    """
    return "\n\n".join(doc.page_content for doc in docs)

parallel_chain=RunnableParallel({
    "context" : itemgetter("question") | RunnableLambda(retrieve_chunks) | RunnableLambda(build_context),
    "question": itemgetter("question"),
    "chat_history": itemgetter("chat_history")
})
#from one input compute multiple things in parallel
#one branch passes context other question
#output of parallel chain ->context 
#                         ->question
#                         ->chat_history

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0
)

main_chain=parallel_chain | rag_prompt | llm | StrOutputParser()

response=main_chain.invoke({
    "question" : "summarize the video",
    "chat_history" : []
    })
print(response)
