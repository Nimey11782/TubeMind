from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langsmith import traceable

from dotenv import load_dotenv

load_dotenv()

#GLOBAL objects (loaded once)
_embeddings = None
_vectorstore = None


def get_vectorstore(index_path="faiss_index"):
    global _embeddings, _vectorstore

    if _vectorstore is None:
        print("🔁 Loading embeddings + FAISS (once)...")

        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder="./model_cache"
        )

        _vectorstore = FAISS.load_local(
            index_path,
            _embeddings,
            allow_dangerous_deserialization=True
        )

    return _vectorstore

_reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
@traceable
def rerank_documents(query, docs, top_n=4):

    # Create query-document pairs
    pairs = [
        (query, doc.page_content)
        for doc in docs
    ]

    # Predict relevance scores
    scores = _reranker.predict(pairs)

    # Combine docs + scores
    scored_docs = list(zip(docs, scores))

    # Sort descending by score
    scored_docs.sort(
        key=lambda x: x[1],
        reverse=True
    )

    # Keep only top_n docs
    reranked_docs = [
        doc for doc, score in scored_docs[:top_n]
    ]

    return reranked_docs

@traceable
def retrieve_chunks(query: str,mode: str = "rerank",k: int=10):
    vector_store=get_vectorstore()

    # ------------------------------- # 1. Similarity Search 
    # # ------------------------------- 
    if mode == "similarity": 
        docs = vector_store.similarity_search( query, k=4 ) 
        return docs
    

    # ------------------------------- # 2. MMR Retrieval
    #  # ------------------------------- 
    if mode == "mmr":
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "lambda_mult": 0.6
            }
        )
        docs = retriever.invoke(query) 
        return docs
    

    # ------------------------------- # 2. MMR + Rereanking Retrieval
    #  # ------------------------------- 
    if mode == "rerank":
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "lambda_mult": 0.6
            }
        )
        retrieved_docs = retriever.invoke(query) 
        reranked_docs = rerank_documents( query, retrieved_docs, top_n=4 ) 
        return reranked_docs
    

    raise ValueError( f"Unknown retrieval mode: {mode}" )



if __name__ == "__main__":

    query = "what does he say about nuclear fission"

    docs = retrieve_chunks(query)

    print("\n========= FINAL RETURNED DOCS =========\n")

    for idx, doc in enumerate(docs):

        print(f"\nFINAL {idx+1}")
        print(doc.page_content[:300])
        print(doc.metadata)
    

