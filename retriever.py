from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

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

DEBUG = False
def retrieve_chunks(query: str,k: int=10):
    vector_store=get_vectorstore()
    base_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "lambda_mult": 0.6
        }
    )
    retrieved_docs = base_retriever.invoke(query)
    reranked_docs = rerank_documents(
        query,
        retrieved_docs,
        top_n=4
    )
    if DEBUG:
        print("\n========= BEFORE RERANKING =========\n")

        for idx, doc in enumerate(retrieved_docs):

            print(f"\nRESULT {idx+1}")
            print(doc.page_content[:300])
            print(doc.metadata)

        # Stage 2 → precision reranking
        reranked_docs = rerank_documents(
            query,
            retrieved_docs,
            top_n=4
        )

        print("\n========= AFTER RERANKING =========\n")

        for idx, doc in enumerate(reranked_docs):

            print(f"\nRERANKED {idx+1}")
            print(doc.page_content[:300])
            print(doc.metadata)

    return reranked_docs


if __name__ == "__main__":

    query = "what does he say about nuclear fission"

    docs = retrieve_chunks(query)

    print("\n========= FINAL RETURNED DOCS =========\n")

    for idx, doc in enumerate(docs):

        print(f"\nFINAL {idx+1}")
        print(doc.page_content[:300])
        print(doc.metadata)
    '''
    # Gemini LLM (for query rewriting + compression)
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        temperature=0
    )
    
    # Multi-Query Retrieval
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )

    # Contextual Compression (sentence extraction)
    compressor = LLMChainExtractor.from_llm(llm)

    compression_retriever = ContextualCompressionRetriever(
        base_retriever=multi_query_retriever,
        base_compressor=compressor
    )
    docs = compression_retriever.invoke(query)

    return docs 
    '''

"""
  don't get confused by code order 
  runtime order is : multiquery->mmr->compressor
  when contextulcompressionretriever is called it will call the base retreiver ->multiquery
  for each rewritten query by multiqueery retriever it call its base retreiver -> mmr
  for each rewritten query mmr-> embeds the query ,performs similarity search,applies 
   mmr diversification , return diverse chunks
  compression runs last only extracts relevant sentences,removes noise
"""