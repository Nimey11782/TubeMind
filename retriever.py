from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_classic.retrievers import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

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

def retrieve_chunks(query: str,k: int=4):
    vector_store=get_vectorstore()
    base_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "lambda_mult": 0.6
        }
    )
    docs = base_retriever.invoke(query)
    return docs
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