from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from dotenv import load_dotenv

load_dotenv()

def store_embeddings(docs,index_path="faiss_index"): #index_path->where to store FAISS
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder="./model_cache",
        encode_kwargs={'batch_size': 8},
        model_kwargs={'device': 'cpu'}
    )


    vectorstore = FAISS.from_documents(
        documents=docs,
        embedding=embeddings
    )

    vectorstore.save_local(index_path)
    return vectorstore




#FAISS is a vector store (only stores vectors)
#when we call .from_documents it iterates over documents (in our case split_docs) 
# and for every chunk the google model creates embeddings which are stored in FAISS
#and there is a mapping from vector->original_chunk

#vectorstore.save_local("faiss_index")
#This creates a folder like this: faiss_index/
#                                       ├── index.faiss
#                                       └── index.pkl
#🔹 index.faiss
#       Contains:
#           vector values , optimized search structure
#           VERY fast (C++ optimized)

# 🔹 index.pkl
#       Contains:
#           Document objects , metadata , mapping from FAISS IDs → Documents