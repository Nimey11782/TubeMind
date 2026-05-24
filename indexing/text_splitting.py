from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,        # ~150–200 words
        chunk_overlap=200,      # preserve context
        separators=[
            "\n\n",             # paragraph
            "\n",               # line
            ". ",               # sentence
            " ",                # word
            ""
        ]
    )
    return text_splitter.split_documents(docs) #not split text
# split document objects rather than text(i.e. into document objects) bcoz metadata 
# is presserved acroos chunks enabling traceability ,citation and advanced retreival

#each chunk is a langchain document object
#our split doc looks like -> split_docs[ Document(page_content="",metadata) ,  Document(....)]


