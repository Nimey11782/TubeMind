from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

rag_prompt = ChatPromptTemplate.from_messages([
    # 1️⃣ SYSTEM MESSAGE (behavior + rules)
    (
        "system",
        """You are an AI assistant analyzing a YouTube video transcript. 
        Answer using ONLY the provided context chunks.
        dont mention timestamps
        If the answer is not in the context, say 'This wasn't covered in the video."""
    ),

    # 2️⃣ CHAT HISTORY (previous Human & AI messages)
    MessagesPlaceholder(variable_name="chat_history"),

    # 3️⃣ HUMAN MESSAGE (current turn)
    (
        "human",
        """
        CONTEXT:
        {context}

        QUESTION:
        {question}
        """
    ),
])