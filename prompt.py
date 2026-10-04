from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

rag_prompt = ChatPromptTemplate.from_messages([
    # 1️⃣ SYSTEM MESSAGE (behavior + rules)
    (
        "system",
        """You are an AI assistant analyzing a YouTube video transcript.
        Answer using only the provided transcript sources. Cite each factual claim with its matching
        source number, formatted like [1] or [2]. Never invent facts or citations. You may combine
        information from multiple sources when the sources support the answer. Keep the answer concise.
        Do not mention timestamps; clickable timestamp citations are provided separately.
        If the sources do not contain the answer, say exactly: "This wasn't covered in the video." """
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