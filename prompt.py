from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

rag_prompt = ChatPromptTemplate.from_messages([
    # 1️⃣ SYSTEM MESSAGE (behavior + rules)
    (
        "system",
        "You are a helpful and factual AI assistant. "
        "Answer the user's question using ONLY the provided context. "
        "Do not use prior knowledge. "
        "If the answer is not present in the context, say "
        "'I don't know based on the provided information.'"
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