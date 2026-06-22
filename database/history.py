from langchain_core.messages import (
    HumanMessage,
    AIMessage
)

from database.chat_store import get_chat_history


def load_chat_history(user_id):

    rows = get_chat_history(user_id)

    history = []

    for row in rows:

        if row["role"] == "user":

            history.append(
                HumanMessage(
                    content=row["content"]
                )
            )

        else:

            history.append(
                AIMessage(
                    content=row["content"]
                )
            )

    return history