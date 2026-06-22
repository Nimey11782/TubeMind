from database.db import get_connection


def save_message(
    user_id,
    role,
    content
):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO messages(
            user_id,
            role,
            content
        )
        VALUES(?,?,?)
        """,
        (
            user_id,
            role,
            content
        )
    )

    conn.commit()
    conn.close()


def get_chat_history(user_id, limit=8):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT role, content
        FROM messages
        WHERE user_id=?
        ORDER BY id DESC
        LIMIT ?
        """,
        (user_id, limit)
    )

    rows = cursor.fetchall()

    conn.close()

    return list(reversed(rows))


def clear_user_messages(user_id):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        DELETE FROM messages
        WHERE user_id=?
        """,
        (user_id,)
    )

    conn.commit()
    conn.close()


if __name__ == "__main__":

    save_message(
        1,
        "user",
        "What is open source AI?"
    )

    save_message(
        1,
        "assistant",
        "Open source AI allows..."
    )

    history = get_chat_history(1)

    for row in history:
        print(dict(row))