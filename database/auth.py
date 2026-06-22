import bcrypt
from database.db import get_connection


def register_user(username, password):

    conn = get_connection()
    cursor = conn.cursor()

    password_hash = bcrypt.hashpw(
        password.encode(),
        bcrypt.gensalt()
    ).decode()

    try:
        cursor.execute(
            """
            INSERT INTO users(username,password_hash)
            VALUES(?,?)
            """,
            (username, password_hash)
        )

        conn.commit()
        return True

    except Exception as e:
        print(e)
        return False

    finally:
        conn.close()


def login_user(username, password):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT * FROM users
        WHERE username=?
        """,
        (username,)
    )

    user = cursor.fetchone()

    conn.close()

    if user is None:
        return None

    if not bcrypt.checkpw(
        password.encode(),
        user["password_hash"].encode()
    ):
        return None

    return user


if __name__ == "__main__":

    print(
        login_user(
            "nimey",
            "123456"
        )
    )