import sqlite3
import os

DB_NAME = os.getenv("DB_PATH", "/app/data/chatbot.db")

def get_connection():
    os.makedirs(os.path.dirname(DB_NAME), exist_ok=True)
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn