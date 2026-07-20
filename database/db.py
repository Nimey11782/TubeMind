import sqlite3
import os

DB_NAME = os.getenv("DB_PATH", "/app/data/chatbot.db")

def get_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn