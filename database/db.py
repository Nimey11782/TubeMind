import sqlite3

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_NAME = os.getenv("DB_PATH", str(PROJECT_ROOT / "chatbot.db"))

def get_connection():
    database_path = Path(DB_NAME).expanduser()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(database_path)
    conn.row_factory = sqlite3.Row
    return conn