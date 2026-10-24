import Database from "better-sqlite3";
import fs from "node:fs";
import path from "node:path";

function migrateLegacyMessages(db, rows) {
  const createChat = db.prepare(`
    INSERT INTO chats (user_id, title, created_at, updated_at)
    VALUES (?, 'Previous chat', ?, ?)
  `);
  const attachMessages = db.prepare(
    "UPDATE messages SET chat_id = ? WHERE user_id = ? AND chat_id IS NULL",
  );
  for (const row of rows) {
    const timestamp = row.created_at ?? new Date().toISOString();
    const result = createChat.run(row.user_id, timestamp, timestamp);
    attachMessages.run(result.lastInsertRowid, row.user_id);
  }
}

export function openDatabase(databasePath) {
  fs.mkdirSync(path.dirname(path.resolve(databasePath)), { recursive: true });
  const db = new Database(databasePath);
  db.pragma("foreign_keys = ON");
  db.pragma("journal_mode = WAL");

  db.exec(`
    CREATE TABLE IF NOT EXISTS users (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      username TEXT UNIQUE NOT NULL,
      password_hash TEXT NOT NULL,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS chats (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      title TEXT NOT NULL DEFAULT 'New chat',
      video_id TEXT,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP,
      updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS messages (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id INTEGER NOT NULL REFERENCES users(id),
      role TEXT NOT NULL,
      content TEXT NOT NULL,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
  `);

  const columns = db.prepare("PRAGMA table_info(messages)").all();
  const columnNames = [];
  for (const column of columns) columnNames.push(column.name);
  if (!columnNames.includes("chat_id")) {
    db.exec("ALTER TABLE messages ADD COLUMN chat_id INTEGER REFERENCES chats(id)");
  }
  if (!columnNames.includes("citations")) {
    db.exec("ALTER TABLE messages ADD COLUMN citations TEXT");
  }
  db.exec(`
    CREATE INDEX IF NOT EXISTS idx_chats_user_updated ON chats(user_id, updated_at);
    CREATE INDEX IF NOT EXISTS idx_messages_chat_id_id ON messages(chat_id, id);
  `);

  const legacyRows = db.prepare(`
    SELECT user_id, MIN(created_at) AS created_at
    FROM messages
    WHERE chat_id IS NULL
    GROUP BY user_id
  `).all();
  db.transaction(migrateLegacyMessages.bind(null, db))(legacyRows);

  return db;
}
