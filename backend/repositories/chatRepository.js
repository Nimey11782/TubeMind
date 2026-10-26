export async function listChats(db, userId) {
  const result = await db.query(`
    SELECT chats.*,
      (SELECT content FROM messages WHERE chat_id = chats.id ORDER BY id DESC LIMIT 1) AS last_message
    FROM chats
    WHERE user_id = $1
    ORDER BY updated_at DESC, id DESC
  `, [userId]);
  return result.rows;
}

export async function createChat(db, userId) {
  const result = await db.query(
    "INSERT INTO chats (user_id) VALUES ($1) RETURNING *",
    [userId],
  );
  return result.rows[0];
}

export async function findOwnedChat(db, chatId, userId) {
  if (!/^[1-9]\d*$/.test(String(chatId))) return null;
  const result = await db.query(
    "SELECT * FROM chats WHERE id = $1 AND user_id = $2",
    [Number(chatId), userId],
  );
  return result.rows[0] ?? null;
}

export async function listMessages(db, chatId) {
  const result = await db.query(`
    SELECT id, role, content, citations, created_at
    FROM messages
    WHERE chat_id = $1
    ORDER BY id
  `, [chatId]);
  return result.rows.map((message) => ({
    ...message,
    citations: message.citations ?? [],
  }));
}

export async function getRecentMessages(db, chatId, limit = 8) {
  const result = await db.query(`
    SELECT role, content FROM messages WHERE chat_id = $1 ORDER BY id DESC LIMIT $2
  `, [chatId, limit]);
  return result.rows.reverse();
}

export async function setChatVideo(db, chatId, videoId) {
  await db.query(`
    UPDATE chats SET video_id = $1, updated_at = CURRENT_TIMESTAMP WHERE id = $2
  `, [videoId, chatId]);
}

export async function saveConversation(db, { userId, chatId, question, answer, citations }) {
  const client = await db.connect();
  try {
    await client.query("BEGIN");
    await client.query(
      "INSERT INTO messages (user_id, chat_id, role, content) VALUES ($1, $2, 'user', $3)",
      [userId, chatId, question],
    );
    await client.query(`
      INSERT INTO messages (user_id, chat_id, role, content, citations)
      VALUES ($1, $2, 'assistant', $3, $4)
    `, [userId, chatId, answer, JSON.stringify(citations)]);
    await client.query(`
      UPDATE chats
      SET title = CASE WHEN title = 'New chat' THEN $1 ELSE title END,
          updated_at = CURRENT_TIMESTAMP
      WHERE id = $2
    `, [question.slice(0, 60), chatId]);
    await client.query("COMMIT");
  } catch (error) {
    await client.query("ROLLBACK");
    throw error;
  } finally {
    client.release();
  }
}

export async function deleteChat(db, chatId) {
  const client = await db.connect();
  try {
    await client.query("BEGIN");
    await client.query(
      "DELETE FROM transcript_chunks WHERE chat_id = $1",
      [chatId],
    );
    await client.query("DELETE FROM chats WHERE id = $1", [chatId]);
    await client.query("COMMIT");
  } catch (error) {
    await client.query("ROLLBACK");
    throw error;
  } finally {
    client.release();
  }
}
