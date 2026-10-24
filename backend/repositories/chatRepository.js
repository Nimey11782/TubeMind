export function listChats(db, userId) {
  return db.prepare(`
    SELECT chats.*,
      (SELECT content FROM messages WHERE chat_id = chats.id ORDER BY id DESC LIMIT 1) AS last_message
    FROM chats
    WHERE user_id = ?
    ORDER BY updated_at DESC, id DESC
  `).all(userId);
}

export function createChat(db, userId) {
  const result = db.prepare("INSERT INTO chats (user_id) VALUES (?)").run(userId);
  return db.prepare("SELECT * FROM chats WHERE id = ?").get(result.lastInsertRowid);
}

export function findOwnedChat(db, chatId, userId) {
  if (!/^[1-9]\d*$/.test(String(chatId))) return null;
  return db.prepare("SELECT * FROM chats WHERE id = ? AND user_id = ?")
    .get(Number(chatId), userId) ?? null;
}

export function listMessages(db, chatId) {
  return db.prepare(`
    SELECT id, role, content, citations, created_at
    FROM messages
    WHERE chat_id = ?
    ORDER BY id
  `).all(chatId).map(decodeMessage);
}

function decodeMessage(message) {
  return {
    ...message,
    citations: message.citations ? JSON.parse(message.citations) : [],
  };
}

export function getRecentMessages(db, chatId, limit = 8) {
  return db.prepare(`
    SELECT role, content FROM messages WHERE chat_id = ? ORDER BY id DESC LIMIT ?
  `).all(chatId, limit).reverse();
}

export function setChatVideo(db, chatId, videoId) {
  db.prepare(`
    UPDATE chats SET video_id = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?
  `).run(videoId, chatId);
}

export function saveConversation(db, { userId, chatId, question, answer, citations }) {
  db.transaction(saveConversationStatements.bind(null, db, {
    userId,
    chatId,
    question,
    answer,
    citations,
  }))();
}

export function deleteChat(db, chatId) {
  db.transaction(deleteChatStatements.bind(null, db, chatId))();
}

function saveConversationStatements(db, { userId, chatId, question, answer, citations }) {
  db.prepare("INSERT INTO messages (user_id, chat_id, role, content) VALUES (?, ?, 'user', ?)")
    .run(userId, chatId, question);
  db.prepare(`
    INSERT INTO messages (user_id, chat_id, role, content, citations)
    VALUES (?, ?, 'assistant', ?, ?)
  `).run(userId, chatId, answer, JSON.stringify(citations));
  db.prepare(`
    UPDATE chats
    SET title = CASE WHEN title = 'New chat' THEN ? ELSE title END,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = ?
  `).run(question.slice(0, 60), chatId);
}

function deleteChatStatements(db, chatId) {
  db.prepare("DELETE FROM messages WHERE chat_id = ?").run(chatId);
  db.prepare("DELETE FROM chats WHERE id = ?").run(chatId);
}
