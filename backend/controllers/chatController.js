import {
  createUserChat,
  deleteUserChat,
  getChatMessages,
  getUserChats,
  ingestChatVideo,
  persistChatAnswer,
  prepareChatRequest,
} from "../services/chatService.js";
import { openRagChatStream, readRagEvents } from "../services/ragClient.js";

export async function listChats(req, res) {
  return res.json(await getUserChats(req.app.locals.db, req.user.user_id));
}

export async function createChat(req, res) {
  return res.status(201).json(await createUserChat(req.app.locals.db, req.user.user_id));
}

export async function getMessages(req, res) {
  return res.json(await getChatMessages(
    req.app.locals.db,
    req.params.chatId,
    req.user.user_id,
  ));
}

export async function removeChat(req, res) {
  await deleteUserChat(req.app.locals.db, req.params.chatId, req.user.user_id);
  return res.status(204).end();
}

export async function ingest(req, res) {
  const result = await ingestChatVideo(
    req.app.locals.db,
    req.app.locals.ragUrl,
    req.params.chatId,
    req.user.user_id,
    req.body?.url,
  );
  return res.json(result);
}

export async function streamChat(req, res) {
  const request = await prepareChatRequest(req.app.locals.db, {
    chatId: req.params.chatId,
    userId: req.user.user_id,
    question: req.body?.question,
    retrievalMode: req.body?.retrieval_mode,
  });
  const upstream = await openRagChatStream(req.app.locals.ragUrl, request);

  res.status(200);
  res.set({
    "content-type": "text/event-stream",
    "cache-control": "no-cache",
    connection: "keep-alive",
    "x-accel-buffering": "no",
  });
  let answer = "";
  let citations = [];
  let completed = false;
  for await (const event of readRagEvents(upstream)) {
    if (event.type === "token") answer += event.content;
    if (event.type === "done") {
      citations = event.citations ?? [];
      completed = true;
    }
    if (!res.destroyed) res.write(`data: ${JSON.stringify(event)}\n\n`);
  }
  if (!completed) {
    if (!res.destroyed) {
      res.write(`data: ${JSON.stringify({
        type: "error",
        content: "Answer stream ended unexpectedly",
      })}\n\n`);
      res.end();
    }
    return;
  }
  await persistChatAnswer(req.app.locals.db, request, answer, citations);
  if (!res.destroyed) res.end();
}
