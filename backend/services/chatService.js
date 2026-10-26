import { HttpError } from "../errors/HttpError.js";
import {
  createChat,
  deleteChat,
  findOwnedChat,
  getRecentMessages,
  listChats,
  listMessages,
  saveConversation,
  setChatVideo,
} from "../repositories/chatRepository.js";
import { ingestVideo } from "./ragClient.js";

const VIDEO_ID_PATTERN = /^[\w-]{11}$/;
const RETRIEVAL_MODES = new Set(["similarity", "mmr", "rerank", "hybrid"]);

export function getVideoId(url) {
  let parsed;
  try {
    parsed = new URL(url);
  } catch {
    return null;
  }
  const host = parsed.hostname.toLowerCase();
  let videoId = null;
  if (["youtube.com", "www.youtube.com", "m.youtube.com"].includes(host)) {
    if (parsed.pathname === "/watch") {
      videoId = parsed.searchParams.get("v");
    } else {
      videoId = parsed.pathname.match(/^\/(?:shorts|embed|live)\/([\w-]{11})(?:\/|$)/)?.[1];
    }
  } else if (host === "youtu.be") {
    videoId = parsed.pathname.split("/").filter(Boolean)[0];
  }
  return videoId && VIDEO_ID_PATTERN.test(videoId) ? videoId : null;
}

async function requireOwnedChat(db, chatId, userId) {
  const chat = await findOwnedChat(db, chatId, userId);
  if (!chat) throw new HttpError(404, "Chat not found");
  return chat;
}

export async function getUserChats(db, userId) {
  return listChats(db, userId);
}

export async function createUserChat(db, userId) {
  return createChat(db, userId);
}

export async function getChatMessages(db, chatId, userId) {
  const chat = await requireOwnedChat(db, chatId, userId);
  return { chat, messages: await listMessages(db, chat.id) };
}

export async function deleteUserChat(db, chatId, userId) {
  const chat = await requireOwnedChat(db, chatId, userId);
  await deleteChat(db, chat.id);
}

export async function ingestChatVideo(db, ragUrl, chatId, userId, url) {
  const chat = await requireOwnedChat(db, chatId, userId);
  if (typeof url !== "string" || url.length > 2048) {
    throw new HttpError(400, "A valid YouTube URL is required");
  }
  const videoId = getVideoId(url);
  if (!videoId) throw new HttpError(400, "Invalid YouTube URL");
  if (chat.video_id && chat.video_id !== videoId) {
    throw new HttpError(
      409,
      "This chat already has a video. Create a new chat to use another video.",
    );
  }
  await ingestVideo(ragUrl, { url, userId, chatId: chat.id });
  await setChatVideo(db, chat.id, videoId);
  return { status: "ok", video_id: videoId };
}

export async function prepareChatRequest(db, { chatId, userId, question, retrievalMode }) {
  const chat = await requireOwnedChat(db, chatId, userId);
  if (!chat.video_id) {
    throw new HttpError(409, "Add a video to this chat before asking a question");
  }
  if (typeof question !== "string" || !question.trim() || question.length > 2000) {
    throw new HttpError(400, "Question must be 1-2000 characters");
  }
  const mode = retrievalMode ?? "hybrid";
  if (!RETRIEVAL_MODES.has(mode)) {
    throw new HttpError(400, "Invalid retrieval mode");
  }
  return {
    chat,
    userId,
    question: question.trim(),
    retrievalMode: mode,
    history: await getRecentMessages(db, chat.id),
  };
}

export async function persistChatAnswer(db, request, answer, citations) {
  await saveConversation(db, {
    userId: request.userId,
    chatId: request.chat.id,
    question: request.question,
    answer,
    citations,
  });
}
