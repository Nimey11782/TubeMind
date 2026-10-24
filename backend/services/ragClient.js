import { Readable } from "node:stream";
import { HttpError } from "../errors/HttpError.js";

async function postRag(ragUrl, endpoint, body) {
  let response;
  try {
    response = await fetch(`${ragUrl}${endpoint}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (error) {
    throw new HttpError(502, `RAG service unavailable: ${error.message}`);
  }
  if (!response.ok) {
    let detail = "RAG service request failed";
    try {
      const body = await response.json();
      detail = body.detail ?? detail;
    } catch {
      // Non-JSON upstream responses still retain their HTTP status.
    }
    throw new HttpError(response.status, detail);
  }
  return response;
}

export async function ingestVideo(ragUrl, { url, userId, chatId }) {
  const response = await postRag(ragUrl, "/ingest", {
    url,
    user_id: userId,
    chat_id: chatId,
  });
  await response.arrayBuffer();
}

export async function openRagChatStream(ragUrl, request) {
  const response = await postRag(ragUrl, "/chat/stream", {
    question: request.question,
    retrieval_mode: request.retrievalMode,
    user_id: request.userId,
    chat_id: request.chat.id,
    history: request.history,
  });
  if (!response.body) throw new HttpError(502, "RAG service returned an empty response");
  return response;
}

export async function* readRagEvents(response) {
  let lineBuffer = "";
  const decoder = new TextDecoder();
  for await (const chunk of Readable.fromWeb(response.body)) {
    lineBuffer += decoder.decode(chunk, { stream: true });
    const lines = lineBuffer.split("\n");
    lineBuffer = lines.pop() ?? "";
    for (const line of lines) {
      const event = parseRagEvent(line);
      if (event) yield event;
    }
  }
  lineBuffer += decoder.decode();
  if (lineBuffer) {
    const event = parseRagEvent(lineBuffer);
    if (event) yield event;
  }
}

function parseRagEvent(line) {
  if (!line.startsWith("data:")) return null;
  return JSON.parse(line.slice(5).trim());
}
