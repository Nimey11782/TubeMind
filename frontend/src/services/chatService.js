import { apiRequest, apiUrl } from "./apiClient.js";

export function listChats(token) {
  return apiRequest("/chats", token);
}

export function createChat(token) {
  return apiRequest("/chats", token, { method: "POST" });
}

export function loadChat(token, chatId) {
  return apiRequest(`/chats/${chatId}/messages`, token);
}

export function deleteChat(token, chatId) {
  return apiRequest(`/chats/${chatId}`, token, { method: "DELETE" });
}

export function ingestVideo(token, chatId, url) {
  return apiRequest(`/chats/${chatId}/ingest`, token, {
    method: "POST",
    body: JSON.stringify({ url }),
  });
}

export async function* streamChat(token, chatId, question) {
  const response = await fetch(apiUrl(`/chats/${chatId}/chat/stream`), {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({ question }),
  });
  if (!response.ok) {
    const body = await readError(response);
    throw new Error(body.error ?? body.detail ?? "Could not generate an answer");
  }
  if (!response.body) throw new Error("The response stream is unavailable");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let completed = false;
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      const event = parseEventLine(line);
      if (!event) continue;
      if (event.type === "error") {
        throw new Error(event.content ?? "Answer generation failed");
      }
      if (event.type === "done") completed = true;
      yield event;
    }
  }
  buffer += decoder.decode();
  const finalEvent = parseEventLine(buffer);
  if (finalEvent) {
    if (finalEvent.type === "error") {
      throw new Error(finalEvent.content ?? "Answer generation failed");
    }
    if (finalEvent.type === "done") completed = true;
    yield finalEvent;
  }
  if (!completed) throw new Error("The answer stream ended unexpectedly");
}

async function readError(response) {
  try {
    return await response.json();
  } catch {
    return {};
  }
}

function parseEventLine(line) {
  if (!line.startsWith("data:")) return null;
  return JSON.parse(line.slice(5).trim());
}
