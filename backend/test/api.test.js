import assert from "node:assert/strict";
import { after, test } from "node:test";
import { randomUUID } from "node:crypto";
import { createServer } from "node:http";

import { createApp } from "../app.js";
import { initializeDatabase, openDatabase } from "../db.js";

const servers = [];
const integrationTest = process.env.TEST_DATABASE_URL ? test : test.skip;

async function listen(server) {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  servers.push(server);
  return `http://127.0.0.1:${server.address().port}`;
}

after(async () => {
  await Promise.all(servers.map((server) => new Promise((resolve, reject) => {
    server.close((error) => error ? reject(error) : resolve());
  })));
});

integrationTest("users get isolated chats, videos, messages, and chat-specific retrieval", async () => {
  const ragRequests = [];
  const ragServer = createServer(async (req, res) => {
    const chunks = [];
    for await (const chunk of req) chunks.push(chunk);
    const body = JSON.parse(Buffer.concat(chunks).toString("utf8"));
    ragRequests.push({ path: req.url, body });
    res.setHeader("content-type", "application/json");
    if (req.url === "/ingest") {
      res.end(JSON.stringify({ status: "ok", video_id: "abcdefghijk" }));
    } else if (req.url === "/chat/stream") {
      res.writeHead(200, { "content-type": "text/event-stream" });
      res.end([
        'data: {"type":"token","content":"Answer ✓"}\n\n',
        'data: {"type":"done","citations":[{"source_number":1,"video_id":"abcdefghijk","timestamp":"00:01 - 00:02","youtube_link":"https://youtube.com/watch?v=abcdefghijk&t=1","snippet":"Source"}]}\n\n',
      ].join(""));
    } else {
      res.writeHead(404);
      res.end();
    }
  });
  const ragUrl = await listen(ragServer);
  const username = `sample-${randomUUID().slice(0, 30)}`;
  const otherUsername = `other-${randomUUID().slice(0, 30)}`;
  const db = openDatabase(process.env.TEST_DATABASE_URL);
  await initializeDatabase(db);
  const apiUrl = await listen(createApp({ db, jwtSecret: "test-secret", ragUrl }).listen(0, "127.0.0.1"));

  try {
    let response = await fetch(`${apiUrl}/register`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ username, password: "password123" }),
    });
    assert.equal(response.status, 201);
    response = await fetch(`${apiUrl}/login`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ username, password: "password123" }),
    });
    const { access_token: token } = await response.json();
    const auth = { authorization: `Bearer ${token}`, "content-type": "application/json" };

    const firstChat = await fetch(`${apiUrl}/chats`, { method: "POST", headers: auth }).then((r) => r.json());
    const secondChat = await fetch(`${apiUrl}/chats`, { method: "POST", headers: auth }).then((r) => r.json());
    assert.notEqual(firstChat.id, secondChat.id);

    response = await fetch(`${apiUrl}/chats/${firstChat.id}/ingest`, {
      method: "POST",
      headers: auth,
      body: JSON.stringify({ url: "https://youtu.be/abcdefghijk" }),
    });
    assert.equal(response.status, 200);
    response = await fetch(`${apiUrl}/chats/${firstChat.id}/ingest`, {
      method: "POST",
      headers: auth,
      body: JSON.stringify({ url: "https://youtu.be/lmnopqrstuv" }),
    });
    assert.equal(response.status, 409);

    response = await fetch(`${apiUrl}/chats/${firstChat.id}/chat/stream`, {
      method: "POST",
      headers: auth,
      body: JSON.stringify({ question: "What happened?" }),
    });
    assert.equal(response.status, 200);
    assert.match(await response.text(), /"type":"done"/);

    const firstMessages = await fetch(`${apiUrl}/chats/${firstChat.id}/messages`, { headers: auth }).then((r) => r.json());
    const secondMessages = await fetch(`${apiUrl}/chats/${secondChat.id}/messages`, { headers: auth }).then((r) => r.json());
    assert.equal(firstMessages.messages.length, 2);
    assert.equal(firstMessages.messages[1].content, "Answer ✓");
    assert.equal(firstMessages.messages[1].citations[0].video_id, "abcdefghijk");
    assert.equal(secondMessages.messages.length, 0);
    assert.equal(firstMessages.chat.title, "What happened?");
    assert.equal(ragRequests.filter((request) => request.path === "/ingest").length, 1);
    assert.equal(ragRequests.find((request) => request.path === "/chat/stream").body.chat_id, firstChat.id);

    const otherUser = await fetch(`${apiUrl}/register`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ username: otherUsername, password: "password123" }),
    });
    assert.equal(otherUser.status, 201);
    const otherLogin = await fetch(`${apiUrl}/login`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ username: otherUsername, password: "password123" }),
    }).then((r) => r.json());
    response = await fetch(`${apiUrl}/chats/${firstChat.id}/messages`, {
      headers: { authorization: `Bearer ${otherLogin.access_token}` },
    });
    assert.equal(response.status, 404);
  } finally {
    await db.query("DELETE FROM users WHERE username = ANY($1::text[])", [[username, otherUsername]]);
    await db.end();
  }
});

test("PostgreSQL configuration is required", () => {
  assert.throws(() => openDatabase(""), /DATABASE_URL is required/);
});
