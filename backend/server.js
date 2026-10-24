import dotenv from "dotenv";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createApp } from "./app.js";
import { openDatabase } from "./db.js";

const directory = path.dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: path.resolve(directory, "..", ".env") });
const databasePath = process.env.DB_PATH ?? path.resolve(directory, "..", "chatbot.db");
const db = openDatabase(databasePath);
const app = createApp({
  db,
  jwtSecret: process.env.JWT_SECRET_KEY,
  ragUrl: process.env.RAG_URL ?? "http://127.0.0.1:8001",
  frontendOrigin: process.env.FRONTEND_ORIGIN,
});
const port = Number(process.env.PORT ?? 8000);
const server = app.listen(port, "0.0.0.0", logServerStart);

function shutdown() {
  server.close(closeDatabase);
}

function logServerStart() {
  console.log(`Express API listening on port ${port}`);
}

function closeDatabase() {
  db.close();
  process.exit(0);
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
