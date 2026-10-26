import dotenv from "dotenv";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createApp } from "./app.js";
import { initializeDatabase, openDatabase } from "./db.js";

const directory = path.dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: path.resolve(directory, "..", ".env") });

async function startServer() {
  if (!process.env.JWT_SECRET_KEY) {
    throw new Error("JWT_SECRET_KEY is required");
  }
  const db = openDatabase();
  try {
    await initializeDatabase(db);
    const app = createApp({
      db,
      jwtSecret: process.env.JWT_SECRET_KEY,
      ragUrl: process.env.RAG_URL ?? "http://127.0.0.1:8001",
      frontendOrigin: process.env.FRONTEND_ORIGIN,
    });
    const port = Number(process.env.PORT ?? 8000);
    const server = app.listen(port, "0.0.0.0", () => {
      console.log(`Express API listening on port ${port}`);
    });

    function shutdown() {
      server.close(() => db.end().then(() => process.exit(0)));
    }

    process.on("SIGINT", shutdown);
    process.on("SIGTERM", shutdown);
  } catch (error) {
    await db.end();
    throw error;
  }
}

startServer().catch((error) => {
  console.error("Failed to start Express API", error);
  process.exitCode = 1;
});
