import cors from "cors";
import express from "express";
import { errorHandler } from "./middleware/errorHandler.js";
import authRoutes from "./routes/authRoutes.js";
import chatRoutes from "./routes/chatRoutes.js";

export function createApp({ db, jwtSecret, ragUrl = "http://localhost:8001", frontendOrigin }) {
  if (!jwtSecret) throw new Error("JWT_SECRET_KEY is required");

  const app = express();
  app.locals.db = db;
  app.locals.jwtSecret = jwtSecret;
  app.locals.ragUrl = ragUrl;
  app.use(cors({ origin: frontendOrigin ?? false }));
  app.use(express.json({ limit: "1mb" }));
  app.get("/health", healthCheck);
  app.use(authRoutes);
  app.use("/chats", chatRoutes);
  app.use(errorHandler);
  return app;
}

function healthCheck(_req, res) {
  return res.json({ status: "alive" });
}
