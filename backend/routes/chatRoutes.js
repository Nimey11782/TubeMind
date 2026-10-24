import { Router } from "express";
import {
  createChat,
  getMessages,
  ingest,
  listChats,
  removeChat,
  streamChat,
} from "../controllers/chatController.js";
import { authenticate } from "../middleware/authenticate.js";

const router = Router();
router.use(authenticate);
router.get("/", listChats);
router.post("/", createChat);
router.get("/:chatId/messages", getMessages);
router.delete("/:chatId", removeChat);
router.post("/:chatId/ingest", ingest);
router.post("/:chatId/chat/stream", streamChat);

export default router;
