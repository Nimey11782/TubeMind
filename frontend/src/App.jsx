import { useEffect, useRef, useState } from "react";

import AuthForm from "./components/AuthForm.jsx";
import ChatPanel from "./components/ChatPanel.jsx";
import ChatSidebar from "./components/ChatSidebar.jsx";
import { login, register } from "./services/authService.js";
import {
  createChat,
  deleteChat,
  ingestVideo,
  listChats,
  loadChat,
  streamChat,
} from "./services/chatService.js";

export default function App() {
  const [token, setToken] = useState(() => localStorage.getItem("tubemind_token"));
  const [authMode, setAuthMode] = useState("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [chats, setChats] = useState([]);
  const [activeChat, setActiveChat] = useState(null);
  const [messages, setMessages] = useState([]);
  const [videoUrl, setVideoUrl] = useState("");
  const [question, setQuestion] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const endRef = useRef(null);

  useEffect(() => {
    if (!token) return;
    localStorage.setItem("tubemind_token", token);
    loadWorkspace(token, setChats, setActiveChat, setMessages, setError);
  }, [token]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleAuthSubmit(event) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      if (authMode === "register") await register(username, password);
      const result = await login(username, password);
      localStorage.setItem("tubemind_token", result.access_token);
      setPassword("");
      setToken(result.access_token);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  }

  async function handleCreateChat() {
    setError("");
    try {
      const chat = await createChat(token);
      setChats((current) => [chat, ...current]);
      setActiveChat(chat);
      setMessages([]);
      setVideoUrl("");
    } catch (requestError) {
      setError(requestError.message);
    }
  }

  async function handleSelectChat(chat) {
    setError("");
    try {
      const result = await loadChat(token, chat.id);
      setActiveChat(result.chat);
      setMessages(result.messages);
      setVideoUrl("");
    } catch (requestError) {
      setError(requestError.message);
    }
  }

  async function handleDeleteChat(chat) {
    if (!window.confirm("Delete this chat and its messages?")) return;
    try {
      await deleteChat(token, chat.id);
      const remainingChats = chats.filter((item) => item.id !== chat.id);
      setChats(remainingChats);
      if (activeChat?.id === chat.id) {
        setActiveChat(null);
        setMessages([]);
        if (remainingChats.length > 0) {
          await handleSelectChat(remainingChats[0]);
        }
      }
    } catch (requestError) {
      setError(requestError.message);
    }
  }

  function handleLogout() {
    localStorage.removeItem("tubemind_token");
    setToken(null);
    setChats([]);
    setActiveChat(null);
    setMessages([]);
    setError("");
  }

  async function handleIngestVideo(event) {
    event.preventDefault();
    if (!activeChat) return;
    setBusy(true);
    setError("");
    try {
      await ingestVideo(token, activeChat.id, videoUrl);
      const result = await loadChat(token, activeChat.id);
      const updatedChats = await listChats(token);
      setActiveChat(result.chat);
      setMessages(result.messages);
      setChats(updatedChats);
      setVideoUrl("");
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  }

  async function handleSendQuestion(event) {
    event.preventDefault();
    const prompt = question.trim();
    if (!activeChat || !prompt || busy) return;

    const chat = activeChat;
    const previousMessages = messages;
    setQuestion("");
    setBusy(true);
    setError("");
    setMessages([
      ...previousMessages,
      { role: "user", content: prompt },
      { role: "assistant", content: "", citations: [] },
    ]);

    let answer = "";
    try {
      for await (const event of streamChat(token, chat.id, prompt)) {
        if (event.type === "token") {
          answer += event.content;
          setMessages((current) => current.map((message, index) => (
            index === current.length - 1 ? { ...message, content: answer } : message
          )));
        } else if (event.type === "done") {
          setMessages((current) => current.map((message, index) => (
            index === current.length - 1
              ? { ...message, citations: event.citations }
              : message
          )));
        }
      }
      const updatedChats = await listChats(token);
      const result = await loadChat(token, chat.id);
      setChats(updatedChats);
      setActiveChat(result.chat);
      setMessages(result.messages);
    } catch (requestError) {
      setMessages(previousMessages);
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  }

  if (!token) {
    return (
      <AuthForm
        mode={authMode}
        username={username}
        password={password}
        error={error}
        busy={busy}
        onModeChange={(event) => setAuthMode(event.currentTarget.dataset.mode)}
        onUsernameChange={(event) => setUsername(event.target.value)}
        onPasswordChange={(event) => setPassword(event.target.value)}
        onSubmit={handleAuthSubmit}
      />
    );
  }

  return (
    <div className="layout">
      <ChatSidebar
        chats={chats}
        activeChat={activeChat}
        onCreate={handleCreateChat}
        onSelect={handleSelectChat}
        onDelete={handleDeleteChat}
        onLogout={handleLogout}
      />
      <ChatPanel
        activeChat={activeChat}
        messages={messages}
        videoUrl={videoUrl}
        onVideoUrlChange={(event) => setVideoUrl(event.target.value)}
        question={question}
        onQuestionChange={(event) => setQuestion(event.target.value)}
        busy={busy}
        error={error}
        endRef={endRef}
        onAddVideo={handleIngestVideo}
        onSendQuestion={handleSendQuestion}
        onCreateChat={handleCreateChat}
      />
    </div>
  );
}

async function loadWorkspace(token, setChats, setActiveChat, setMessages, setError) {
  try {
    const chats = await listChats(token);
    const result = chats.length > 0 ? await loadChat(token, chats[0].id) : null;
    setChats(chats);
    setActiveChat(result?.chat ?? null);
    setMessages(result?.messages ?? []);
    setError("");
  } catch (requestError) {
    setError(requestError.message);
  }
}
