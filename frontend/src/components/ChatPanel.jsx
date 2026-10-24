import MessageList from "./MessageList.jsx";

export default function ChatPanel({
  activeChat,
  messages,
  videoUrl,
  busy,
  error,
  endRef,
  onAddVideo,
  onVideoUrlChange,
  onSendQuestion,
  onCreateChat,
  question,
  onQuestionChange,
}) {
  return (
    <main className="conversation">
      {!activeChat ? (
        <section className="empty-state">
          <h2>Chat with a video</h2>
          <p>Create a chat, add one YouTube video, then ask questions about it.</p>
          <button className="primary" onClick={onCreateChat}>Start a new chat</button>
        </section>
      ) : (
        <>
          <header className="conversation-header">
            <div>
              <h2>{activeChat.title}</h2>
              {activeChat.video_id && <a href={`https://youtube.com/watch?v=${activeChat.video_id}`} target="_blank" rel="noreferrer">Video: {activeChat.video_id}</a>}
            </div>
          </header>
          {!activeChat.video_id && (
            <form className="ingest-form" onSubmit={onAddVideo}>
              <label htmlFor="video-url">Add one YouTube video to this chat</label>
              <div className="input-row">
                <input id="video-url" type="url" placeholder="https://www.youtube.com/watch?v=..." value={videoUrl} onChange={onVideoUrlChange} required />
                <button className="primary" disabled={busy}>{busy ? "Loading video..." : "Add video"}</button>
              </div>
              <small>A different video needs a separate chat.</small>
            </form>
          )}
          <MessageList messages={messages} hasVideo={Boolean(activeChat.video_id)} endRef={endRef} />
          {activeChat.video_id && (
            <form className="question-form" onSubmit={onSendQuestion}>
              <textarea value={question} onChange={onQuestionChange} placeholder="Ask about this video..." rows="2" maxLength="2000" disabled={busy} />
              <button className="primary" disabled={busy || !question.trim()}>{busy ? "Thinking..." : "Send"}</button>
            </form>
          )}
        </>
      )}
      {error && <p className="error toast">{error}</p>}
    </main>
  );
}
