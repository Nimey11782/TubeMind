export function ChatMessage({ message }) {
  return (
    <article className={`message ${message.role}`}>
      <strong>{message.role === "user" ? "You" : "TubeMind"}</strong>
      <p>{message.content}</p>
      {message.citations?.length > 0 && (
        <div className="citations">
          <strong>Sources</strong>
          {message.citations.map(renderCitation)}
        </div>
      )}
    </article>
  );
}

export default function MessageList({ messages, hasVideo, endRef }) {
  return (
    <section className="messages">
      {messages.length === 0 && hasVideo && <p className="muted">Ask a question about this video to get started.</p>}
      {messages.map(renderMessage)}
      <div ref={endRef} />
    </section>
  );
}

function renderCitation(citation, index) {
  return (
    <a key={`${citation.video_id}-${citation.source_number ?? index}`} href={citation.youtube_link} target="_blank" rel="noreferrer">
      [{citation.source_number ?? index + 1}] {citation.timestamp}
    </a>
  );
}

function renderMessage(message, index) {
  return <ChatMessage key={message.id ?? index} message={message} />;
}
