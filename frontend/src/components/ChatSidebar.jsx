export default function ChatSidebar({
  chats,
  activeChat,
  onCreate,
  onSelect,
  onDelete,
  onLogout,
}) {
  return (
    <aside className="sidebar">
      <div className="sidebar-heading">
        <h1>TubeMind</h1>
        <button className="primary" onClick={onCreate}>+ New chat</button>
      </div>
      <div className="chat-list">
        {chats.map((chat) => (
          <div className={`chat-item ${activeChat?.id === chat.id ? "selected" : ""}`} key={chat.id}>
            <button className="chat-select" onClick={onSelect.bind(null, chat)}>
              <span>{chat.title}</span>
              <small>{chat.video_id ? `Video · ${chat.video_id}` : "No video yet"}</small>
            </button>
            <button className="delete-chat" aria-label={`Delete ${chat.title}`} onClick={onDelete.bind(null, chat)}>×</button>
          </div>
        ))}
        {chats.length === 0 && <p className="muted">Your chats will appear here.</p>}
      </div>
      <button className="logout" onClick={onLogout}>Log out</button>
    </aside>
  );
}
