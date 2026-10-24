export default function AuthForm({
  mode,
  username,
  password,
  error,
  busy,
  onModeChange,
  onUsernameChange,
  onPasswordChange,
  onSubmit,
}) {
  return (
    <main className="auth-page">
      <form className="auth-card" onSubmit={onSubmit}>
        <h1>TubeMind</h1>
        <p>Ask questions about a YouTube video.</p>
        <div className="auth-tabs">
          <button type="button" data-mode="login" className={mode === "login" ? "active" : ""} onClick={onModeChange}>Log in</button>
          <button type="button" data-mode="register" className={mode === "register" ? "active" : ""} onClick={onModeChange}>Create account</button>
        </div>
        <label>
          Username
          <input value={username} onChange={onUsernameChange} autoComplete="username" required />
        </label>
        <label>
          Password
          <input type="password" value={password} onChange={onPasswordChange} autoComplete={mode === "login" ? "current-password" : "new-password"} required />
        </label>
        {error && <p className="error">{error}</p>}
        <button className="primary" disabled={busy}>{busy ? "Please wait..." : mode === "login" ? "Log in" : "Create account"}</button>
      </form>
    </main>
  );
}
