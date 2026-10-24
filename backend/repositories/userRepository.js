import bcrypt from "bcryptjs";

export function createUser(db, username, password) {
  try {
    return db.prepare("INSERT INTO users (username, password_hash) VALUES (?, ?)")
      .run(username, bcrypt.hashSync(password, 10));
  } catch (error) {
    if (error.code === "SQLITE_CONSTRAINT_UNIQUE") return null;
    throw error;
  }
}

export function findUserByUsername(db, username) {
  return db.prepare("SELECT * FROM users WHERE username = ?").get(username);
}

export function verifyPassword(password, passwordHash) {
  return bcrypt.compareSync(password, passwordHash);
}
