import bcrypt from "bcryptjs";

export function createUser(db, username, password) {
  return db.query(
    "INSERT INTO users (username, password_hash) VALUES ($1, $2) RETURNING id",
    [username, bcrypt.hashSync(password, 10)],
  );
}

export function findUserByUsername(db, username) {
  return db.query("SELECT * FROM users WHERE username = $1", [username])
    .then((result) => result.rows[0] ?? null);
}

export function verifyPassword(password, passwordHash) {
  return bcrypt.compareSync(password, passwordHash);
}
