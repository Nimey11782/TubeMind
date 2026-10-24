import jwt from "jsonwebtoken";
import { HttpError } from "../errors/HttpError.js";
import { createUser, findUserByUsername, verifyPassword } from "../repositories/userRepository.js";

export function registerUser(db, username, password) {
  if (typeof username !== "string" || username.trim().length < 3 ||
      username.trim().length > 40 || typeof password !== "string" ||
      password.length < 8 || password.length > 200) {
    throw new HttpError(
      400,
      "Username must be 3-40 characters and password at least 8 characters",
    );
  }
  if (!createUser(db, username.trim(), password)) {
    throw new HttpError(409, "Username already exists");
  }
  return { message: "User registered successfully" };
}

export function loginUser(db, jwtSecret, username, password) {
  if (typeof username !== "string" || typeof password !== "string") {
    throw new HttpError(400, "Username and password are required");
  }
  const user = findUserByUsername(db, username.trim());
  if (!user || !verifyPassword(password, user.password_hash)) {
    throw new HttpError(401, "Invalid credentials");
  }
  const accessToken = jwt.sign(
    { user_id: user.id, username: user.username },
    jwtSecret,
    { algorithm: "HS256", expiresIn: "1d" },
  );
  return { access_token: accessToken, token_type: "bearer", user_id: user.id };
}
