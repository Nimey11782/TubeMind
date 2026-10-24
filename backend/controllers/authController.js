import { loginUser, registerUser } from "../services/authService.js";

export function register(req, res) {
  const result = registerUser(req.app.locals.db, req.body?.username, req.body?.password);
  return res.status(201).json(result);
}

export function login(req, res) {
  const result = loginUser(
    req.app.locals.db,
    req.app.locals.jwtSecret,
    req.body?.username,
    req.body?.password,
  );
  return res.json(result);
}
