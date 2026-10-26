import { loginUser, registerUser } from "../services/authService.js";

export async function register(req, res) {
  const result = await registerUser(
    req.app.locals.db,
    req.body?.username,
    req.body?.password,
  );
  return res.status(201).json(result);
}

export async function login(req, res) {
  const result = await loginUser(
    req.app.locals.db,
    req.app.locals.jwtSecret,
    req.body?.username,
    req.body?.password,
  );
  return res.json(result);
}
