import jwt from "jsonwebtoken";

export function authenticate(req, res, next) {
  const authorization = req.get("authorization") ?? "";
  const match = authorization.match(/^Bearer\s+(.+)$/i);
  if (!match) {
    return res.status(401).json({ error: "Authentication required" });
  }
  try {
    req.user = jwt.verify(match[1], req.app.locals.jwtSecret, {
      algorithms: ["HS256"],
    });
    return next();
  } catch {
    return res.status(401).json({ error: "Invalid or expired token" });
  }
}
