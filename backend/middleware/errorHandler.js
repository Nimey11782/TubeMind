export function errorHandler(error, _req, res, _next) {
  if (!error.status || error.status >= 500) console.error(error);
  if (res.headersSent) {
    res.end();
    return;
  }
  res.status(error.status ?? 500).json({
    error: error.status ? error.message : "Internal server error",
  });
}
