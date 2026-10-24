const API_URL = import.meta.env.VITE_API_URL ?? "/api";

export async function apiRequest(path, token, options = {}) {
  const response = await fetch(`${API_URL}${path}`, {
    ...options,
    headers: {
      ...(options.body ? { "content-type": "application/json" } : {}),
      ...(token ? { authorization: `Bearer ${token}` } : {}),
      ...options.headers,
    },
  });
  if (!response.ok) {
    const body = await readError(response);
    throw new Error(body.error ?? body.detail ?? "Request failed");
  }
  return response.status === 204 ? null : response.json();
}

export function apiUrl(path) {
  return `${API_URL}${path}`;
}

async function readError(response) {
  try {
    return await response.json();
  } catch {
    return {};
  }
}
