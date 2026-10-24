import { apiRequest } from "./apiClient.js";

export function register(username, password) {
  return apiRequest("/register", null, {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
}

export function login(username, password) {
  return apiRequest("/login", null, {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
}
