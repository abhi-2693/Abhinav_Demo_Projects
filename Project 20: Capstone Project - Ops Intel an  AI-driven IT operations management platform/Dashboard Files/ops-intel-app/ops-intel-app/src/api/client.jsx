// src/api/client.js
//
// This file handles all communication with the FastAPI server.
// Every page calls apiGet() to fetch its data.

const BASE_URL = "http://localhost:8000";

export async function apiGet(path) {
  const response = await fetch(`${BASE_URL}${path}`);
  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`);
  }
  return response.json();
}