// Backend URLs — configurable for cloud deployment via env vars.
// Next.js exposes NEXT_PUBLIC_* vars at build time.

export const API_BASE =
  (typeof process !== "undefined" && process.env.NEXT_PUBLIC_API_URL)
    ? process.env.NEXT_PUBLIC_API_URL
    : "http://127.0.0.1:8765";

// WebSocket base — swap http(s) for ws(s). Evaluated at runtime on client.
export const WS_BASE = API_BASE.replace(/^https/, "wss").replace(/^http/, "ws");
