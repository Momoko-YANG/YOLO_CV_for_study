/**
 * API service layer — all calls to the FastAPI backend.
 * Mirrors the Qt signal/slot interactions in LoginWindow.py and System_noLogin.py.
 */
import type {
  LoginRequest, UserInfo, DetectionResult, DetectorParams,
} from "../types";

const BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

// ── helpers ───────────────────────────────────────────────────────────────
async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "request failed");
  }
  return res.json() as Promise<T>;
}

async function postForm<T>(path: string, form: FormData): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "request failed");
  }
  return res.json() as Promise<T>;
}

// ── Auth ──────────────────────────────────────────────────────────────────
export const authApi = {
  login: (req: LoginRequest) =>
    post<UserInfo>("/auth/login", req),

  register: (
    username: string,
    password: string,
    captcha: string,
    captchaAnswer: string,
    avatar?: File,
  ) => {
    const form = new FormData();
    form.append("username", username);
    form.append("password", password);
    form.append("captcha", captcha);
    form.append("captcha_answer", captchaAnswer);
    if (avatar) form.append("avatar", avatar);
    return postForm<{ message: string }>("/auth/register", form);
  },

  changePassword: (username: string, new_password: string, captcha: string) =>
    post("/auth/change-password", { username, new_password, captcha }),

  changeAvatar: (username: string, password: string, avatar: File) => {
    const form = new FormData();
    form.append("username", username);
    form.append("password", password);
    form.append("avatar", avatar);
    return postForm<{ avatar_path: string }>("/auth/change-avatar", form);
  },

  deleteAccount: (username: string, password: string) =>
    fetch(`${BASE}/auth/account`, {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    }).then((r) => r.json()),
};

// ── Detection ─────────────────────────────────────────────────────────────
export const detectionApi = {
  /** Upload a static image; receive annotated result + detection data. */
  detectImage: async (file: File): Promise<DetectionResult> => {
    const form = new FormData();
    form.append("file", file);
    return postForm<DetectionResult>("/detect/image/annotated", form);
  },

  /** Update confidence / IOU sliders. */
  updateParams: (params: DetectorParams) =>
    post("/detect/params", params),

  /** Open a WebSocket for live camera / video streaming. */
  openStream: (
    onFrame: (result: DetectionResult) => void,
    onError?: (e: Event) => void,
  ): WebSocket => {
    const ws = new WebSocket(`${BASE.replace("http", "ws")}/detect/stream`);
    ws.binaryType = "arraybuffer";
    ws.onmessage = (e) => {
      try {
        onFrame(JSON.parse(e.data) as DetectionResult);
      } catch {/* ignore parse errors */}
    };
    if (onError) ws.onerror = onError;
    return ws;
  },
};
