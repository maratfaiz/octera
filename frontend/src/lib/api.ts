import type { AnalysisResult, Study, User } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const TOKEN_KEY = "octera_token";

// Mirrors the backend's ALLOWED_CONTENT_TYPES / MAX_UPLOAD_SIZE_MB (see
// backend/app/api/routes/studies.py, backend/app/core/config.py). Client-side
// validation is a UX nicety only -- the backend enforces these regardless.
export const ALLOWED_UPLOAD_TYPES = ["image/jpeg", "image/png", "image/tiff"];
export const MAX_UPLOAD_SIZE_MB = 20;

export function validateUploadFile(file: File): string | null {
  if (!ALLOWED_UPLOAD_TYPES.includes(file.type)) {
    return "Поддерживаются только изображения JPEG, PNG или TIFF";
  }
  if (file.size > MAX_UPLOAD_SIZE_MB * 1024 * 1024) {
    return `Файл слишком большой. Максимальный размер: ${MAX_UPLOAD_SIZE_MB} МБ`;
  }
  return null;
}

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY);
}

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const token = getToken();
  const headers = new Headers(options.headers);
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (options.body && !(options.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const res = await fetch(`${API_URL}${path}`, { ...options, headers });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new ApiError(res.status, body.detail ?? `Ошибка запроса: ${res.status}`);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export const auth = {
  register: (email: string, full_name: string, password: string) =>
    request<User>("/api/v1/auth/register", {
      method: "POST",
      body: JSON.stringify({ email, full_name, password }),
    }),
  login: async (email: string, password: string) => {
    const data = await request<{ access_token: string }>("/api/v1/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    });
    setToken(data.access_token);
    return data;
  },
  me: () => request<User>("/api/v1/auth/me"),
};

export const studies = {
  list: () => request<Study[]>("/api/v1/studies"),
  get: (id: string) => request<Study>(`/api/v1/studies/${id}`),
  upload: (file: File, eye?: string) => {
    const form = new FormData();
    form.append("file", file);
    const query = eye ? `?${new URLSearchParams({ eye }).toString()}` : "";
    return request<Study>(`/api/v1/studies${query}`, { method: "POST", body: form });
  },
};

export const analysis = {
  run: (studyId: string) => request<AnalysisResult>(`/api/v1/analysis/${studyId}/run`, { method: "POST" }),
  get: (studyId: string) => request<AnalysisResult>(`/api/v1/analysis/${studyId}`),
};

export async function fetchImageObjectUrl(path: string): Promise<string> {
  const token = getToken();
  const headers = new Headers();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const res = await fetch(`${API_URL}${path}`, { headers });
  if (!res.ok) throw new ApiError(res.status, "Не удалось загрузить изображение");
  const blob = await res.blob();
  return URL.createObjectURL(blob);
}

export { ApiError };
