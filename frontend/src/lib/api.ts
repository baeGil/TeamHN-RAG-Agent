import type {
  AppConfig,
  DocumentItem,
  Message,
  SessionItem,
} from "./types";

const BASE = "/api";

async function jget<T>(path: string): Promise<T> {
  const r = await fetch(`${BASE}${path}`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

async function jpost<T>(path: string, body?: any): Promise<T> {
  const r = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export const api = {
  config: () => jget<AppConfig>("/config"),
  stats: () => jget<any>("/stats"),

  listDocuments: () => jget<DocumentItem[]>("/documents"),
  documentPdfUrl: (id: number) => `${BASE}/documents/${id}/pdf`,
  ingestUrl: (url: string) => jpost("/documents/url", { url }),
  ingestText: (text: string, title?: string) =>
    jpost("/documents/text", { text, title }),
  deleteDocument: (id: number) =>
    fetch(`${BASE}/documents/${id}`, { method: "DELETE" }).then((r) => r.json()),
  uploadPdf: async (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    const r = await fetch(`${BASE}/documents/upload`, { method: "POST", body: fd });
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  },

  listSessions: () => jget<SessionItem[]>("/sessions"),
  createSession: () => jpost<{ id: string }>("/sessions", {}),
  getSession: (id: string) =>
    jget<{ id: string; messages: Message[]; summary: string | null }>(`/sessions/${id}`),
  deleteSession: (id: string) =>
    fetch(`${BASE}/sessions/${id}`, { method: "DELETE" }).then((r) => r.json()),
};

export interface StreamHandlers {
  onEvent: (type: string, data: any) => void;
  onError?: (msg: string) => void;
  onDone?: () => void;
}

// POST-based SSE streaming via fetch ReadableStream.
export async function streamChat(
  sessionId: string | null,
  message: string,
  handlers: StreamHandlers
): Promise<void> {
  const resp = await fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message }),
  });
  if (!resp.ok || !resp.body) {
    handlers.onError?.(await resp.text());
    return;
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const flush = (raw: string) => {
    const lines = raw.split("\n");
    let event = "message";
    const dataLines: string[] = [];
    for (const line of lines) {
      if (line.startsWith(":")) continue; // SSE comment (ping/keepalive)
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
    }
    if (dataLines.length === 0) return;
    try {
      const data = JSON.parse(dataLines.join("\n"));
      if (event === "error") handlers.onError?.(data.message || "Lỗi");
      else handlers.onEvent(event, data);
    } catch {
      /* ignore partial */
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const chunk = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      flush(chunk);
    }
  }
  if (buffer.trim()) flush(buffer);
  handlers.onDone?.();
}
