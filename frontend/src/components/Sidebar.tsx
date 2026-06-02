import { useRef, useState } from "react";
import { api } from "../lib/api";
import type { DocumentItem, SessionItem } from "../lib/types";

interface Props {
  documents: DocumentItem[];
  sessions: SessionItem[];
  activeSession: string | null;
  collapsed: boolean;
  onToggleCollapsed: () => void;
  onRefreshDocs: () => void;
  onRefreshSessions: () => void;
  onSelectSession: (id: string) => void;
  onNewSession: () => void;
}

export default function Sidebar({
  documents,
  sessions,
  activeSession,
  collapsed,
  onToggleCollapsed,
  onRefreshDocs,
  onRefreshSessions,
  onSelectSession,
  onNewSession,
}: Props) {
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  const wrap = async (fn: () => Promise<any>) => {
    setBusy(true);
    setErr("");
    try {
      await fn();
      onRefreshDocs();
    } catch (e: any) {
      setErr(String(e.message || e).slice(0, 200));
    } finally {
      setBusy(false);
    }
  };

  if (collapsed) {
    return (
      <aside className="sidebar sidebar-collapsed">
        <button
          className="collapse-rail-btn"
          title="Mở thanh bên"
          onClick={onToggleCollapsed}
          aria-label="Mở thanh bên"
        >
          📚
        </button>
        <button
          className="collapse-rail-btn"
          title="Cuộc trò chuyện mới"
          onClick={onNewSession}
          aria-label="Cuộc trò chuyện mới"
        >
          +
        </button>
      </aside>
    );
  }

  return (
    <aside className="sidebar">
      <div className="sidebar-head">
        <div className="brand">📚 RAG Agent <span>Tiếng Việt</span></div>
        <button
          className="panel-toggle"
          title="Thu gọn thanh bên"
          onClick={onToggleCollapsed}
          aria-label="Thu gọn thanh bên"
        >
          ◀
        </button>
      </div>

      <button className="btn primary block" onClick={onNewSession}>
        + Cuộc trò chuyện mới
      </button>

      <div className="section-title">Tài liệu</div>
      <div className="uploader">
        <input
          ref={fileRef}
          type="file"
          accept="application/pdf"
          hidden
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) wrap(() => api.uploadPdf(f));
            if (fileRef.current) fileRef.current.value = "";
          }}
        />
        <button className="btn block" disabled={busy} onClick={() => fileRef.current?.click()}>
          ⬆ Tải lên PDF
        </button>
        <div className="url-row">
          <input
            className="input"
            placeholder="Dán URL website…"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
          />
          <button
            className="btn"
            disabled={busy || !url.trim()}
            onClick={() => wrap(async () => { await api.ingestUrl(url.trim()); setUrl(""); })}
          >
            +
          </button>
        </div>
      </div>
      {busy && <div className="hint">Đang xử lý & lập chỉ mục…</div>}
      {err && <div className="error-box">{err}</div>}

      <div className="doc-list">
        {documents.length === 0 && <div className="muted small">Chưa có tài liệu nào.</div>}
        {documents.map((d) => (
          <div key={d.id} className="doc-item">
            <div className="doc-meta">
              <div className="doc-title" title={d.source}>
                {d.source_type === "pdf" ? "📄" : d.source_type === "url" ? "🌐" : "📝"} {d.title}
              </div>
              <div className="muted small">{d.n_chunks} đoạn</div>
            </div>
            <div className="doc-actions">
              {d.source_type === "pdf" && (
                <button
                  className="icon-btn"
                  title="Xem PDF"
                  onClick={() => window.open(api.documentPdfUrl(d.id), "_blank", "noopener,noreferrer")}
                >
                  👁
                </button>
              )}
              <button
                className="icon-btn"
                title="Xoá tài liệu"
                onClick={() => wrap(() => api.deleteDocument(d.id))}
              >
                ✕
              </button>
            </div>
          </div>
        ))}
      </div>

      <div className="section-title">Lịch sử trò chuyện</div>
      <div className="session-list">
        {sessions.map((s) => (
          <div
            key={s.id}
            className={`session-item ${s.id === activeSession ? "active" : ""}`}
            onClick={() => onSelectSession(s.id)}
          >
            <span className="session-title">{s.title}</span>
            <button
              className="icon-btn"
              title="Xoá"
              onClick={async (e) => {
                e.stopPropagation();
                await api.deleteSession(s.id);
                onRefreshSessions();
              }}
            >
              🗑
            </button>
          </div>
        ))}
      </div>
    </aside>
  );
}
