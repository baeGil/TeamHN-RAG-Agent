import { useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import type { DocumentItem, SessionItem } from "../lib/types";

const DEFAULT_MAX_UPLOAD_SIZE = 5 * 1024 * 1024;
const MAX_CONCURRENT_UPLOADS = 3;

interface UploadProgress {
  name: string;
  status: "uploading" | "done" | "error";
  error?: string;
}

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
  maxUploadSize?: number;
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
  maxUploadSize,
}: Props) {
  const [url, setUrl] = useState("");
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<UploadProgress[]>([]);
  const [err, setErr] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  const maxSize = maxUploadSize || DEFAULT_MAX_UPLOAD_SIZE;

  const hasProcessing = documents.some((d) => d.status === "processing");

  useEffect(() => {
    if (!hasProcessing) return;
    const t = setInterval(() => onRefreshDocs(), 2000);
    return () => clearInterval(t);
  }, [hasProcessing, onRefreshDocs]);

  const uploadFiles = async (files: File[]) => {
    if (files.length === 0) return;
    setUploading(true);
    setErr("");

    const progress: UploadProgress[] = files.map((f) => ({
      name: f.name,
      status: "uploading" as const,
    }));
    setUploadProgress([...progress]);

    let nextIdx = 0;
    const results: { file: File; ok: boolean; error?: string }[] = [];

    const uploadNext = async () => {
      while (nextIdx < files.length) {
        const idx = nextIdx++;
        const file = files[idx];
        try {
          await api.uploadPdf(file);
          progress[idx] = { name: file.name, status: "done" };
        } catch (e: any) {
          progress[idx] = { name: file.name, status: "error", error: String(e.message || e).slice(0, 100) };
        }
        setUploadProgress([...progress]);
      }
    };

    const concurrency = Math.min(MAX_CONCURRENT_UPLOADS, files.length);
    await Promise.all(Array.from({ length: concurrency }, () => uploadNext()));

    const errors = progress.filter((p) => p.status === "error");
    if (errors.length > 0) {
      setErr(errors.map((e) => `${e.name}: ${e.error || "Lỗi"}`).join("\n"));
    }

    onRefreshDocs();
    setUploading(false);
    setTimeout(() => setUploadProgress([]), 3000);
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
          multiple
          hidden
          onChange={(e) => {
            const files = Array.from(e.target.files || []);
            if (fileRef.current) fileRef.current.value = "";
            if (files.length === 0) return;
            const oversized = files.find((f) => f.size > maxSize);
            if (oversized) {
              setErr(
                `"${oversized.name}" quá lớn (${(oversized.size / 1024 / 1024).toFixed(1)}MB). Tối đa ${maxSize / 1024 / 1024}MB.`
              );
              return;
            }
            const nonPdf = files.find((f) => !f.name.toLowerCase().endsWith(".pdf"));
            if (nonPdf) {
              setErr(`"${nonPdf.name}" không phải PDF. Chỉ hỗ trợ tệp PDF.`);
              return;
            }
            uploadFiles(files);
          }}
        />
        <button className="btn block" disabled={uploading} onClick={() => fileRef.current?.click()}>
          {uploading ? "⏳ Đang tải lên…" : "⬆ Tải lên PDF"}
        </button>
        <div className="muted small" style={{ marginTop: 2 }}>
          Tối đa {maxSize / 1024 / 1024}MB mỗi file
        </div>
        <div className="url-row">
          <input
            className="input"
            placeholder="Dán URL website…"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
          />
          <button
            className="btn"
            disabled={uploading || !url.trim()}
            onClick={async () => {
              setUploading(true);
              setErr("");
              try {
                await api.ingestUrl(url.trim());
                setUrl("");
                onRefreshDocs();
              } catch (e: any) {
                setErr(String(e.message || e).slice(0, 200));
              } finally {
                setUploading(false);
              }
            }}
          >
            +
          </button>
        </div>
      </div>
      {err && <div className="error-box">{err}</div>}

      <div className="doc-list">
        {documents.length === 0 && !uploading && <div className="muted small">Chưa có tài liệu nào.</div>}
        {documents.map((d) => (
          <div key={d.id} className="doc-item">
            <div className="doc-meta">
              {d.source_type === "pdf" ? (
                <button
                  className="doc-title doc-title-link"
                  title="Xem PDF"
                  onClick={() => window.open(api.documentPdfUrl(d.id), "_blank", "noopener,noreferrer")}
                >
                  📄 {d.title}
                </button>
              ) : (
                <div className="doc-title" title={d.source}>
                  {d.source_type === "url" ? "🌐" : "📝"} {d.title}
                </div>
              )}
              <div className="muted small">
                {d.status === "processing" && <span style={{ color: "var(--accent)", fontWeight: 500 }}>⏳ Đang xử lý…</span>}
                {d.status === "failed" && `❌ Lỗi: ${d.error_message || "Không xác định"}`}
                {d.status === "ready" && `${d.n_chunks} đoạn`}
                {!d.status && `${d.n_chunks} đoạn`}
              </div>
            </div>
            <div className="doc-actions">
              <button
                className="icon-btn"
                title="Xoá tài liệu"
                onClick={async () => {
                  await api.deleteDocument(d.id);
                  onRefreshDocs();
                }}
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