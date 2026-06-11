import { useCallback, useEffect, useState } from "react";
import { api } from "./lib/api";
import type {
  AppConfig,
  Citation,
  DocumentItem,
  Message,
  SessionItem,
} from "./lib/types";
import Sidebar from "./components/Sidebar";
import ChatPanel from "./components/ChatPanel";

const LS_KEY = "vnrag.session";

export default function App() {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [sessions, setSessions] = useState<SessionItem[]>([]);
  const [sidebarWidth, setSidebarWidth] = useState(300);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(
    localStorage.getItem(LS_KEY)
  );
  const [messages, setMessages] = useState<Message[]>([]);
  const [messagesLoading, setMessagesLoading] = useState(() => !!localStorage.getItem(LS_KEY));
  const [citation, setCitation] = useState<Citation | null>(null);

  const refreshDocs = useCallback(() => {
    api.listDocuments().then(setDocuments).catch(() => {});
  }, []);
  const refreshSessions = useCallback(() => {
    api.listSessions().then(setSessions).catch(() => {});
  }, []);

  useEffect(() => {
    api.config().then(setConfig).catch(() => {});
    refreshDocs();
    refreshSessions();
  }, [refreshDocs, refreshSessions]);

  // Restore messages on load / when session changes (persistence across reload)
  useEffect(() => {
    if (!sessionId) {
      setMessages([]);
      setMessagesLoading(false);
      return;
    }
    localStorage.setItem(LS_KEY, sessionId);
    setMessagesLoading(true);
    api
      .getSession(sessionId)
      .then((s) => {
        const msgs = (s.messages || []).map((m: any) => ({
          ...m,
          citations: m.citations || [],
          trace: m.trace || [],
          status: m.status || "complete",
          error_message: m.error_message || null,
        }));
        setMessages(msgs);
      })
      .catch(() => setMessages([]))
      .finally(() => setMessagesLoading(false));
  }, [sessionId]);

  const handleSession = (id: string) => {
    if (id !== sessionId) {
      localStorage.setItem(LS_KEY, id);
      setSessionId(id);
      refreshSessions();
    }
  };

  const newSession = async () => {
    const { id } = await api.createSession();
    localStorage.setItem(LS_KEY, id);
    setSessionId(id);
    setMessages([]);
    refreshSessions();
  };

  const selectSession = (id: string) => setSessionId(id);

  const startSidebarResize = useCallback((e: React.MouseEvent<HTMLButtonElement>) => {
    e.preventDefault();
    setSidebarCollapsed(false);
    document.body.classList.add("is-resizing");

    const onMove = (event: MouseEvent) => {
      const next = Math.min(460, Math.max(220, event.clientX));
      setSidebarWidth(next);
    };
    const onUp = () => {
      document.body.classList.remove("is-resizing");
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };

    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }, []);

  return (
    <div
      className={`layout ${sidebarCollapsed ? "sidebar-is-collapsed" : ""}`}
      style={{ gridTemplateColumns: `${sidebarCollapsed ? 56 : sidebarWidth}px 1fr` }}
    >
      <Sidebar
        documents={documents}
        sessions={sessions}
        activeSession={sessionId}
        collapsed={sidebarCollapsed}
        onToggleCollapsed={() => setSidebarCollapsed((v) => !v)}
        onRefreshDocs={refreshDocs}
        onRefreshSessions={refreshSessions}
        onSelectSession={selectSession}
        onNewSession={newSession}
        maxUploadSize={config?.max_upload_size}
      />
      {!sidebarCollapsed && (
        <button
          className="pane-resizer sidebar-resizer"
          aria-label="Kéo để đổi độ rộng thanh bên"
          onMouseDown={startSidebarResize}
        />
      )}
      <main className="main">
        <header className="topbar">
          <div>
            <span className={`dot ${config?.openai_configured ? "on" : "off"}`} />
            {config
              ? `${config.llm_model} · embed: ${config.embed_model}` +
                (config.use_reranker ? " · reranker bật" : "") +
                (config.enable_drag ? " · DRAG bật" : "")
              : "Đang tải…"}
          </div>
          <div className="muted small">Hybrid: BM25 + turbovec (TurboQuant) + RRF</div>
        </header>
        <ChatPanel
          sessionId={sessionId}
          messages={messages}
          setMessages={setMessages}
          onSession={handleSession}
          onOpenCitation={setCitation}
          hasDocs={documents.length > 0}
          openaiReady={!!config?.openai_configured}
          configLoaded={config !== null}
          messagesLoading={messagesLoading}
        />
      </main>

      {citation && (
        <div className="drawer-overlay" onClick={() => setCitation(null)}>
          <div className="drawer" onClick={(e) => e.stopPropagation()}>
            <div className="drawer-head">
              <div>
                <div className="drawer-title">Nguồn [{citation.label}]</div>
                <div className="muted small">
                  {citation.doc_title}
                  {citation.page ? ` · trang ${citation.page}` : ""}
                  {citation.section ? ` · ${citation.section}` : ""}
                </div>
              </div>
              <button className="icon-btn" onClick={() => setCitation(null)}>
                ✕
              </button>
            </div>
            <div className="drawer-meta">
              <span className="chip">chunk #{citation.chunk_id}</span>
              <span className="chip score">score {citation.score}</span>
              <span className="chip">{citation.doc_source}</span>
            </div>
            <div className="drawer-body">{citation.text}</div>
          </div>
        </div>
      )}
    </div>
  );
}
