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
  const [sessionId, setSessionId] = useState<string | null>(
    localStorage.getItem(LS_KEY)
  );
  const [messages, setMessages] = useState<Message[]>([]);
  const [messagesLoading, setMessagesLoading] = useState(false);
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

  return (
    <div className="layout">
      <Sidebar
        documents={documents}
        sessions={sessions}
        activeSession={sessionId}
        onRefreshDocs={refreshDocs}
        onRefreshSessions={refreshSessions}
        onSelectSession={selectSession}
        onNewSession={newSession}
      />
      <main className="main">
        <header className="topbar">
          <div>
            <span className={`dot ${config?.openai_configured ? "on" : "off"}`} />
            {config
              ? `${config.llm_model} · embed: ${config.embed_model}` +
                (config.use_reranker ? " · reranker bật" : "")
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
