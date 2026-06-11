import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Citation, Message, TraceEvent } from "../lib/types";
import { api, streamChat } from "../lib/api";
import { conflictLabel, conflictSentence, normalizeConflictType } from "../lib/conflicts";
import Markdown from "./Markdown";
import AgentTrace from "./AgentTrace";
import AgentGraph from "./AgentGraph";

interface Props {
  sessionId: string | null;
  messages: Message[];
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
  onSession: (id: string) => void;
  onOpenCitation: (c: Citation) => void;
  hasDocs: boolean;
  openaiReady: boolean;
  configLoaded: boolean;
  messagesLoading: boolean;
}

function normalizeMessage(m: any): Message {
  return {
    ...m,
    citations: m.citations || [],
    trace: m.trace || [],
    status: m.status || "complete",
    error_message: m.error_message || null,
  };
}

function getConflictEvent(events?: TraceEvent[]) {
  return (events || []).find((e) => e.type === "conflict")?.data || null;
}

function ConflictStrip({ conflict }: { conflict: any }) {
  if (!conflict) return null;
  const type = normalizeConflictType(conflict.conflict_type);
  const isClear = type === "no_conflict";
  return (
    <div className={`conflict-strip ${isClear ? "clear" : "attention"}`}>
      <div className="conflict-kicker">DRAG conflict</div>
      <div className="conflict-main">
        <b>{conflictLabel(type)}</b>
        {typeof conflict.confidence === "number" ? (
          <span className="muted small"> {Math.round(conflict.confidence * 100)}%</span>
        ) : null}
      </div>
      <div className="conflict-sentence">
        {conflictSentence(conflict.conflict_type, conflict.display_sentence)}
      </div>
    </div>
  );
}

export default function ChatPanel({
  sessionId,
  messages,
  setMessages,
  onSession,
  onOpenCitation,
  hasDocs,
  openaiReady,
  configLoaded,
  messagesLoading,
}: Props) {
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [liveTrace, setLiveTrace] = useState<TraceEvent[]>([]);
  const [liveAnswer, setLiveAnswer] = useState("");
  const [agentCollapsed, setAgentCollapsed] = useState(false);
  const [agentWidth, setAgentWidth] = useState(360);
  const [polling, setPolling] = useState(false);
  const [chatError, setChatError] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, liveAnswer, liveTrace]);

  useEffect(() => {
    if (!sessionId) {
      setPolling(false);
      return;
    }
    const hasProcessing = messages.some(
      (m) => m.role === "assistant" && m.status === "processing"
    );
    if (hasProcessing && !streaming && !polling) {
      setPolling(true);
    }
    if (!hasProcessing && polling) {
      setPolling(false);
    }
  }, [sessionId, messages, streaming, polling]);

  useEffect(() => {
    if (!polling || !sessionId) return;
    const poll = async () => {
      try {
        const s = await api.getSession(sessionId);
        const msgs = (s.messages || []).map(normalizeMessage);
        setMessages(msgs);
        const stillHasProcessing = msgs.some(
          (m: Message) => m.role === "assistant" && m.status === "processing"
        );
        if (!stillHasProcessing) {
          setPolling(false);
          setLiveAnswer("");
          setLiveTrace([]);
        }
      } catch {
        setPolling(false);
        setLiveAnswer("");
        setLiveTrace([]);
      }
    };
    pollRef.current = setInterval(poll, 2000);
    poll();
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [polling, sessionId, setMessages]);

  const cancelChat = async () => {
    if (!sessionId) return;
    // Abort the HTTP request
    abortRef.current?.abort();
    // Tell backend to stop processing
    try {
      await api.cancelChat(sessionId);
    } catch {
      /* ignore */
    }
    setStreaming(false);
    // Mark the processing message as cancelled in UI immediately
    setMessages((prev) => {
      const processingIdx = prev.findIndex(
        (m) => m.role === "assistant" && m.status === "processing"
      );
      if (processingIdx >= 0) {
        const updated = [...prev];
        updated[processingIdx] = {
          ...updated[processingIdx],
          status: "cancelled",
          error_message: "Đã hủy.",
        };
        return updated;
      }
      // Nếu đang streaming mà chưa có message assistant trong state,
      // thêm placeholder cancelled để UI hiện "Đã hủy" ngay lập tức
      if (streaming) {
        return [
          ...prev,
          {
            role: "assistant",
            content: liveAnswer,
            citations: [],
            trace: liveTrace,
            status: "cancelled",
            error_message: "Đã hủy.",
          },
        ];
      }
      return prev;
    });
    setLiveAnswer("");
    setLiveTrace([]);
  };

  const send = async () => {
    const msg = input.trim();
    if (!msg || streaming) return;
    setInput("");
    setChatError("");
    setMessages((m) => [...m, { role: "user", content: msg, citations: [], trace: [] }]);
    setStreaming(true);
    setLiveTrace([]);
    setLiveAnswer("");

    let answer = "";
    let citations: Citation[] = [];
    const trace: TraceEvent[] = [];
    let gotFinal = false;

    abortRef.current = new AbortController();

    try {
      await streamChat(sessionId, msg, {
        onEvent: (type, data) => {
          if (type === "session") {
            onSession(data.session_id);
          } else if (type === "token") {
            answer += data.text;
            setLiveAnswer(answer);
          } else if (type === "final") {
            gotFinal = true;
            answer = data.answer;
            citations = data.citations || [];
          } else {
            trace.push({ type, data });
            setLiveTrace([...trace]);
          }
        },
        onError: (m) => {
          answer = answer || `⚠️ Lỗi: ${m}`;
          setChatError(m);
        },
        onDone: () => {
          setStreaming(false);
          if (gotFinal) {
            // Normal completion — update message and clear live state
            setMessages((prev) => {
              const processingIdx = prev.findIndex(
                (m) => m.role === "assistant" && m.status === "processing"
              );
              if (processingIdx >= 0) {
                const updated = [...prev];
                updated[processingIdx] = {
                  ...updated[processingIdx],
                  content: answer,
                  citations,
                  trace,
                  status: "complete",
                };
                return updated;
              }
              return [
                ...prev,
                { role: "assistant", content: answer, citations, trace, status: "complete" },
              ];
            });
            setLiveAnswer("");
            setLiveTrace([]);
          }
          // If !gotFinal (interrupted): keep liveTrace/liveAnswer for AgentGraph resume
        },
      }, abortRef.current.signal);
    } catch (e: any) {
      if (e.name === "AbortError") {
        // User cancelled — already handled by cancelChat
        return;
      }
      setChatError(e.message || String(e));
      setStreaming(false);
      // DON'T clear liveAnswer/liveTrace — keep them for AgentGraph resume
      const s = await api.getSession(sessionId || "").catch(() => null);
      if (s) {
        setMessages((s.messages || []).map(normalizeMessage));
      }
    }
  };

  const lastAssistant = useMemo(
    () => [...messages].reverse().find((m) => m.role === "assistant"),
    [messages]
  );
  const hasProcessing = messages.some(
    (m) => m.role === "assistant" && m.status === "processing"
  );
  // graphEvents priority:
  // 1. streaming → liveTrace (realtime events from current SSE)
  // 2. hasProcessing (interrupted/reload) → lastAssistant.trace from DB (persisted)
  //    only fall back to synthetic router if absolutely nothing available
  // 3. done → lastAssistant.trace
  let _graphEvents: TraceEvent[];
  if (streaming) {
    _graphEvents = liveTrace.length > 0 ? liveTrace : [{ type: "thinking" as const, data: { node: "router" } }];
  } else if (hasProcessing) {
    const dbTrace = lastAssistant?.trace || [];
    _graphEvents = dbTrace.length > 0 ? dbTrace : [{ type: "thinking" as const, data: { node: "router" } }];
  } else {
    _graphEvents = lastAssistant?.trace || [];
  }
  const graphEvents = _graphEvents;
  const graphAnswer = streaming ? liveAnswer : lastAssistant?.content || "";

  const startAgentResize = useCallback((e: React.MouseEvent<HTMLButtonElement>) => {
    e.preventDefault();
    setAgentCollapsed(false);
    document.body.classList.add("is-resizing");
    const startX = e.clientX;
    const startWidth = agentWidth;

    const onMove = (event: MouseEvent) => {
      const next = Math.min(560, Math.max(280, startWidth + startX - event.clientX));
      setAgentWidth(next);
    };
    const onUp = () => {
      document.body.classList.remove("is-resizing");
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };

    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }, [agentWidth]);

  const chatColumns = `minmax(320px, 1fr) ${agentCollapsed ? 56 : agentWidth}px`;
  const isProcessing = streaming || polling;

  return (
    <div
      className={`chat ${agentCollapsed ? "agent-is-collapsed" : ""}`}
      style={{ gridTemplateColumns: chatColumns }}
    >
      <section className="conversation-panel">
        <div className="messages" ref={scrollRef}>
              {messagesLoading && messages.length === 0 && (
                <div className="empty-state">
                  <h2>Đang tải…</h2>
                </div>
              )}

              {messages.length === 0 && !isProcessing && !messagesLoading && (
                <div className="empty-state">
                  <h2>Hỏi đáp dựa trên tài liệu của bạn</h2>
                  <p className="muted">
                    {hasDocs
                      ? "Đặt câu hỏi — Agent sẽ trả lời kèm trích dẫn nguồn, hiển thị quá trình suy luận theo thời gian thực."
                      : "Hãy tải lên một tệp PDF hoặc thêm URL ở thanh bên để bắt đầu."}
                  </p>
                  {configLoaded && !openaiReady && (
                    <div className="error-box">
                      Chưa cấu hình OPENAI_API_KEY trong <code>backend/.env</code>.
                    </div>
                  )}
                </div>
              )}

              {messages.map((m, i) => (
                <div key={i} className={`msg ${m.role}`}>
                  <div className="avatar">{m.role === "user" ? "🧑" : "🤖"}</div>
                  <div className="bubble">
                    {m.role === "assistant" && m.status === "processing" && !streaming && (
                      <>
                        <AgentTrace events={m.trace || []} live />
                        <div className="bubble-actions">
                          <div className="typing">●●●</div>
                          <button className="btn danger small" onClick={cancelChat}>Hủy</button>
                        </div>
                      </>
                    )}
                    {m.role === "assistant" && m.status === "cancelled" && (
                      <div className="error-box">🛑 Đã hủy.</div>
                    )}
                    {m.role === "assistant" && m.status === "failed" && m.error_message && (
                      <div className="error-box">⚠️ {m.error_message}</div>
                    )}
                    {m.role === "assistant" && (
                      <ConflictStrip conflict={getConflictEvent(m.trace)} />
                    )}
                    {m.role === "assistant" && m.content ? (
                      <Markdown content={m.content} citations={m.citations || []} onCite={(label) => {
                        const c = (m.citations || []).find((x) => x.label === label);
                        if (c) onOpenCitation(c);
                      }} />
                    ) : m.role === "assistant" ? null : (
                      <div className="user-text">{m.content}</div>
                    )}
                    {m.role === "assistant" && m.citations && m.citations.length > 0 && m.status === "complete" && (
                      <div className="sources">
                        <div className="sources-title">Nguồn trích dẫn</div>
                        <div className="sources-list">
                          {m.citations.map((c) => (
                            <button
                              key={c.label}
                              className={`source-chip ${c.cited ? "used" : ""}`}
                              onClick={() => onOpenCitation(c)}
                              title={c.text.slice(0, 120)}
                            >
                              [{c.label}] {c.doc_title}
                              {c.page ? ` · tr.${c.page}` : ""}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {streaming && (
                <div className="msg assistant">
                  <div className="avatar">🤖</div>
                  <div className="bubble">
                    <AgentTrace events={liveTrace} live />
                    <ConflictStrip conflict={getConflictEvent(liveTrace)} />
                    {liveAnswer ? (
                      <Markdown content={liveAnswer} citations={[]} />
                    ) : (
                      <div className="typing">●●●</div>
                    )}
                    <div className="bubble-actions">
                      <button className="btn danger small" onClick={cancelChat}>Hủy</button>
                    </div>
                  </div>
                </div>
              )}

              {chatError && !streaming && (
                <div className="error-box">{chatError}</div>
              )}
        </div>

        <div className="composer">
              <textarea
                className="composer-input"
                placeholder="Nhập câu hỏi của bạn…"
                value={input}
                rows={1}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    if (streaming) {
                      cancelChat();
                    } else {
                      send();
                    }
                  }
                }}
              />
              {streaming ? (
                <button className="btn danger" onClick={cancelChat}>
                  ✕ Hủy
                </button>
              ) : (
                <button className="btn primary" disabled={isProcessing || !input.trim()} onClick={send}>
                  {isProcessing ? "…" : "Gửi"}
                </button>
              )}
        </div>
      </section>

      <aside className={`agent-panel ${agentCollapsed ? "panel-collapsed" : ""}`}>
        {agentCollapsed ? (
          <button
            className="collapse-rail-btn vertical"
            title="Mở Agent Graph"
            onClick={() => setAgentCollapsed(false)}
            aria-label="Mở Agent Graph"
          >
            Graph
          </button>
        ) : (
          <>
            <button
              className="pane-resizer agent-resizer"
              aria-label="Kéo để đổi độ rộng Agent Graph"
              onMouseDown={startAgentResize}
            />
            <div className="panel-head">
              <div className="agent-panel-head">Agent Graph</div>
              <button
                className="panel-toggle"
                title="Thu gọn Agent Graph"
                onClick={() => setAgentCollapsed(true)}
                aria-label="Thu gọn Agent Graph"
              >
                ▶
              </button>
            </div>
            <AgentGraph events={graphEvents} liveAnswer={graphAnswer} done={!isProcessing && !!lastAssistant} />
            <div className="agent-panel-legend">
              <div><span className="legend-dot active" /> <b>Đang xử lý</b></div>
              <div><span className="legend-dot done" /> <b>Hoàn thành</b></div>
              <div><span className="legend-dot error" /> <b>Lỗi</b></div>
              <div><span className="legend-dot idle" /> <b>Chờ</b></div>
              <div><span className="legend-dot skipped" /> <b>Bỏ qua</b></div>
            </div>
          </>
        )}
      </aside>
    </div>
  );
}
