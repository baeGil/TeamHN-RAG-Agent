import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Citation, Message, TraceEvent } from "../lib/types";
import { streamChat } from "../lib/api";
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
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, liveAnswer, liveTrace]);

  const send = async () => {
    const msg = input.trim();
    if (!msg || streaming) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", content: msg, citations: [], trace: [] }]);
    setStreaming(true);
    setLiveTrace([]);
    setLiveAnswer("");

    let answer = "";
    let citations: Citation[] = [];
    const trace: TraceEvent[] = [];

    await streamChat(sessionId, msg, {
      onEvent: (type, data) => {
        if (type === "session") {
          onSession(data.session_id);
        } else if (type === "token") {
          answer += data.text;
          setLiveAnswer(answer);
        } else if (type === "final") {
          answer = data.answer;
          citations = data.citations || [];
        } else {
          trace.push({ type, data });
          setLiveTrace([...trace]);
        }
      },
      onError: (m) => {
        answer = answer || `⚠️ Lỗi: ${m}`;
      },
      onDone: () => {
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: answer, citations, trace },
        ]);
        setStreaming(false);
        setLiveAnswer("");
        setLiveTrace([]);
      },
    });
  };

  const lastAssistant = useMemo(
    () => [...messages].reverse().find((m) => m.role === "assistant"),
    [messages]
  );
  const graphEvents = streaming ? liveTrace : lastAssistant?.trace || [];
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

              {messages.length === 0 && !streaming && !messagesLoading && (
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
                    {m.role === "assistant" && m.trace && m.trace.length > 0 && m.trace.some((e) => e.type !== "thinking") && (
                      <AgentTrace events={m.trace} />
                    )}
                    {m.role === "assistant" ? (
                      <Markdown content={m.content} citations={m.citations || []} onCite={(label) => {
                        const c = (m.citations || []).find((x) => x.label === label);
                        if (c) onOpenCitation(c);
                      }} />
                    ) : (
                      <div className="user-text">{m.content}</div>
                    )}
                    {m.role === "assistant" && m.citations && m.citations.length > 0 && (
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
                    {liveAnswer ? (
                      <Markdown content={liveAnswer} citations={[]} />
                    ) : (
                      <div className="typing">●●●</div>
                    )}
                  </div>
                </div>
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
                    send();
                  }
                }}
              />
              <button className="btn primary" disabled={streaming || !input.trim()} onClick={send}>
                {streaming ? "…" : "Gửi"}
              </button>
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
            <AgentGraph events={graphEvents} liveAnswer={graphAnswer} done={!streaming && !!lastAssistant} />
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
