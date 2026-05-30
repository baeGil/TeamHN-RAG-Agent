import { useEffect, useRef, useState } from "react";
import type { Citation, Message, TraceEvent } from "../lib/types";
import { streamChat } from "../lib/api";
import Markdown from "./Markdown";
import AgentTrace from "./AgentTrace";

interface Props {
  sessionId: string | null;
  messages: Message[];
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
  onSession: (id: string) => void;
  onOpenCitation: (c: Citation) => void;
  hasDocs: boolean;
  openaiReady: boolean;
}

export default function ChatPanel({
  sessionId,
  messages,
  setMessages,
  onSession,
  onOpenCitation,
  hasDocs,
  openaiReady,
}: Props) {
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [liveTrace, setLiveTrace] = useState<TraceEvent[]>([]);
  const [liveAnswer, setLiveAnswer] = useState("");
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

  return (
    <div className="chat">
      <div className="messages" ref={scrollRef}>
        {messages.length === 0 && !streaming && (
          <div className="empty-state">
            <h2>Hỏi đáp dựa trên tài liệu của bạn</h2>
            <p className="muted">
              {hasDocs
                ? "Đặt câu hỏi — Agent sẽ trả lời kèm trích dẫn nguồn, hiển thị quá trình suy luận theo thời gian thực."
                : "Hãy tải lên một tệp PDF hoặc thêm URL ở thanh bên để bắt đầu."}
            </p>
            {!openaiReady && (
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
              {m.role === "assistant" && m.trace.length > 0 && (
                <AgentTrace events={m.trace} />
              )}
              {m.role === "assistant" ? (
                <Markdown content={m.content} citations={m.citations} onCite={(label) => {
                  const c = m.citations.find((x) => x.label === label);
                  if (c) onOpenCitation(c);
                }} />
              ) : (
                <div className="user-text">{m.content}</div>
              )}
              {m.role === "assistant" && m.citations.length > 0 && (
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
    </div>
  );
}
