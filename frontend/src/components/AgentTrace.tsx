import { useState } from "react";
import type { TraceEvent } from "../lib/types";

const ROUTE_LABEL: Record<string, string> = {
  simple: "Single-hop (đường nhanh)",
  complex: "Multi-hop (agent đầy đủ)",
  no_retrieval: "Không cần tra cứu",
  not_found: "Không tìm thấy",
};

function Icon({ type }: { type: string }) {
  const map: Record<string, string> = {
    route: "🧭",
    plan: "🗺️",
    subquestion: "❓",
    retrieved: "🔎",
    distill: "⚗️",
    verify: "✓",
    synthesize: "🧩",
    final: "🏁",
    replan: "🔄",
    sufficiency: "📊",
    converged: "✅",
    max_iters: "⏹",
    early_stop: "⚡",
    verify_answer: "🔍",
  };
  return <span className="trace-icon">{map[type] || "•"}</span>;
}

export default function AgentTrace({
  events,
  live,
}: {
  events: TraceEvent[];
  live?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const visible = events.filter((e) => e.type !== "thinking");
  if (!visible.length) return null;

  return (
    <div className="trace">
      <div className="trace-head" onClick={() => setOpen(!open)}>
        <span>{live ? "Đang suy luận…" : "Quá trình suy luận của Agent"}</span>
        <span className="trace-toggle">{open ? "−" : "+"}</span>
      </div>
      {open && (
        <div className="trace-body">
          {visible.map((e, i) => (
            <div key={i} className="trace-item">
              <Icon type={e.type} />
              <div className="trace-content">
                {e.type === "route" && (
                  <div>
                    <b>Định tuyến:</b> {ROUTE_LABEL[e.data.route] || e.data.route}
                    <div className="trace-sub">{e.data.reason}</div>
                  </div>
                )}
                {e.type === "plan" && (
                  <div>
                    <b>Kế hoạch ({e.data.subquestions.length} bước):</b>
                    <ol className="trace-plan">
                      {e.data.subquestions.map((s: string, j: number) => (
                        <li key={j}>{s}</li>
                      ))}
                    </ol>
                  </div>
                )}
                {e.type === "replan" && (
                  <div>
                    <b>Lập lại kế hoạch (vòng {e.data.iteration}):</b>
                    <ol className="trace-plan">
                      {e.data.subquestions.map((s: string, j: number) => (
                        <li key={j}>{s}</li>
                      ))}
                    </ol>
                  </div>
                )}
                {e.type === "subquestion" && (
                  <div>
                    <b>Bước {e.data.index + 1}:</b> {e.data.subquestion}
                  </div>
                )}
                {e.type === "retrieved" && (
                  <div>
                    <b>Truy hồi {e.data.chunks.length} đoạn</b>
                    <div className="trace-chunks">
                      {e.data.chunks.map((c: any) => (
                        <div key={c.chunk_id} className="trace-chunk">
                          <span className="chip">#{c.chunk_id}</span>
                          {c.page ? <span className="chip">tr.{c.page}</span> : null}
                          <span className="chip score">score {c.score}</span>
                          <span className="trace-preview">{c.preview}…</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {e.type === "distill" && (
                  <div>
                    <b>Chắt lọc:</b>{" "}
                    {e.data.relevant ? (
                      <span>{e.data.note}</span>
                    ) : (
                      <span className="muted">Không liên quan</span>
                    )}
                  </div>
                )}
                {e.type === "verify" && (
                  <div>
                    <b>Kiểm chứng:</b>{" "}
                    <span className={e.data.grounded ? "ok" : "warn"}>
                      {e.data.grounded ? "Bám nguồn ✓" : "Chưa chắc chắn ⚠"}
                    </span>
                    {e.data.reason ? <span className="trace-sub"> {e.data.reason}</span> : null}
                  </div>
                )}
                {e.type === "sufficiency" && (
                  <div>
                    <b>Kiểm tra độ đủ:</b>{" "}
                    <span className={e.data.sufficient ? "ok" : "warn"}>
                      {e.data.sufficient ? "Đủ thông tin ✓" : "Chưa đủ thông tin ⚠"}
                    </span>
                    {e.data.reason ? <span className="trace-sub"> {e.data.reason}</span> : null}
                  </div>
                )}
                {e.type === "converged" && (
                  <div>
                    <b>Tất cả bước đã bám nguồn ✓</b> (vòng {e.data.iteration + 1})
                  </div>
                )}
                {e.type === "max_iters" && (
                  <div>
                    <b>Đạt giới hạn lặp ⏹</b> — {e.data.failed?.length || 0} bước chưa bám nguồn
                  </div>
                )}
                {e.type === "early_stop" && (
                  <div>
                    <b>Dừng sớm ⚡</b> — {e.data.reason}
                  </div>
                )}
                {e.type === "verify_answer" && (
                  <div>
                    <b>Kiểm chứng câu trả lời:</b>{" "}
                    <span className={e.data.grounded ? "ok" : "warn"}>
                      {e.data.grounded ? "Bám nguồn ✓" : "Chưa chắc chắn ⚠"}
                    </span>
                    {e.data.reason ? <span className="trace-sub"> {e.data.reason}</span> : null}
                  </div>
                )}
                {e.type === "synthesize" && (
                  <div>
                    <b>Tổng hợp câu trả lời</b> từ {e.data.n_context} đoạn ngữ cảnh.
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}