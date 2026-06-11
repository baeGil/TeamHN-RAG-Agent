import type { ConflictReport } from "../lib/types";

function typeLabel(type: string) {
  if (type === "temporal") return "Mâu thuẫn thời gian";
  if (type === "factual") return "Mâu thuẫn sự kiện";
  if (type === "opinion") return "Mâu thuẫn quan điểm";
  return type;
}

function percent(x: number) {
  return `${Math.round(x * 100)}%`;
}

export default function ConflictReportCard({
  report,
  compact = false,
}: {
  report?: ConflictReport | null;
  compact?: boolean;
}) {
  if (!report || !report.enabled) return null;

  const pairs = report.conflict_pairs || [];

  if (!report.has_conflict || pairs.length === 0) {
    return (
      <div className="conflict-card conflict-ok">
        <div className="conflict-title">✅ ConflictRAG: Không phát hiện mâu thuẫn</div>
        <div className="conflict-sub">
          Đã kiểm tra {report.num_pairs ?? 0} cặp đoạn trích
          {report.duration_ms ? ` · ${report.duration_ms}ms` : ""}.
        </div>
      </div>
    );
  }

  return (
    <div className="conflict-card conflict-warn">
      <div className="conflict-title">
        ⚠️ ConflictRAG: Phát hiện {pairs.length} cặp mâu thuẫn
      </div>
      <div className="conflict-sub">
        Đã kiểm tra {report.num_pairs ?? 0} cặp từ {report.num_documents ?? "?"} đoạn trích
        {report.threshold !== undefined ? ` · threshold ${report.threshold}` : ""}
        {report.duration_ms ? ` · ${report.duration_ms}ms` : ""}.
      </div>

      <div className="conflict-list">
        {pairs.slice(0, compact ? 2 : 5).map((p, idx) => (
          <details key={idx} className="conflict-pair" open={idx === 0 && !compact}>
            <summary>
              <span className="conflict-badge">{typeLabel(p.type_label)}</span>
              <span>
                [{p.doc_i_label ?? p.doc_i_id}] ↔ [{p.doc_j_label ?? p.doc_j_id}]
              </span>
              <span className="conflict-prob">{percent(p.conflict_probability)}</span>
            </summary>

            {!compact && (
              <div className="conflict-preview-grid">
                <div className="conflict-preview">
                  <div className="conflict-preview-head">
                    Đoạn [{p.doc_i_label ?? p.doc_i_id}]
                    {p.doc_i_title ? ` · ${p.doc_i_title}` : ""}
                    {p.doc_i_page ? ` · tr.${p.doc_i_page}` : ""}
                  </div>
                  <div>{p.doc_i_preview || "Không có preview."}</div>
                </div>

                <div className="conflict-preview">
                  <div className="conflict-preview-head">
                    Đoạn [{p.doc_j_label ?? p.doc_j_id}]
                    {p.doc_j_title ? ` · ${p.doc_j_title}` : ""}
                    {p.doc_j_page ? ` · tr.${p.doc_j_page}` : ""}
                  </div>
                  <div>{p.doc_j_preview || "Không có preview."}</div>
                </div>
              </div>
            )}
          </details>
        ))}
      </div>

      {pairs.length > (compact ? 2 : 5) && (
        <div className="conflict-sub">Còn {pairs.length - (compact ? 2 : 5)} cặp khác.</div>
      )}
    </div>
  );
}