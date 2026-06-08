import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import type { AppSettings } from "../lib/types";

// ─── Default values per tab ─────────────────────────────────────────────────

const DEFAULTS: Record<string, Record<string, any>> = {
  connection: {
    openai_api_key: "",
    openai_base_url: "",
  },
  parsing: {
    parser: "pymupdf",
    vlm_parse: "auto",
    vlm_model: "gpt-4o-mini",
    mineru_cmd: "",
    reducto_parse: "off",
    reducto_api_key: "",
    reducto_chunk_mode: "page_sections",
    reducto_chunk_size: 1200,
    reducto_filter_blocks: "Header,Footer,Page Number",
    reducto_table_format: "dynamic",
    chunk_max_chars: 1000,
    chunk_overlap: 200,
  },
  indexing: {
    embed_model: "text-embedding-3-small",
    embed_dim: "",
    enable_doc_summary: true,
    doc_summary_chars: 24000,
    doc_summary_model: "",
    enable_section_summary: true,
    section_summary_chars: 2000,
  },
  retrieval: {
    bm25_top_k: 30,
    dense_top_k: 30,
    rrf_k: 60,
    use_reranker: true,
    reranker_model: "BAAI/bge-reranker-v2-m3",
    rerank_top_n: 20,
    final_top_k: 5,
    use_hyde: false,
    use_rse: true,
    rse_irrelevant_penalty: 0.2,
    rse_max_segment_chunks: 15,
    rse_overall_max_chunks: 30,
    rse_window_extension: 2,
    rse_chunk_length_adjustment: true,
    min_chunk_chars: 50,
    complex_ctx_limit: 8,
  },
  generation: {
    llm_model: "gpt-4o-mini",
    llm_model_fast: "gpt-4o-mini",
    enable_replan: true,
    max_replan_iters: 3,
    enable_sufficiency: true,
    enable_answer_verify: true,
    enable_answer_verify_simple: false,
    enable_answer_verify_complex: true,
    max_answer_regenerations: 1,
  },
  memory: {
    enable_summarization: true,
    summary_threshold: 12,
    history_window: 6,
    summary_model: "",
  },
};

// ─── Tooltip ─────────────────────────────────────────────────────────────────

function Tooltip({ text }: { text: string }) {
  const [show, setShow] = useState(false);
  const ref = useRef<HTMLButtonElement>(null);
  return (
    <span className="sp-tooltip-wrap">
      <button
        ref={ref}
        className="sp-tooltip-btn"
        type="button"
        aria-label="Giải thích"
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
        onFocus={() => setShow(true)}
        onBlur={() => setShow(false)}
      >
        ?
      </button>
      {show && <div className="sp-tooltip-box">{text}</div>}
    </span>
  );
}

// ─── Field primitives ─────────────────────────────────────────────────────────

interface FieldProps {
  label: string;
  tip: string;
  children: React.ReactNode;
}
function Field({ label, tip, children }: FieldProps) {
  return (
    <div className="sp-field">
      <div className="sp-field-label">
        <span>{label}</span>
        <Tooltip text={tip} />
      </div>
      <div className="sp-field-control">{children}</div>
    </div>
  );
}

interface ToggleProps {
  label: string;
  tip: string;
  value: boolean;
  onChange: (v: boolean) => void;
  children?: React.ReactNode;
}
function Toggle({ label, tip, value, onChange, children }: ToggleProps) {
  return (
    <div className="sp-toggle-group">
      <div className="sp-field">
        <div className="sp-field-label">
          <span>{label}</span>
          <Tooltip text={tip} />
        </div>
        <button
          type="button"
          className={`sp-toggle ${value ? "on" : "off"}`}
          onClick={() => onChange(!value)}
          aria-pressed={value}
        >
          <span className="sp-toggle-thumb" />
          <span className="sp-toggle-label">{value ? "Bật" : "Tắt"}</span>
        </button>
      </div>
      {value && children && <div className="sp-nested">{children}</div>}
    </div>
  );
}

interface SelectProps {
  label: string;
  tip: string;
  value: string;
  options: { value: string; label: string }[];
  onChange: (v: string) => void;
}
function Select({ label, tip, value, options, onChange }: SelectProps) {
  return (
    <Field label={label} tip={tip}>
      <select
        className="sp-select"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </Field>
  );
}

interface SliderProps {
  label: string;
  tip: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  unit?: string;
  onChange: (v: number) => void;
}
function Slider({ label, tip, value, min, max, step = 1, unit = "", onChange }: SliderProps) {
  return (
    <Field label={label} tip={tip}>
      <div className="sp-slider-row">
        <input
          type="range"
          className="sp-slider"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
        />
        <input
          type="number"
          className="sp-slider-num"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => {
            const v = Number(e.target.value);
            if (!isNaN(v)) onChange(Math.min(max, Math.max(min, v)));
          }}
        />
        {unit && <span className="sp-unit">{unit}</span>}
      </div>
    </Field>
  );
}

interface TextInputProps {
  label: string;
  tip: string;
  value: string;
  placeholder?: string;
  masked?: boolean;
  onChange: (v: string) => void;
}
function TextInput({ label, tip, value, placeholder, masked, onChange }: TextInputProps) {
  return (
    <Field label={label} tip={tip}>
      <input
        type={masked ? "password" : "text"}
        className="sp-text"
        value={value}
        placeholder={placeholder}
        autoComplete={masked ? "new-password" : undefined}
        onChange={(e) => onChange(e.target.value)}
      />
    </Field>
  );
}

// ─── Tabs ─────────────────────────────────────────────────────────────────────

type TabKey = "connection" | "parsing" | "indexing" | "retrieval" | "generation" | "memory";
const TABS: { key: TabKey; label: string; icon: string }[] = [
  { key: "connection", label: "Kết nối", icon: "🔌" },
  { key: "parsing", label: "Parsing", icon: "📄" },
  { key: "indexing", label: "Indexing", icon: "🗂" },
  { key: "retrieval", label: "Retrieval", icon: "🔍" },
  { key: "generation", label: "Sinh văn bản", icon: "✍" },
  { key: "memory", label: "Bộ nhớ", icon: "💾" },
];

// ─── Deep helpers ─────────────────────────────────────────────────────────────

function deepEqual(a: any, b: any): boolean {
  if (a === b) return true;
  if (typeof a !== typeof b) return false;
  if (typeof a !== "object" || a === null || b === null) return false;
  const ka = Object.keys(a);
  const kb = Object.keys(b);
  if (ka.length !== kb.length) return false;
  return ka.every((k) => deepEqual(a[k], b[k]));
}

// Get only the fields that differ between draft and original
function getDirtyFields(
  draft: AppSettings,
  original: AppSettings,
): Record<string, any> {
  const dirty: Record<string, any> = {};
  for (const section of Object.keys(draft) as (keyof AppSettings)[]) {
    for (const key of Object.keys(draft[section]) as (keyof AppSettings[typeof section])[]) {
      const dVal = (draft as any)[section][key];
      const oVal = (original as any)[section][key];
      // Skip masked API keys that still contain *** (user didn't re-enter)
      if (typeof dVal === "string" && dVal.includes("***")) continue;
      if (!deepEqual(dVal, oVal)) {
        dirty[key as string] = dVal;
      }
    }
  }
  return dirty;
}

// Check if a specific section has any dirty fields
function isSectionDirty(
  draft: AppSettings,
  original: AppSettings,
  section: keyof AppSettings,
): boolean {
  for (const key of Object.keys(draft[section]) as (keyof AppSettings[typeof section])[]) {
    const dVal = (draft as any)[section][key];
    const oVal = (original as any)[section][key];
    if (typeof dVal === "string" && dVal.includes("***")) continue;
    if (!deepEqual(dVal, oVal)) return true;
  }
  return false;
}

// ─── Main component ───────────────────────────────────────────────────────────

interface Props {
  onClose: () => void;
}

export default function SettingsPanel({ onClose }: Props) {
  const [tab, setTab] = useState<TabKey>("connection");
  const [draft, setDraft] = useState<AppSettings | null>(null);
  const [original, setOriginal] = useState<AppSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [reindexWarning, setReindexWarning] = useState(false);

  useEffect(() => {
    api.getSettings().then((s) => {
      setDraft(s);
      setOriginal(JSON.parse(JSON.stringify(s))); // deep clone
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  // Close on Escape
  useEffect(() => {
    const h = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [onClose]);

  if (loading || !draft || !original) {
    return (
      <div className="sp-overlay" onClick={onClose}>
        <div className="sp-modal" onClick={(e) => e.stopPropagation()}>
          <div className="sp-loading">Đang tải cài đặt…</div>
        </div>
      </div>
    );
  }

  // Count dirty fields for save button
  const dirtyFields = getDirtyFields(draft, original);
  const dirtyCount = Object.keys(dirtyFields).length;
  const hasChanges = dirtyCount > 0;

  // Helper to patch draft
  function set<K extends keyof AppSettings>(section: K, key: keyof AppSettings[K], value: any) {
    setDraft((prev) => {
      if (!prev) return prev;
      return { ...prev, [section]: { ...prev[section], [key]: value } };
    });
    // Clear save message when user edits
    setSaveMsg(null);
  }

  // Reset current tab to defaults
  function resetTab() {
    const defaults = DEFAULTS[tab];
    if (!defaults) return;
    setDraft((prev) => {
      if (!prev) return prev;
      return { ...prev, [tab]: { ...defaults } };
    });
    setSaveMsg(null);
  }

  // Reset current tab to original (server) values
  function revertTab() {
    if (!original) return;
    setDraft((prev) => {
      if (!prev) return prev;
      return { ...prev, [tab]: { ...(original as any)[tab] } };
    });
    setSaveMsg(null);
  }

  async function save() {
    if (!draft || !original || !hasChanges) return;
    setSaving(true);
    setSaveMsg(null);
    try {
      const res = await api.updateSettings(dirtyFields);
      setReindexWarning(res.needs_reindex);
      setSaveMsg({ ok: true, text: `Đã lưu ${res.updated.length} cài đặt.` });
      // Update original to reflect what was saved
      const newOriginal = JSON.parse(JSON.stringify(draft));
      setOriginal(newOriginal);
    } catch (e: any) {
      setSaveMsg({ ok: false, text: `Lỗi: ${e.message || e}` });
    } finally {
      setSaving(false);
    }
  }

  const p = draft.parsing;
  const idx = draft.indexing;
  const ret = draft.retrieval;
  const gen = draft.generation;
  const mem = draft.memory;
  const tabDirty = isSectionDirty(draft, original, tab);

  return (
    <div className="sp-overlay" onClick={onClose}>
      <div className="sp-modal" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="sp-header">
          <span className="sp-title">⚙️ Cài đặt hệ thống</span>
          <button className="icon-btn" onClick={onClose}>✕</button>
        </div>

        {/* Tab bar */}
        <div className="sp-tabs">
          {TABS.map((t) => (
            <button
              key={t.key}
              type="button"
              className={`sp-tab ${tab === t.key ? "active" : ""} ${isSectionDirty(draft, original, t.key) ? "dirty" : ""}`}
              onClick={() => setTab(t.key)}
            >
              <span className="sp-tab-icon">{t.icon}</span>
              <span className="sp-tab-label">{t.label}</span>
            </button>
          ))}
        </div>

        {/* Tab-level actions */}
        <div className="sp-tab-actions">
          {tabDirty && (
            <button className="btn sp-revert-btn" type="button" onClick={revertTab}>
              ↩ Hoàn tác tab này
            </button>
          )}
          <button className="btn sp-reset-btn" type="button" onClick={resetTab}>
            ⏎ Khôi phục mặc định
          </button>
        </div>

        {/* Content */}
        <div className="sp-body">

          {/* ── Connection ─────────────────────────────────────────────────── */}
          {tab === "connection" && (
            <div className="sp-section">
              <div className="sp-section-title">Kết nối API</div>
              <TextInput
                label="OpenAI API Key"
                tip="Bắt buộc để dùng LLM, embedding và CCH. Lấy tại platform.openai.com/api-keys. Tương thích với Azure OpenAI, Groq, LM Studio và các endpoint OpenAI-compatible."
                value={draft.connection.openai_api_key}
                placeholder="sk-..."
                masked
                onChange={(v) => set("connection", "openai_api_key", v)}
              />
              <TextInput
                label="Base URL (tùy chọn)"
                tip="Để trống = dùng OpenAI mặc định (api.openai.com). Điền nếu dùng proxy hoặc API tương thích: ví dụ https://api.groq.com/openai/v1 cho Groq, hoặc http://localhost:11434/v1 cho Ollama."
                value={draft.connection.openai_base_url}
                placeholder="https://api.openai.com/v1"
                onChange={(v) => set("connection", "openai_base_url", v)}
              />
            </div>
          )}

          {/* ── Parsing ────────────────────────────────────────────────────── */}
          {tab === "parsing" && (
            <div className="sp-section">
              <div className="sp-section-title">Parser PDF</div>
              <Select
                label="Parser"
                tip="Cách trích xuất nội dung từ PDF. PyMuPDF: nhanh, miễn phí, tốt cho PDF có text rõ ràng. MinerU: OCR local dùng deep learning, tốt cho PDF scan hoặc có hình ảnh. Reducto: API có phí, tốt nhất cho bảng phức tạp và tài liệu khó."
                value={p.parser}
                options={[
                  { value: "pymupdf", label: "PyMuPDF (mặc định, miễn phí, nhanh)" },
                  { value: "mineru", label: "MinerU (local OCR, tốt cho scan)" },
                  { value: "reducto", label: "Reducto (API, tốt nhất cho bảng/scan)" },
                ]}
                onChange={(v) => set("parsing", "parser", v)}
              />

              {p.parser === "pymupdf" && (
                <>
                  <Select
                    label="VLM Parse"
                    tip="Dùng Vision LLM để đọc trang PDF. off: chỉ dùng text layer (nhanh nhất). auto: dùng VLM cho trang scan/ảnh không có text (khuyến nghị). on: luôn dùng VLM cho mọi trang (chính xác nhất, tốn token lúc ingest)."
                    value={p.vlm_parse}
                    options={[
                      { value: "off", label: "off — chỉ text layer" },
                      { value: "auto", label: "auto — VLM khi trang không có text (khuyến nghị)" },
                      { value: "on", label: "on — luôn dùng VLM" },
                    ]}
                    onChange={(v) => set("parsing", "vlm_parse", v as any)}
                  />
                  {(p.vlm_parse === "auto" || p.vlm_parse === "on") && (
                    <div className="sp-nested">
                      <TextInput
                        label="VLM Model"
                        tip="Model có khả năng vision để đọc ảnh/trang scan. gpt-4o-mini: tiết kiệm, đủ tốt. gpt-4o: chính xác hơn, đắt hơn. Gemini 1.5 Flash cũng hỗ trợ qua OpenAI-compatible endpoint."
                        value={p.vlm_model}
                        placeholder="gpt-4o-mini"
                        onChange={(v) => set("parsing", "vlm_model", v)}
                      />
                    </div>
                  )}
                </>
              )}

              {p.parser === "mineru" && (
                <div className="sp-nested">
                  <TextInput
                    label="Đường dẫn binary MinerU"
                    tip="Để trống = tự detect theo thứ tự: .venv_parser/bin/mineru → mineru trong PATH. Điền đường dẫn tuyệt đối nếu cài ở vị trí khác, ví dụ: /home/user/.venv_ml/bin/mineru"
                    value={p.mineru_cmd}
                    placeholder="(tự detect)"
                    onChange={(v) => set("parsing", "mineru_cmd", v)}
                  />
                </div>
              )}

              {p.parser === "reducto" && (
                <div className="sp-nested">
                  <TextInput
                    label="Reducto API Key"
                    tip="Lấy tại reducto.ai sau khi đăng ký. Mỗi trang PDF tốn 1-2 credits tùy chế độ."
                    value={p.reducto_api_key}
                    placeholder="rdk-..."
                    masked
                    onChange={(v) => set("parsing", "reducto_api_key", v)}
                  />
                  <Select
                    label="Chế độ Reducto"
                    tip="default: 1 credit/trang, chất lượng tốt, nhanh. agentic: 2 credits/trang, VLM review text và bảng, chất lượng tốt nhất cho tài liệu phức tạp."
                    value={p.reducto_parse === "off" ? "default" : p.reducto_parse}
                    options={[
                      { value: "default", label: "default — 1 credit/trang (khuyến nghị)" },
                      { value: "agentic", label: "agentic — 2 credits/trang, chất lượng tốt nhất" },
                    ]}
                    onChange={(v) => set("parsing", "reducto_parse", v as any)}
                  />
                  <Select
                    label="Chunk mode"
                    tip="page_sections: chia theo section trong trang, tốt nhất cho hầu hết tài liệu. page: mỗi trang là 1 chunk."
                    value={p.reducto_chunk_mode}
                    options={[
                      { value: "page_sections", label: "page_sections (khuyến nghị)" },
                      { value: "page", label: "page — mỗi trang 1 chunk" },
                    ]}
                    onChange={(v) => set("parsing", "reducto_chunk_mode", v)}
                  />
                  <TextInput
                    label="Filter blocks"
                    tip="Loại bỏ các block không cần thiết, phân cách bằng dấu phẩy. Ví dụ: Header,Footer,Page Number. Giúp giảm nhiễu trong chunks."
                    value={p.reducto_filter_blocks}
                    placeholder="Header,Footer,Page Number"
                    onChange={(v) => set("parsing", "reducto_filter_blocks", v)}
                  />
                </div>
              )}

              <div className="sp-divider" />
              <div className="sp-section-title">Chunking</div>

              <Slider
                label="Chunk Max Chars"
                tip="Kích thước tối đa của 1 chunk (ký tự). Kịch bản: Câu hỏi cần context ngắn gọn → 400-600. Câu hỏi cần nhiều ngữ cảnh → 1000-1500. Khi dùng RSE: có thể để 1000-1200 vì RSE tự gom segment phù hợp."
                value={p.chunk_max_chars}
                min={200}
                max={3000}
                step={100}
                unit="chars"
                onChange={(v) => set("parsing", "chunk_max_chars", v)}
              />
              <Slider
                label="Chunk Overlap"
                tip="Số ký tự chồng lấn giữa 2 chunk liền kề. Dùng khi RSE tắt để không bỏ lỡ thông tin ở ranh giới chunk. Khi RSE bật: nên để 0 vì RSE tự gom chunk liên tục."
                value={p.chunk_overlap}
                min={0}
                max={500}
                step={50}
                unit="chars"
                onChange={(v) => set("parsing", "chunk_overlap", v)}
              />
            </div>
          )}

          {/* ── Indexing ────────────────────────────────────────────────────── */}
          {tab === "indexing" && (
            <div className="sp-section">
              <div className="sp-section-title">Embedding</div>
              <TextInput
                label="Embedding Model"
                tip="Model sinh vector embedding. text-embedding-3-small: 1536 chiều, $0.02/1M tokens, nhanh, tốt cho hầu hết use case. text-embedding-3-large: 3072 chiều, chính xác hơn, đắt hơn. ⚠️ Thay đổi model yêu cầu xóa và nạp lại toàn bộ tài liệu."
                value={idx.embed_model}
                placeholder="text-embedding-3-small"
                onChange={(v) => set("indexing", "embed_model", v)}
              />
              <TextInput
                label="Embed Dim (tùy chọn)"
                tip="Số chiều vector embedding. Để trống = tự suy ra từ model (khuyến nghị). Điền số cụ thể nếu dùng Matryoshka embedding và muốn giảm chiều để tiết kiệm RAM."
                value={idx.embed_dim}
                placeholder="(tự suy ra)"
                onChange={(v) => set("indexing", "embed_dim", v)}
              />

              <div className="sp-divider" />
              <div className="sp-section-title">Context Enrichment (4-tier CCH)</div>
              <div className="sp-hint">
                CCH (Contextual Chunk Headers) prepend thêm ngữ cảnh vào embed text mỗi chunk, giúp vector search tìm đúng hơn. Không ảnh hưởng đến text hiển thị cho người dùng.
              </div>

              <Toggle
                label="Tóm tắt tài liệu (Tier 2)"
                tip="Sinh 1 câu tóm tắt toàn bộ tài liệu, thêm vào đầu embed text mỗi chunk. Chi phí ~$0.001/tài liệu với gpt-4o-mini. Nên bật: cải thiện đáng kể recall cho câu hỏi về chủ đề tổng quát."
                value={idx.enable_doc_summary}
                onChange={(v) => set("indexing", "enable_doc_summary", v)}
              >
                <Slider
                  label="Số ký tự đầu vào (doc summary)"
                  tip="Số ký tự tài liệu gửi cho LLM để sinh tóm tắt. 24000 ≈ 6000 tokens, đủ bao quát tài liệu 30-50 trang. Tài liệu rất dài (100+ trang): tăng lên 50000."
                  value={idx.doc_summary_chars}
                  min={1000}
                  max={50000}
                  step={1000}
                  unit="chars"
                  onChange={(v) => set("indexing", "doc_summary_chars", v)}
                />
                <TextInput
                  label="Model sinh tóm tắt tài liệu"
                  tip="Model LLM để sinh doc summary. Để trống = dùng LLM_MODEL_FAST (thường là gpt-4o-mini). gpt-4o-mini là lựa chọn tốt: rẻ và đủ chất lượng cho tóm tắt 1 câu."
                  value={idx.doc_summary_model}
                  placeholder="(dùng LLM Model Fast)"
                  onChange={(v) => set("indexing", "doc_summary_model", v)}
                />
              </Toggle>

              <Toggle
                label="Tóm tắt section (Tier 4)"
                tip="Sinh 1 câu/section, gọi LLM song song cho mỗi section trong tài liệu. Chi phí ~$0.001/tài liệu cho 10-20 section. Cải thiện recall cho câu hỏi về nội dung một phần cụ thể. Nên bật nếu tài liệu có nhiều section."
                value={idx.enable_section_summary}
                onChange={(v) => set("indexing", "enable_section_summary", v)}
              >
                <Slider
                  label="Số ký tự đầu vào (section summary)"
                  tip="Số ký tự nội dung section gửi cho LLM. 2000 ≈ 500 tokens, đủ cho phần lớn section. Section rất dài: tăng lên 3000-5000."
                  value={idx.section_summary_chars}
                  min={200}
                  max={5000}
                  step={200}
                  unit="chars"
                  onChange={(v) => set("indexing", "section_summary_chars", v)}
                />
              </Toggle>
            </div>
          )}

          {/* ── Retrieval ────────────────────────────────────────────────────── */}
          {tab === "retrieval" && (
            <div className="sp-section">
              <div className="sp-section-title">Hybrid Search</div>
              <Slider
                label="BM25 Top-K"
                tip="Số kết quả từ BM25 (keyword matching). Tăng để giảm miss rate nhưng tốn thêm bộ nhớ và thời gian fusion. Kịch bản: document ít (<50 chunks) → 15-20. Document nhiều → 30-50."
                value={ret.bm25_top_k}
                min={5}
                max={100}
                onChange={(v) => set("retrieval", "bm25_top_k", v)}
              />
              <Slider
                label="Dense Top-K"
                tip="Số kết quả từ vector search (semantic). Tăng để bắt được nhiều ngữ nghĩa hơn. Nên để bằng hoặc cao hơn BM25_TOP_K để RRF fusion hiệu quả."
                value={ret.dense_top_k}
                min={5}
                max={100}
                onChange={(v) => set("retrieval", "dense_top_k", v)}
              />
              <Slider
                label="RRF K"
                tip="Hằng số Reciprocal Rank Fusion. 60 là giá trị chuẩn trong literature. Tăng → giảm dominance của rank cao, tăng đa dạng. Giảm → ưu tiên mạnh hơn rank 1."
                value={ret.rrf_k}
                min={10}
                max={200}
                onChange={(v) => set("retrieval", "rrf_k", v)}
              />

              <div className="sp-divider" />
              <div className="sp-section-title">Reranker</div>

              <Toggle
                label="Reranker"
                tip="Dùng cross-encoder để re-score lại candidates sau RRF. Cải thiện độ chính xác đáng kể (+10-15% MRR). Chi phí: thêm 2-3s latency/query với local model. Nên bật nếu latency không quan trọng."
                value={ret.use_reranker}
                onChange={(v) => set("retrieval", "use_reranker", v)}
              >
                <TextInput
                  label="Reranker Model"
                  tip="Model cross-encoder cho reranking. BAAI/bge-reranker-v2-m3: local, miễn phí, tốt cho tiếng Việt. BAAI/bge-reranker-v2-gemma: mạnh hơn, nặng hơn. cross-encoder/ms-marco-MiniLM-L-6-v2: nhẹ, nhanh, tiếng Anh."
                  value={ret.reranker_model}
                  placeholder="BAAI/bge-reranker-v2-m3"
                  onChange={(v) => set("retrieval", "reranker_model", v)}
                />
                <Slider
                  label="Rerank Top-N"
                  tip="Số candidate đưa vào reranker (lấy từ RRF output). Tăng → reranker thấy nhiều hơn, chính xác hơn nhưng chậm hơn. Thường để bằng 1.5-2x FINAL_TOP_K."
                  value={ret.rerank_top_n}
                  min={5}
                  max={50}
                  onChange={(v) => set("retrieval", "rerank_top_n", v)}
                />
              </Toggle>

              <Slider
                label="Final Top-K"
                tip="Số chunk trả về cho LLM (khi RSE tắt). Tăng → LLM nhận được nhiều context hơn nhưng tốn token hơn. Kịch bản: câu hỏi đơn giản → 3-5. Câu hỏi tổng hợp nhiều nguồn → 8-15."
                value={ret.final_top_k}
                min={1}
                max={20}
                onChange={(v) => set("retrieval", "final_top_k", v)}
              />

              <div className="sp-divider" />
              <div className="sp-section-title">Kỹ thuật nâng cao</div>

              <Toggle
                label="HyDE (Hypothetical Document Embedding)"
                tip="Sinh đoạn văn giả định trả lời câu hỏi, rồi dùng vector của đoạn đó để search thay vì vector của câu hỏi. Tăng recall cho câu hỏi trừu tượng hoặc ngắn. Chi phí: 1 LLM call/query."
                value={ret.use_hyde}
                onChange={(v) => set("retrieval", "use_hyde", v)}
              />

              <Toggle
                label="RSE (Relevant Segment Extraction)"
                tip="Thay vì trả về top-k chunks rời rạc, RSE gom các chunks liên tục thành segment hoàn chỉnh bằng thuật toán max-subarray. Hiệu quả nhất khi tài liệu lớn (>30 chunks). Chi phí: ~1ms, không cần LLM."
                value={ret.use_rse}
                onChange={(v) => set("retrieval", "use_rse", v)}
              >
                <Slider
                  label="Irrelevant Penalty"
                  tip="Penalty cho chunk không được retrieve trong thuật toán max-subarray. 0.2 là chuẩn. Tài liệu nhỏ (<30 chunks): giảm xuống 0.05 để segment rộng hơn. Tài liệu lớn: tăng lên 0.3-0.5 để segment chặt chẽ hơn."
                  value={ret.rse_irrelevant_penalty}
                  min={0}
                  max={1}
                  step={0.05}
                  onChange={(v) => set("retrieval", "rse_irrelevant_penalty", v)}
                />
                <Slider
                  label="Max Segment Chunks"
                  tip="Số chunk tối đa trong 1 segment liên tục. Tăng cho phép RSE gom segment dài hơn. Thường để 10-20."
                  value={ret.rse_max_segment_chunks}
                  min={3}
                  max={30}
                  onChange={(v) => set("retrieval", "rse_max_segment_chunks", v)}
                />
                <Slider
                  label="Overall Max Chunks"
                  tip="Tổng số chunk tối đa qua tất cả segment trả về cho LLM. Kiểm soát tổng context window. Thường để 2-3x Final Top-K."
                  value={ret.rse_overall_max_chunks}
                  min={5}
                  max={60}
                  onChange={(v) => set("retrieval", "rse_overall_max_chunks", v)}
                />
                <Slider
                  label="Window Extension"
                  tip="Mở rộng cửa sổ tìm kiếm RSE về phía trước min_idx. Ví dụ: extension=2 sẽ thêm 2 chunk trước hit đầu tiên làm bridge, giúp không bỏ sót câu mở đầu section."
                  value={ret.rse_window_extension}
                  min={0}
                  max={5}
                  onChange={(v) => set("retrieval", "rse_window_extension", v)}
                />
                <Toggle
                  label="Chunk Length Adjustment"
                  tip="Nhân score RSE theo độ dài chunk tương đối. Bật: chunk ngắn (header 19 chars) sẽ có weight rất thấp, chunk dài (thuật toán 1758 chars) có weight cao. Giảm nhiễu từ tiny header chunks."
                  value={ret.rse_chunk_length_adjustment}
                  onChange={(v) => set("retrieval", "rse_chunk_length_adjustment", v)}
                />
              </Toggle>

              <Slider
                label="Min Chunk Chars (index)"
                tip="Chunk có ít ký tự hơn ngưỡng này sẽ KHÔNG được index vào BM25/vector (vẫn lưu DB cho RSE bridge). Tăng để loại bỏ tiny header chunks khỏi search. 50 là mặc định tốt."
                value={ret.min_chunk_chars}
                min={0}
                max={200}
                step={10}
                unit="chars"
                onChange={(v) => set("retrieval", "min_chunk_chars", v)}
              />

              <Slider
                label="Complex Context Limit"
                tip="Số chunk/segment tối đa gửi LLM cho câu hỏi phức tạp (multi-hop). Mỗi segment có thể gom nhiều chunk nên giá trị nhỏ hơn Final Top-K là bình thường. 8 là mặc định."
                value={ret.complex_ctx_limit}
                min={3}
                max={20}
                onChange={(v) => set("retrieval", "complex_ctx_limit", v)}
              />
            </div>
          )}

          {/* ── Generation ───────────────────────────────────────────────────── */}
          {tab === "generation" && (
            <div className="sp-section">
              <div className="sp-section-title">LLM Models</div>
              <TextInput
                label="LLM Model (chính)"
                tip="Model trả lời câu hỏi cuối cùng. gpt-4o: chất lượng cao nhất. gpt-4o-mini: tiết kiệm, tốt cho hầu hết use case. gpt-5: nếu có access. Hỗ trợ bất kỳ model OpenAI-compatible."
                value={gen.llm_model}
                placeholder="gpt-4o-mini"
                onChange={(v) => set("generation", "llm_model", v)}
              />
              <TextInput
                label="LLM Model (nhanh/rẻ)"
                tip="Model cho các task phụ: router, replan, tóm tắt hội thoại, kiểm tra đủ context. gpt-4o-mini là lựa chọn tốt nhất về chi phí/chất lượng."
                value={gen.llm_model_fast}
                placeholder="gpt-4o-mini"
                onChange={(v) => set("generation", "llm_model_fast", v)}
              />

              <div className="sp-divider" />
              <div className="sp-section-title">Agent Loop</div>

              <Toggle
                label="Replan"
                tip="Cho phép agent tự đánh giá lại kế hoạch nếu context chưa đủ và thực hiện thêm bước retrieval. Cải thiện chất lượng cho câu hỏi phức tạp. Chi phí: +1-3 LLM calls/query khi replan xảy ra."
                value={gen.enable_replan}
                onChange={(v) => set("generation", "enable_replan", v)}
              >
                <Slider
                  label="Max Replan Iterations"
                  tip="Số lần tối đa agent được replan trong 1 query. Tăng = câu trả lời đầy đủ hơn nhưng chậm hơn. 2-3 thường là đủ."
                  value={gen.max_replan_iters}
                  min={1}
                  max={5}
                  onChange={(v) => set("generation", "max_replan_iters", v)}
                />
              </Toggle>

              <Toggle
                label="Kiểm tra đủ context (Sufficiency)"
                tip="Trước khi trả lời, agent kiểm tra context retrieved có đủ để trả lời câu hỏi không. Nếu không đủ, trigger replan. Giảm hallucination đáng kể. Chi phí: 1 LLM call/query."
                value={gen.enable_sufficiency}
                onChange={(v) => set("generation", "enable_sufficiency", v)}
              />

              <Toggle
                label="Kiểm tra câu trả lời (Verify)"
                tip="Sau khi sinh câu trả lời, agent tự review và đánh giá chất lượng. Nếu không đạt, regenerate. Cải thiện chất lượng, đặc biệt cho câu hỏi khó. Chi phí: 1-2 LLM calls thêm. Bật tổng tại đây, tinh chỉnh per-route bên dưới."
                value={gen.enable_answer_verify}
                onChange={(v) => set("generation", "enable_answer_verify", v)}
              >
                <div className="sp-hint" style={{ marginTop: 4, marginBottom: 8 }}>
                  Khi bật tổng, chọn route nào cần verify:
                </div>
                <Toggle
                  label="Verify cho câu hỏi đơn giản (simple)"
                  tip="Simple route: câu hỏi trực tiếp, ít nguy cơ hallucination. Mặc định TẮT để stream ngay, giảm TTFT. Bật nếu cần đảm bảo grounding 100%."
                  value={gen.enable_answer_verify_simple}
                  onChange={(v) => set("generation", "enable_answer_verify_simple", v)}
                />
                <Toggle
                  label="Verify cho câu hỏi phức tạp (complex)"
                  tip="Complex route: multi-hop synthesis, dễ hallucination hơn. Mặc định BẬT để kiểm chứng. Tắt nếu ưu tiên tốc độ hơn chất lượng."
                  value={gen.enable_answer_verify_complex}
                  onChange={(v) => set("generation", "enable_answer_verify_complex", v)}
                />
                <Slider
                  label="Max tái sinh câu trả lời"
                  tip="Số lần tối đa tái sinh câu trả lời nếu verify thất bại. 1 thường là đủ. Tăng → chất lượng hơn nhưng chậm hơn."
                  value={gen.max_answer_regenerations}
                  min={0}
                  max={3}
                  onChange={(v) => set("generation", "max_answer_regenerations", v)}
                />
              </Toggle>
            </div>
          )}

          {/* ── Memory ───────────────────────────────────────────────────────── */}
          {tab === "memory" && (
            <div className="sp-section">
              <div className="sp-section-title">Tóm tắt hội thoại</div>
              <Toggle
                label="Tóm tắt hội thoại tự động"
                tip="Khi hội thoại dài, tóm tắt lịch sử cũ thành 1 đoạn ngắn để tiết kiệm token. Không mất thông tin quan trọng, nhưng LLM sẽ không thấy từng tin nhắn cũ mà chỉ thấy bản tóm tắt."
                value={mem.enable_summarization}
                onChange={(v) => set("memory", "enable_summarization", v)}
              >
                <Slider
                  label="Ngưỡng kích hoạt (số tin nhắn)"
                  tip="Tóm tắt sẽ chạy khi tổng số tin nhắn trong session vượt ngưỡng này. Nhỏ = tóm tắt sớm, tiết kiệm token. Lớn = giữ đầy đủ lịch sử lâu hơn. 10-15 là phù hợp cho hầu hết use case."
                  value={mem.summary_threshold}
                  min={4}
                  max={30}
                  onChange={(v) => set("memory", "summary_threshold", v)}
                />
                <Slider
                  label="Cửa sổ lịch sử giữ nguyên"
                  tip="Số tin nhắn gần nhất KHÔNG bị tóm tắt (luôn giữ đầy đủ). Các tin nhắn cũ hơn sẽ được tóm tắt. Tăng để LLM thấy nhiều context gần đây hơn."
                  value={mem.history_window}
                  min={2}
                  max={10}
                  onChange={(v) => set("memory", "history_window", v)}
                />
                <TextInput
                  label="Model tóm tắt"
                  tip="Model sinh tóm tắt hội thoại. Để trống = dùng LLM_MODEL_FAST. gpt-4o-mini đủ tốt và rẻ cho tóm tắt hội thoại."
                  value={mem.summary_model}
                  placeholder="(dùng LLM Model Fast)"
                  onChange={(v) => set("memory", "summary_model", v)}
                />
              </Toggle>
            </div>
          )}
        </div>

        {/* Warnings */}
        {reindexWarning && (
          <div className="sp-warning">
            ⚠️ Một số cài đặt ảnh hưởng đến index (embedding model hoặc chunk size). Tài liệu đã nạp sẽ cần được xóa và nạp lại để áp dụng thay đổi.
          </div>
        )}

        {/* Footer */}
        <div className="sp-footer">
          {saveMsg && (
            <span className={`sp-save-msg ${saveMsg.ok ? "ok" : "err"}`}>{saveMsg.text}</span>
          )}
          {hasChanges && !saveMsg && (
            <span className="sp-dirty-count">{dirtyCount} thay đổi chưa lưu</span>
          )}
          <button className="btn" onClick={onClose}>Đóng</button>
          <button className="btn primary" disabled={saving || !hasChanges} onClick={save}>
            {saving ? "Đang lưu…" : hasChanges ? `Lưu ${dirtyCount} thay đổi` : "Lưu thay đổi"}
          </button>
        </div>
      </div>
    </div>
  );
}
