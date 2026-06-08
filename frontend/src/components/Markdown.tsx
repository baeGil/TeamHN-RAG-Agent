import { useCallback, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import remarkGfm from "remark-gfm";
import rehypeKatex from "rehype-katex";
import rehypeRaw from "rehype-raw";
import type { Citation } from "../lib/types";

interface Props {
  content: string;
  citations: Citation[];
  onCite?: (label: number) => void;
}

function injectCitations(text: string): string {
  return text.replace(/\[(\d{1,3})\]/g, (_m, n) => `<cite data-label="${n}">${n}</cite>`);
}

function CitationBadge({ label, c, onCite }: { label: number; c: Citation | undefined; onCite?: (label: number) => void }) {
  const ref = useRef<HTMLSpanElement>(null);
  const [pos, setPos] = useState<{ top: number; left: number; dir: "above" | "below" }>({ top: 0, left: 0, dir: "above" });

  const handleMouseEnter = useCallback(() => {
    if (!ref.current) return;
    const rect = ref.current.getBoundingClientRect();
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const pw = 340;
    const gap = 8;

    let left: number;
    let dir: "above" | "below";

    if (rect.top > pw * 0.6 + gap * 2) {
      dir = "above";
      left = rect.left + rect.width / 2 - pw / 2;
    } else {
      dir = "below";
      left = rect.left + rect.width / 2 - pw / 2;
    }

    if (left < gap) left = gap;
    if (left + pw > vw - gap) left = vw - gap - pw;

    setPos({ top: dir === "above" ? rect.top - gap : rect.bottom + gap, left, dir });
  }, []);

  return (
    <span
      ref={ref}
      className={`cite-badge ${c ? "" : "cite-unknown"}`}
      onClick={() => onCite?.(label)}
      onMouseEnter={handleMouseEnter}
      title={c ? `${c.doc_title}${c.page ? ` · trang ${c.page}` : ""}` : ""}
    >
      {label}
      {c && (
        <span
          className={`cite-pop ${pos.dir === "above" ? "cite-pop-above" : "cite-pop-below"}`}
          style={{ top: pos.top, left: pos.left, display: undefined }}
        >
          <span className="cite-pop-head">
            {c.doc_title}
            {c.page != null ? ` · trang ${c.page}` : ""}
            {c.is_segment && c.n_chunks ? ` (${c.n_chunks} đoạn)` : ""}
          </span>
          <span className="cite-pop-body">
            {c.is_segment && c.n_chunks
              ? c.text.slice(0, 600) + (c.text.length > 600 ? "…" : "")
              : c.text}
          </span>
        </span>
      )}
    </span>
  );
}

export default function Markdown({ content, citations, onCite }: Props) {
  const citeMap = new Map(citations.map((c) => [c.label, c]));
  const processed = injectCitations(content);

  return (
    <div className="md">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeRaw, rehypeKatex]}
        components={{
          cite: ({ node, ...props }: any) => {
            const label = Number(props["data-label"]);
            const c = citeMap.get(label);
            return <CitationBadge label={label} c={c} onCite={onCite} />;
          },
        }}
      >
        {processed}
      </ReactMarkdown>
    </div>
  );
}