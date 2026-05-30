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

// Replace citation markers [n] (single integer) with an inline <cite> element,
// avoiding intervals like [0, 1] (which contain commas/spaces).
function injectCitations(text: string): string {
  return text.replace(/\[(\d{1,3})\]/g, (_m, n) => `<cite data-label="${n}">${n}</cite>`);
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
            return (
              <span
                className={`cite-badge ${c ? "" : "cite-unknown"}`}
                onClick={() => onCite?.(label)}
                title={c ? `${c.doc_title}${c.page ? ` · trang ${c.page}` : ""}` : ""}
              >
                {label}
                {c && (
                  <span className="cite-pop">
                    <span className="cite-pop-head">
                      {c.doc_title}
                      {c.page ? ` · trang ${c.page}` : ""}
                    </span>
                    <span className="cite-pop-body">{c.text}</span>
                  </span>
                )}
              </span>
            );
          },
        }}
      >
        {processed}
      </ReactMarkdown>
    </div>
  );
}
