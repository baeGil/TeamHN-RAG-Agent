import { useMemo } from "react";
import type { TraceEvent } from "../lib/types";

type NodeState = "idle" | "active" | "done" | "skipped" | "error";
type EdgeState = "idle" | "traversed" | "skipped";

interface NodeDef {
  id: string;
  label: string;
  sub?: string;
  x: number;
  y: number;
  w: number;
  h: number;
}

interface EdgeDef {
  from: string;
  to: string;
  label: string;       // every edge has a label
  lx: number;          // label x (absolute coords)
  ly: number;          // label y
  anchor?: "start" | "middle" | "end";     // text-anchor
}

const NODES: NodeDef[] = [
  { id: "router",        label: "Router",         sub: "Phân loại câu hỏi",      x: 180, y: 10,  w: 140, h: 56 },
  { id: "simple",        label: "Single-hop",     sub: "Đường nhanh",            x: 30,  y: 110, w: 130, h: 56 },
  { id: "complex",        label: "Multi-hop",       sub: "Agent đầy đủ",            x: 330, y: 110, w: 140, h: 56 },
  { id: "planner",        label: "Planner",         sub: "Tách câu hỏi con",        x: 330, y: 200, w: 140, h: 56 },
  { id: "retrieve",       label: "Retrieve",        sub: "BM25 + Dense + Rerank",   x: 180, y: 300, w: 140, h: 60 },
  { id: "distill_verify", label: "Distill+Verify",  sub: "Chắt lọc & bám nguồn (1 call)", x: 180, y: 400, w: 170, h: 56 },
  { id: "sufficiency",    label: "Sufficiency",     sub: "Đủ thông tin?",           x: 330, y: 490, w: 140, h: 56 },
  { id: "replan",         label: "Replan",          sub: "Lập lại kế hoạch",        x: 330, y: 580, w: 140, h: 56 },
  { id: "synthesize",     label: "Synthesize",      sub: "Sinh câu trả lời",        x: 180, y: 580, w: 140, h: 56 },
  { id: "verify_answer",  label: "Verify Answer",   sub: "Kiểm chứng trả lời",      x: 180, y: 670, w: 140, h: 56 },
  { id: "answer",         label: "Answer",          sub: "Stream + Citations",      x: 180, y: 760, w: 140, h: 56 },
];

// Every edge has a short Vietnamese label + absolute label position.
// Labels are NEVER rotated — always horizontal for readability.
const EDGES: EdgeDef[] = [
  // ── Router branches ──
  { from: "router", to: "simple",   label: "simple",         lx: 145, ly: 82 },
  { from: "router", to: "complex",  label: "complex",      lx: 340, ly: 82 },
  { from: "router", to: "answer",   label: "no retrieval",  lx: 30,  ly: 430, anchor: "middle" },

  // ── Complex path ──
  { from: "complex", to: "planner", label: "",              lx: 0,   ly: 0 },
  { from: "planner", to: "retrieve", label: "",             lx: 0,   ly: 0 },

  // ── Simple path ──
  { from: "simple", to: "retrieve",  label: "",             lx: 0,   ly: 0 },

  // ── Retrieve outputs ──
  { from: "retrieve", to: "distill_verify", label: "multi-hop",  lx: 260, ly: 350 },
  { from: "retrieve", to: "synthesize",     label: "single-hop", lx: 255, ly: 475 },

  // ── Complex verification chain ──
  { from: "distill_verify", to: "sufficiency", label: "",         lx: 280, ly: 445 },

  // ── Sufficiency exits ──
  { from: "sufficiency", to: "synthesize", label: "đủ",          lx: 280, ly: 538 },
  { from: "sufficiency", to: "replan",     label: "chưa đủ",      lx: 415, ly: 538 },

  // ── Replan loop ──
  { from: "replan", to: "retrieve",       label: "lặp lại",  lx: 488, ly: 440, anchor: "start" },

  // ── Post-retrieval convergence ──
  { from: "synthesize", to: "verify_answer", label: "",     lx: 0,   ly: 0 },
  { from: "synthesize", to: "answer",          label: "stream", lx: 120, ly: 620 },

  // ── Verify Answer exits ──
  { from: "verify_answer", to: "answer",     label: "bám nguồn", lx: 258, ly: 718 },
  { from: "verify_answer", to: "synthesize", label: "tái tạo",  lx: 35,  ly: 628, anchor: "middle" },
];

const NODE_MAP: Record<string, NodeDef> = Object.fromEntries(NODES.map((n) => [n.id, n]));

function deriveStates(events: TraceEvent[], hasAnswer: boolean): {
  states: Record<string, NodeState>;
  edgeStates: Record<string, EdgeState>;
  route: string | null;
  meta: Record<string, string>;
} {
  const states: Record<string, NodeState> = Object.fromEntries(
    NODES.map((n) => [n.id, "idle"])
  );
  const edgeStates: Record<string, EdgeState> = Object.fromEntries(
    EDGES.map((e, i) => [`${e.from}→${e.to}`, "idle"])
  );
  const meta: Record<string, string> = {};
  let route: string | null = null;
  let active: string | null = null;
  let iteration = 0;

  const markDone = (id: string) => {
    if (states[id] !== "skipped" && states[id] !== "error") {
      states[id] = "done";
    }
  };

  const setActive = (id: string) => {
    if (active && active !== id && states[active] !== "skipped" && states[active] !== "error") {
      markDone(active);
    }
    active = id;
    if (states[id] !== "skipped" && states[id] !== "error") {
      states[id] = "active";
    }
  };

  const traverseEdge = (from: string, to: string) => {
    const key = `${from}→${to}`;
    if (key in edgeStates) {
      edgeStates[key] = "traversed";
    }
  };

  const markSkippedEdges = () => {
    for (const e of EDGES) {
      const key = `${e.from}→${e.to}`;
      if (edgeStates[key] === "idle") {
        const fromSkipped = states[e.from] === "skipped";
        const toSkipped = states[e.to] === "skipped";
        if (fromSkipped || toSkipped) {
          edgeStates[key] = "skipped";
        }
      }
    }
  };

  for (const ev of events) {
    if (ev.type === "thinking") {
      const node = ev.data.node;
      if (node && NODE_MAP[node]) {
        setActive(node);
      }
    } else if (ev.type === "route") {
      route = ev.data.route;
      markDone("router");
      if (route === "simple") {
        traverseEdge("router", "simple");
        markDone("simple");
        states["complex"] = "skipped";
        states["planner"] = "skipped";
        states["distill_verify"] = "skipped";
        states["sufficiency"] = "skipped";
        states["replan"] = "skipped";
        markSkippedEdges();
        setActive("retrieve");
      } else if (route === "complex") {
        traverseEdge("router", "complex");
        markDone("complex");
        states["simple"] = "skipped";
        markSkippedEdges();
        setActive("planner");
      } else if (route === "no_retrieval") {
        traverseEdge("router", "answer");
        states["simple"] = "skipped";
        states["complex"] = "skipped";
        states["planner"] = "skipped";
        states["retrieve"] = "skipped";
        states["distill_verify"] = "skipped";
        states["sufficiency"] = "skipped";
        states["replan"] = "skipped";
        states["synthesize"] = "skipped";
        states["verify_answer"] = "skipped";
        markSkippedEdges();
        setActive("answer");
      }
    } else if (ev.type === "plan") {
      traverseEdge("complex", "planner");
      markDone("planner");
      meta["planner"] = `${ev.data.subquestions?.length || 0} bước`;
      traverseEdge("planner", "retrieve");
      setActive("retrieve");
    } else if (ev.type === "subquestion") {
      if (states["simple"] !== "skipped") markDone("simple");
      if (states["complex"] !== "skipped") markDone("complex");
    } else if (ev.type === "retrieved") {
      markDone("retrieve");
      const chunks = ev.data.chunks || [];
      // Count context entries (segments count as 1 each, not their constituent chunks)
      const nEntries = chunks.length;
      meta["retrieve"] = `${nEntries} đoạn`;
      if (route === "simple") {
        traverseEdge("simple", "retrieve");
        traverseEdge("retrieve", "synthesize");
        setActive("synthesize");
      } else {
        traverseEdge("retrieve", "distill_verify");
        setActive("distill_verify");
      }
    } else if (ev.type === "distill_verify") {
      markDone("distill_verify");
      const parts: string[] = [];
      if (ev.data.relevant) parts.push("✓ liên quan");
      else parts.push("✗ không liên quan");
      if (ev.data.grounded === false) parts.push("⚠ chưa chắc");
      else if (ev.data.grounded === true) parts.push("✓ bám nguồn");
      meta["distill_verify"] = parts.join(" · ");
      traverseEdge("distill_verify", "sufficiency");
      setActive("sufficiency");
    } else if (ev.type === "distill" || ev.type === "verify") {
      // Legacy: still handle old separate events, map to distill_verify node
      if (ev.type === "distill") {
        meta["distill_verify"] = ev.data.relevant ? "✓ liên quan" : "✗ không liên quan";
        if (states["distill_verify"] === "idle" || states["distill_verify"] === "active") {
          setActive("distill_verify");
        }
      } else if (ev.type === "verify") {
        let prev = meta["distill_verify"] || "";
        if (ev.data.grounded === false) prev += " · ⚠ chưa chắc";
        else if (ev.data.grounded === true) prev += " · ✓ bám nguồn";
        meta["distill_verify"] = prev.trim();
        markDone("distill_verify");
        traverseEdge("distill_verify", "sufficiency");
        setActive("sufficiency");
      }
    } else if (ev.type === "sufficiency") {
      markDone("sufficiency");
      const suf = ev.data.sufficient;
      meta["sufficiency"] = suf ? "✓ đủ thông tin" : "⚠ chưa đủ";
      if (suf) {
        traverseEdge("sufficiency", "synthesize");
        setActive("synthesize");
      } else {
        traverseEdge("sufficiency", "replan");
        setActive("replan");
      }
    } else if (ev.type === "converged") {
      markDone("distill_verify");
      traverseEdge("distill_verify", "sufficiency");
      markDone("sufficiency");
      meta["sufficiency"] = "✓ tất cả bám nguồn";
      traverseEdge("sufficiency", "synthesize");
      setActive("synthesize");
    } else if (ev.type === "max_iters") {
      markDone("replan");
      meta["replan"] = "⏹ đạt giới hạn";
      setActive("synthesize");
    } else if (ev.type === "early_stop") {
      markDone("replan");
      meta["replan"] = "⚡ dừng sớm";
      setActive("synthesize");
    } else if (ev.type === "replan") {
      traverseEdge("replan", "retrieve");
      markDone("replan");
      iteration = ev.data.iteration || iteration + 1;
      meta["replan"] = `🔄 vòng ${iteration}`;
      setActive("retrieve");
    } else if (ev.type === "synthesize") {
      markDone("synthesize");
      meta["synthesize"] = `${ev.data.n_context || 0} ngữ cảnh`;
      // Will be followed by verify_answer event if verify is ON for this route.
      // If no verify_answer event comes (verify OFF), the "final" event will
      // traverse synthesize→answer directly.
      setActive("verify_answer");
    } else if (ev.type === "verify_answer") {
      traverseEdge("synthesize", "verify_answer");
      markDone("verify_answer");
      if (ev.data.grounded) {
        traverseEdge("verify_answer", "answer");
        meta["verify_answer"] = "✓ bám nguồn";
        setActive("answer");
      } else {
        traverseEdge("verify_answer", "synthesize");
        meta["verify_answer"] = "⚠ hallucinate → tái tạo";
        setActive("synthesize");
      }
    } else if (ev.type === "error") {
      const node = ev.data.node;
      if (node && NODE_MAP[node]) {
        states[node] = "error";
        meta[node] = ev.data.message || "Lỗi";
      }
    }
  }

  markSkippedEdges();

  // If synthesize is done but verify_answer was skipped (verify OFF for simple route),
  // traverse synthesize→answer directly.
  if (states["synthesize"] === "done" && states["verify_answer"] !== "done" && states["verify_answer"] !== "active") {
    if (states["verify_answer"] !== "skipped") states["verify_answer"] = "skipped";
    edgeStates["synthesize→verify_answer"] = "skipped";
    if (hasAnswer) {
      edgeStates["synthesize→answer"] = "traversed";
    }
  }

  if (hasAnswer) {
    for (const n of NODES) {
      if (states[n.id] === "active") states[n.id] = "done";
    }
    states["answer"] = "done";
    active = null;
  } else if (active === "answer") {
    states["answer"] = "active";
  }

  return { states, edgeStates, route, meta };
}

interface Props {
  events: TraceEvent[];
  liveAnswer?: string;
  done?: boolean;
}

export default function AgentGraph({ events, liveAnswer, done }: Props) {
  const hasAnswer = !!(liveAnswer && liveAnswer.length > 0) || !!done;
  const { states, edgeStates, meta } = useMemo(
    () => deriveStates(events, hasAnswer),
    [events, hasAnswer]
  );

  return (
    <div className="agent-graph">
      <svg viewBox="0 0 520 890" preserveAspectRatio="xMidYMin meet">
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" fill="#8b9bc3" />
          </marker>
          <marker id="arrow-done" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" fill="#7fa0ff" />
          </marker>
        </defs>

        {EDGES.map((e, i) => {
          const a = NODE_MAP[e.from];
          const b = NODE_MAP[e.to];
          if (!a || !b) return null;

          const edgeKey = `${e.from}→${e.to}`;
          const eState = edgeStates[edgeKey] || "idle";
          const isTraversed = eState === "traversed";
          const isSkipped = eState === "skipped";

          const x1 = a.x + a.w / 2;
          const y1 = a.y + a.h;
          const x2 = b.x + b.w / 2;
          const y2 = b.y;

          let path: string;

          if (e.from === "replan" && e.to === "retrieve") {
            const rx = 495;
            path = `M ${x1} ${y1} C ${rx} ${y1}, ${rx} ${y2}, ${x2} ${y2}`;
          } else if (e.from === "verify_answer" && e.to === "synthesize") {
            const rx = 35;
            path = `M ${x1} ${y1} C ${rx} ${y1}, ${rx} ${y2}, ${x2} ${y2}`;
          } else if (e.from === "router" && e.to === "answer") {
            const rx = 18;
            path = `M ${a.x + a.w * 0.2} ${a.y + a.h} C ${rx} ${a.y + a.h + 30}, ${rx} ${b.y - 30}, ${b.x + b.w * 0.2} ${b.y}`;
          } else {
            path = `M ${x1} ${y1} C ${x1} ${y1 + 30}, ${x2} ${y2 - 30}, ${x2} ${y2}`;
          }

          // Hide label for edges with empty label or when lx=0 (no label position set)
          const showLabel = e.label.length > 0 && e.lx !== 0;

          return (
            <g key={i} className={`edge ${isTraversed ? "edge-done" : ""} ${isSkipped ? "edge-skipped" : ""}`}>
              <path
                d={path}
                fill="none"
                stroke={isTraversed ? "#7fa0ff" : isSkipped ? "#49536a" : "#8b9bc3"}
                strokeWidth={isTraversed ? 2.4 : 1.8}
                strokeDasharray={isSkipped ? "4 4" : undefined}
                markerEnd={isTraversed ? "url(#arrow-done)" : "url(#arrow)"}
              />
              {showLabel && (
                <text
                  x={e.lx}
                  y={e.ly}
                  className="edge-label"
                  textAnchor={e.anchor || "middle"}
                >
                  {e.label}
                </text>
              )}
            </g>
          );
        })}

        {NODES.map((n) => {
          const state = states[n.id];
          return (
            <g key={n.id} className={`node node-${state}`} transform={`translate(${n.x},${n.y})`}>
              {state === "active" && (
                <rect
                  className="node-glow"
                  x={-4}
                  y={-4}
                  width={n.w + 8}
                  height={n.h + 8}
                  rx={12}
                />
              )}
              <rect x={0} y={0} width={n.w} height={n.h} rx={10} className="node-rect" />
              <text x={n.w / 2} y={22} className="node-label" textAnchor="middle">
                {n.label}
              </text>
              <text x={n.w / 2} y={40} className="node-sub" textAnchor="middle">
                {meta[n.id] || n.sub}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
