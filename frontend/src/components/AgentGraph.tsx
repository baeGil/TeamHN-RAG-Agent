import { useMemo } from "react";
import type { TraceEvent } from "../lib/types";

type NodeState = "idle" | "active" | "done" | "skipped" | "error";

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
  label?: string;
}

// Vertical pipeline layout on a 460x720 canvas.
const NODES: NodeDef[] = [
  { id: "router", label: "Router", sub: "Phân loại câu hỏi", x: 180, y: 20, w: 140, h: 56 },
  { id: "simple", label: "Single-hop", sub: "Đường nhanh", x: 30, y: 120, w: 140, h: 56 },
  { id: "complex", label: "Multi-hop", sub: "Agent đầy đủ", x: 330, y: 120, w: 140, h: 56 },
  { id: "planner", label: "Planner", sub: "Tách câu hỏi con", x: 330, y: 220, w: 140, h: 56 },
  { id: "retrieve", label: "Retrieve", sub: "BM25 + Dense + RRF", x: 180, y: 320, w: 140, h: 60 },
  { id: "distill", label: "Distill", sub: "Chắt lọc ngữ cảnh", x: 180, y: 420, w: 140, h: 56 },
  { id: "verify", label: "Verify", sub: "Bám nguồn (Self-RAG)", x: 180, y: 510, w: 140, h: 56 },
  { id: "synthesize", label: "Synthesize", sub: "Sinh câu trả lời", x: 180, y: 600, w: 140, h: 56 },
  { id: "answer", label: "Answer", sub: "Stream + Citations", x: 180, y: 690, w: 140, h: 56 },
];

const EDGES: EdgeDef[] = [
  { from: "router", to: "simple" },
  { from: "router", to: "complex" },
  { from: "complex", to: "planner" },
  { from: "simple", to: "retrieve" },
  { from: "planner", to: "retrieve" },
  { from: "retrieve", to: "distill" },
  { from: "distill", to: "verify" },
  { from: "verify", to: "synthesize" },
  { from: "retrieve", to: "synthesize", label: "single-hop" },
  { from: "synthesize", to: "answer" },
];

const NODE_MAP: Record<string, NodeDef> = Object.fromEntries(NODES.map((n) => [n.id, n]));

// Compute a state per node from the ordered event stream.
function deriveStates(events: TraceEvent[], hasAnswer: boolean): {
  states: Record<string, NodeState>;
  route: string | null;
  meta: Record<string, string>;
} {
  const states: Record<string, NodeState> = Object.fromEntries(
    NODES.map((n) => [n.id, "idle"])
  );
  const meta: Record<string, string> = {};
  let route: string | null = null;
  let active: string | null = null;

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

  for (const ev of events) {
    if (ev.type === "thinking") {
      setActive(ev.data.node);
    } else if (ev.type === "route") {
      route = ev.data.route;
      markDone("router");
      if (route === "simple") {
        markDone("simple");
        states["complex"] = "skipped";
        states["planner"] = "skipped";
        states["distill"] = "skipped";
        states["verify"] = "skipped";
        setActive("retrieve");
      } else if (route === "complex") {
        markDone("complex");
        states["simple"] = "skipped";
        setActive("planner");
      } else if (route === "no_retrieval") {
        states["simple"] = "skipped";
        states["complex"] = "skipped";
        states["planner"] = "skipped";
        states["retrieve"] = "skipped";
        states["distill"] = "skipped";
        states["verify"] = "skipped";
        states["synthesize"] = "skipped";
        setActive("answer");
      }
    } else if (ev.type === "plan") {
      markDone("planner");
      meta["planner"] = `${ev.data.subquestions?.length || 0} bước`;
      setActive("retrieve");
    } else if (ev.type === "subquestion") {
      if (states["simple"] !== "skipped") markDone("simple");
      if (states["complex"] !== "skipped") markDone("complex");
    } else if (ev.type === "retrieved") {
      markDone("retrieve");
      const n = ev.data.chunks?.length || 0;
      meta["retrieve"] = `${n} đoạn`;
      if (route === "simple") {
        setActive("synthesize");
      } else if (!active || active === "retrieve") {
        setActive("distill");
      }
    } else if (ev.type === "distill") {
      markDone("distill");
      meta["distill"] = ev.data.relevant ? "✓ liên quan" : "✗ không liên quan";
      setActive("verify");
    } else if (ev.type === "verify") {
      markDone("verify");
      if (ev.data.grounded === false) meta["verify"] = "⚠ chưa chắc";
      else if (ev.data.grounded === true) meta["verify"] = "✓ bám nguồn";
      setActive("synthesize");
    } else if (ev.type === "synthesize") {
      markDone("synthesize");
      meta["synthesize"] = `${ev.data.n_context || 0} ngữ cảnh`;
      setActive("answer");
    } else if (ev.type === "error") {
      const node = ev.data.node;
      if (node && NODE_MAP[node]) {
        states[node] = "error";
        meta[node] = ev.data.message || "Lỗi";
      }
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

  return { states, route, meta };
}

interface Props {
  events: TraceEvent[];
  liveAnswer?: string;
  done?: boolean;
}

export default function AgentGraph({ events, liveAnswer, done }: Props) {
  const hasAnswer = !!(liveAnswer && liveAnswer.length > 0) || !!done;
  const { states, meta } = useMemo(
    () => deriveStates(events, hasAnswer),
    [events, hasAnswer]
  );

  return (
    <div className="agent-graph">
      <svg viewBox="0 0 500 760" preserveAspectRatio="xMidYMin meet">
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" fill="#5a6378" />
          </marker>
          <marker id="arrow-done" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" fill="var(--accent2)" />
          </marker>
        </defs>

        {EDGES.map((e, i) => {
          const a = NODE_MAP[e.from];
          const b = NODE_MAP[e.to];
          const x1 = a.x + a.w / 2;
          const y1 = a.y + a.h;
          const x2 = b.x + b.w / 2;
          const y2 = b.y;
          const fromDone = states[e.from] === "done" || states[e.from] === "active";
          const toReached = states[e.to] === "done" || states[e.to] === "active";
          const isDone = fromDone && toReached;
          const isSkipped = states[e.from] === "skipped" || states[e.to] === "skipped";
          const isError = states[e.from] === "error" || states[e.to] === "error";
          const mid = `${(x1 + x2) / 2},${(y1 + y2) / 2}`;
          const path = `M ${x1} ${y1} C ${x1} ${y1 + 30}, ${x2} ${y2 - 30}, ${x2} ${y2}`;
          return (
            <g key={i} className={`edge ${isDone ? "edge-done" : ""} ${isSkipped ? "edge-skipped" : ""}`}>
              <path
                d={path}
                fill="none"
                stroke={isDone ? "var(--accent2)" : isError ? "#ef4444" : isSkipped ? "#2a2f3d" : "#5a6378"}
                strokeWidth={isDone ? 2 : 1.5}
                strokeDasharray={isSkipped ? "4 4" : undefined}
                markerEnd={isDone ? "url(#arrow-done)" : "url(#arrow)"}
              />
              {e.label && (
                <text x={mid.split(",")[0]} y={mid.split(",")[1]} className="edge-label" textAnchor="middle">
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
