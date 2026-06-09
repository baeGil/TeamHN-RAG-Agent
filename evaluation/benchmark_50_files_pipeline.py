"""50-files PDF benchmark pipeline.

This runner is intentionally separate from ``backend/eval`` because the
``data/50_files`` dataset has a different schema and must be ingested with
Reducto only. If Reducto is not configured or fails on any PDF, the run stops.

Usage from repository root:
    python evaluation/benchmark_50_files_pipeline.py --reset
    python evaluation/benchmark_50_files_pipeline.py --reset --ingest-only
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.agent.graph import Agent  # noqa: E402
from app.agent.llm import LLM  # noqa: E402
from app.config import Settings  # noqa: E402
from app.indexing.store import KnowledgeBase  # noqa: E402
from app.ingestion.reducto_parser import parse_pdf_reducto  # noqa: E402
from app.ingestion.vn_text import normalize_structure  # noqa: E402

K = 5
DEFAULT_DATA_DIR = ROOT / "data" / "50_files"
DEFAULT_STORAGE = BACKEND / "storage_50_files_reducto"
DEFAULT_OUT_DIR = ROOT / "evaluation" / "benchmark_50_files_runs"


@dataclass
class QAItem:
    doc_id: str
    question_id: str
    question: str
    expected_answer: str
    evidence: str
    difficulty: str
    difficulty_explanation: str
    question_types: str
    human_steps: str
    answerability: str
    notes: str


def split_md_row(line: str) -> list[str]:
    return [cell.strip().replace("<br>", "\n") for cell in line.strip().strip("|").split("|")]


def load_50_files_dataset(golden_dir: Path) -> list[QAItem]:
    items: list[QAItem] = []
    for path in sorted(golden_dir.glob("DOC*_qa.md")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("| DOC"):
                continue
            cells = split_md_row(line)
            if len(cells) < 11:
                raise ValueError(f"Malformed QA row in {path}: {line[:120]}")
            items.append(QAItem(*cells[:11]))
    if not items:
        raise ValueError(f"No QA items found in {golden_dir}")
    return items


def normalize_difficulty(raw: str) -> str:
    text = raw.strip().lower()
    if text in {"dễ", "de", "easy"}:
        return "easy"
    if text in {"trung bình", "trung binh", "medium"}:
        return "medium"
    if text in {"khó", "kho", "hard"}:
        return "hard"
    return text or "unknown"


def load_markdown_qa_dataset(dataset_dir: Path, pdfs: dict[str, Path]) -> list[QAItem]:
    """Load data_hung-style QA markdown files.

    Expected columns: level | question | expected answer | page/source.
    Each markdown file is matched to a PDF by the same stem.
    """
    items: list[QAItem] = []
    pdf_stems = {path.stem for path in pdfs.values()}
    for path in sorted(dataset_dir.glob("*.md")):
        if path.stem not in pdf_stems:
            continue
        q_idx = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip().startswith("|"):
                continue
            cells = split_md_row(line)
            if len(cells) < 4:
                continue
            level, question, expected, evidence = cells[:4]
            if level.startswith(":") or level.lower() in {"cấp độ", "cap do", "difficulty"}:
                continue
            if not question or not expected:
                continue
            q_idx += 1
            doc_id = path.stem
            items.append(
                QAItem(
                    doc_id=doc_id,
                    question_id=f"{doc_id}_Q{q_idx:02d}",
                    question=question,
                    expected_answer=expected,
                    evidence=evidence,
                    difficulty=normalize_difficulty(level),
                    difficulty_explanation="",
                    question_types="",
                    human_steps="",
                    answerability="answerable",
                    notes="",
                )
            )
    if not items:
        raise ValueError(f"No markdown QA rows found in {dataset_dir}")
    return items


def load_document_manifest(data_dir: Path) -> dict[str, Path]:
    pdfs: dict[str, Path] = {}
    pdf_dir = data_dir / "raw_pdf"
    if not pdf_dir.exists():
        pdf_dir = data_dir
    for path in sorted(pdf_dir.glob("*.pdf")):
        match = re.match(r"^(DOC\d{3})_", path.name)
        doc_id = match.group(1) if match else path.stem
        if doc_id in pdfs:
            doc_id = f"{doc_id}_{len(pdfs) + 1}"
        pdfs[doc_id] = path
    if not pdfs:
        raise ValueError(f"No PDFs found in {pdf_dir}")
    return pdfs


def filter_manifest(
    pdfs: dict[str, Path],
    doc_ids: str = "",
    pdf_names: str = "",
) -> dict[str, Path]:
    selected = dict(pdfs)
    if doc_ids.strip():
        wanted = {item.strip().upper() for item in doc_ids.split(",") if item.strip()}
        missing = sorted(wanted - set(selected))
        if missing:
            raise SystemExit(f"Unknown --doc-id value(s): {', '.join(missing)}")
        selected = {doc_id: path for doc_id, path in selected.items() if doc_id in wanted}
    if pdf_names.strip():
        wanted_names = {item.strip() for item in pdf_names.split(",") if item.strip()}
        by_name = {path.name: doc_id for doc_id, path in selected.items()}
        missing_names = sorted(wanted_names - set(by_name))
        if missing_names:
            raise SystemExit(f"Unknown --pdf-name value(s): {', '.join(missing_names)}")
        selected = {
            doc_id: path
            for doc_id, path in selected.items()
            if path.name in wanted_names
        }
    if not selected:
        raise SystemExit("No PDFs selected for ingest/evaluation.")
    return selected


def make_settings(storage_dir: Path, reducto_mode: str) -> Settings:
    settings = Settings()
    settings.storage_dir = storage_dir.resolve()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    settings.reducto_parse = reducto_mode
    if reducto_mode not in {"default", "agentic"}:
        raise SystemExit("--reducto-mode must be 'default' or 'agentic'. Reducto is mandatory.")
    if not settings.reducto_api_key:
        raise SystemExit("REDUCTO_API_KEY is not configured. Stopping because Reducto is mandatory.")
    if not settings.has_openai:
        raise SystemExit("OPENAI_API_KEY is not configured. Evaluation needs embeddings and LLM judges.")
    return settings


def ingest_pdf_reducto_only(kb: KnowledgeBase, pdf_path: Path, doc_id: str) -> dict[str, Any]:
    """Parse with Reducto directly and ingest blocks. No local fallback is used."""
    s = kb.settings
    blocks, meta = parse_pdf_reducto(
        pdf_path=pdf_path,
        api_key=s.reducto_api_key,
        mode=s.reducto_parse,
        chunk_mode=s.reducto_chunk_mode,
        chunk_size=s.reducto_chunk_size,
        filter_blocks=s.reducto_filter_blocks,
        table_format=s.reducto_table_format,
    )
    if not blocks:
        raise RuntimeError(f"Reducto returned zero blocks for {pdf_path.name}")
    for block in blocks:
        block.text = normalize_structure(block.text)
        if block.embed_text:
            block.embed_text = normalize_structure(block.embed_text)
    title = re.sub(r"\.pdf$", "", pdf_path.name, flags=re.IGNORECASE)
    result = kb._ingest(title, pdf_path.name, "pdf", blocks)
    saved_pdf = kb.pdf_path(int(result["document_id"]))
    saved_pdf.parent.mkdir(parents=True, exist_ok=True)
    saved_pdf.write_bytes(pdf_path.read_bytes())
    return {**result, "doc_id": doc_id, "reducto": meta}


def ensure_reducto_index(kb: KnowledgeBase, pdfs: dict[str, Path]) -> list[dict[str, Any]]:
    existing_sources = {doc["source"] for doc in kb.repo.list_documents()}
    ingest_results: list[dict[str, Any]] = []
    for doc_id, pdf_path in sorted(pdfs.items()):
        if pdf_path.name in existing_sources:
            continue
        print(f"[ingest] Reducto parsing {doc_id}: {pdf_path.name}", flush=True)
        started = time.perf_counter()
        result = ingest_pdf_reducto_only(kb, pdf_path, doc_id)
        result["elapsed"] = round(time.perf_counter() - started, 2)
        ingest_results.append(result)
        print(
            f"[ingest] done {doc_id}: chunks={result['n_chunks']} "
            f"credits={result['reducto'].get('total_credits')} elapsed={result['elapsed']}s",
            flush=True,
        )

    ready = kb.repo.list_documents()
    ready_sources = {doc["source"] for doc in ready}
    missing_sources = sorted(path.name for path in pdfs.values() if path.name not in ready_sources)
    if missing_sources:
        raise RuntimeError(f"Index is missing selected PDF(s): {missing_sources}")
    return ingest_results


def metric_hit(ranked: list[int], relevant: set[int], k: int = K) -> float:
    return 1.0 if any(cid in relevant for cid in ranked[:k]) else 0.0


def metric_precision(ranked: list[int], relevant: set[int], k: int = K) -> float:
    if k <= 0:
        return 0.0
    return sum(1 for cid in ranked[:k] if cid in relevant) / k


def metric_recall(ranked: list[int], relevant: set[int], k: int = K) -> float:
    if not relevant:
        return 0.0
    return sum(1 for cid in ranked[:k] if cid in relevant) / len(relevant)


def metric_mrr(ranked: list[int], relevant: set[int], k: int = K) -> float:
    for i, cid in enumerate(ranked[:k], start=1):
        if cid in relevant:
            return 1.0 / i
    return 0.0


def metric_ap(ranked: list[int], relevant: set[int], k: int = K) -> float:
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for i, cid in enumerate(ranked[:k], start=1):
        if cid in relevant:
            hits += 1
            total += hits / i
    return total / len(relevant)


def metric_ndcg(ranked: list[int], relevant: set[int], k: int = K) -> float:
    dcg = 0.0
    for i, cid in enumerate(ranked[:k], start=1):
        if cid in relevant:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg


def compute_retrieval_metrics(ranked: list[int], relevant: set[int]) -> dict[str, float]:
    return {
        "hit@5": metric_hit(ranked, relevant),
        "precision@5": metric_precision(ranked, relevant),
        "recall@5": metric_recall(ranked, relevant),
        "mrr@5": metric_mrr(ranked, relevant),
        "map@5": metric_ap(ranked, relevant),
        "ndcg@5": metric_ndcg(ranked, relevant),
    }


RELEVANCE_SYSTEM = (
    "Bạn là giám khảo relevance cho benchmark RAG tiếng Việt.\n"
    "Cho câu hỏi, đáp án kỳ vọng, evidence mô tả trang/mục, doc_id đích, và các đoạn ứng viên.\n"
    "Chọn các đoạn chứa thông tin cần thiết để suy ra đáp án kỳ vọng. "
    "Ưu tiên đoạn thuộc đúng doc_id, nhưng vẫn chỉ chọn nếu nội dung thực sự liên quan.\n"
    'Trả về JSON: {"relevant_ids": [<chunk_id>]}'
)


def judge_relevant(llm: LLM, qa: QAItem, candidates: list[dict[str, Any]], cache: dict[str, Any]) -> set[int]:
    if qa.question_id in cache:
        return {int(x) for x in cache[qa.question_id]}
    chunks = []
    for c in candidates:
        chunks.append(
            f"[chunk_id={c['chunk_id']}] doc_source={c['doc_source']} page={c.get('page')}\n"
            f"{c['text'][:900]}"
        )
    user = (
        f"DOC_ID ĐÍCH: {qa.doc_id}\n"
        f"CÂU HỎI:\n{qa.question}\n\n"
        f"ĐÁP ÁN KỲ VỌNG:\n{qa.expected_answer}\n\n"
        f"EVIDENCE:\n{qa.evidence}\n\n"
        "ỨNG VIÊN:\n" + "\n\n".join(chunks)
    )
    data = llm.chat_json(
        [{"role": "system", "content": RELEVANCE_SYSTEM}, {"role": "user", "content": user}],
        fast=True,
        node="eval_relevance",
    )
    valid = {int(c["chunk_id"]) for c in candidates}
    relevant = [int(x) for x in data.get("relevant_ids", []) if int(x) in valid]
    cache[qa.question_id] = relevant
    return set(relevant)


RAGAS_PROXY_SYSTEM = (
    "Bạn là giám khảo chất lượng câu trả lời RAG tiếng Việt. "
    "Chấm mỗi tiêu chí từ 0 đến 1, chỉ dựa trên context, câu hỏi và đáp án kỳ vọng.\n"
    "faithfulness: câu trả lời có bám context không.\n"
    "answer_relevancy: câu trả lời có đúng trọng tâm câu hỏi không.\n"
    "answer_correctness: câu trả lời có đúng với đáp án kỳ vọng không.\n"
    "semantic_similarity: mức tương đồng ngữ nghĩa với đáp án kỳ vọng.\n"
    "context_precision: context được đưa vào có tập trung vào câu hỏi không.\n"
    "context_recall: context có đủ thông tin để trả lời không.\n"
    "context_entity_recall: các thực thể/số liệu/điều kiện quan trọng có xuất hiện trong context không.\n"
    "noise_sensitivity: mức câu trả lời bị ảnh hưởng bởi context nhiễu; 0 là tốt, 1 là xấu.\n"
    "Trả về JSON với đúng 8 khóa trên."
)


def score_ragas_proxy(llm: LLM, qa: QAItem, answer: str, contexts: list[str], cache: dict[str, Any]) -> dict[str, float]:
    if qa.question_id in cache:
        return {k: float(v) for k, v in cache[qa.question_id].items()}
    user = (
        f"CÂU HỎI:\n{qa.question}\n\n"
        f"ĐÁP ÁN KỲ VỌNG:\n{qa.expected_answer}\n\n"
        f"CÂU TRẢ LỜI HỆ THỐNG:\n{answer}\n\n"
        "CONTEXT:\n" + "\n\n".join(f"[{i+1}] {ctx[:1200]}" for i, ctx in enumerate(contexts))
    )
    data = llm.chat_json(
        [{"role": "system", "content": RAGAS_PROXY_SYSTEM}, {"role": "user", "content": user}],
        fast=False,
        node="eval_ragas_proxy",
    )
    keys = [
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
        "semantic_similarity",
        "context_precision",
        "context_recall",
        "context_entity_recall",
        "noise_sensitivity",
    ]
    scores = {key: max(0.0, min(1.0, float(data.get(key, 0.0)))) for key in keys}
    cache[qa.question_id] = scores
    return scores


def load_completed_rows(path: Path) -> list[dict[str, Any]]:
    """Load completed JSONL rows so interrupted evaluations can resume."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    bad_lines = 0
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            bad_lines += 1
            print(f"[resume] ignoring malformed row {line_no} in {path}", flush=True)
            continue
        question_id = ((row.get("question") or {}).get("question_id") or "").strip()
        if not question_id:
            bad_lines += 1
            print(f"[resume] ignoring row {line_no} without question_id in {path}", flush=True)
            continue
        rows.append(row)
    if bad_lines:
        print(f"[resume] loaded {len(rows)} completed row(s), ignored {bad_lines} bad row(s)", flush=True)
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def run_agent(agent: Agent, question: str) -> dict[str, Any]:
    started = time.perf_counter()
    final: dict[str, Any] | None = None
    events: list[dict[str, Any]] = []
    for event in agent.run(question):
        events.append(event)
        if event.get("type") == "final":
            final = event.get("data", {})
    elapsed = time.perf_counter() - started
    if final is None:
        raise RuntimeError(f"Agent produced no final answer for question: {question}")
    final["elapsed"] = elapsed
    final["events"] = [e for e in events if e.get("type") not in {"thinking", "token"}]
    return final


def compact_retrieved_chunks(chunks: list[dict[str, Any]], limit: int = K) -> list[dict[str, Any]]:
    """Keep the retrieved context auditable without dumping DB internals."""
    out: list[dict[str, Any]] = []
    for rank, chunk in enumerate(chunks[:limit], start=1):
        out.append(
            {
                "rank": rank,
                "chunk_id": chunk.get("chunk_id"),
                "document_id": chunk.get("document_id"),
                "doc_title": chunk.get("doc_title"),
                "doc_source": chunk.get("doc_source"),
                "page": chunk.get("page"),
                "section": chunk.get("section"),
                "score": chunk.get("score"),
                "rrf_score": chunk.get("rrf_score"),
                "bm25_score": chunk.get("bm25_score"),
                "dense_score": chunk.get("dense_score"),
                "rerank_score": chunk.get("rerank_score"),
                "text": chunk.get("text", ""),
            }
        )
    return out


def aggregate(rows: list[dict[str, Any]], metric_keys: list[str], prefix: str | None = None) -> dict[str, float]:
    out: dict[str, float] = {}
    if not rows:
        return {key: 0.0 for key in metric_keys}
    for key in metric_keys:
        values = []
        for row in rows:
            source = row[prefix] if prefix else row
            values.append(float(source.get(key, 0.0)))
        out[key] = sum(values) / len(values)
    return out


def write_report(path: Path, result: dict[str, Any]) -> None:
    retrieval = result["retrieval"]
    ragas = result["ragas"]
    by_diff = result["by_difficulty"]
    lines = [
        "# 50-Files PDF RAG Benchmark",
        "",
        f"- Questions: **{result['n_questions']}**",
        f"- Total latency: **{result['total_latency']:.2f}s**",
        f"- Avg latency: **{result['avg_latency']:.2f}s**",
        f"- Total tokens: **{result['total_tokens']}**",
        f"- Avg tokens: **{result['avg_tokens']:.1f}**",
        "- Relevance mode: **llm**",
        "- PDF parser: **Reducto only**",
        "- RAGAS mode: **llm_proxy**",
        "",
        "## Retrieval",
        "",
        "| Metric | Score |",
        "|---|---:|",
    ]
    for key in ["hit@5", "precision@5", "recall@5", "mrr@5", "map@5", "ndcg@5"]:
        lines.append(f"| {key} | {retrieval[key]:.3f} |")
    lines += ["", "## RAGAS", ""]
    for key in [
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
        "semantic_similarity",
        "context_precision",
        "context_recall",
        "context_entity_recall",
        "noise_sensitivity",
    ]:
        lines.append(f"- ragas_{key}: **{ragas[key]:.3f}**")
    lines += [
        "",
        "## By Difficulty",
        "",
        "| Difficulty | n | Hit@5 | Precision@5 | Recall@5 | MRR@5 | MAP@5 | NDCG@5 | Avg latency | Avg tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for diff in ["easy", "medium", "hard"]:
        row = by_diff.get(diff)
        if not row:
            continue
        lines.append(
            f"| {diff} | {row['n']} | {row['hit@5']:.3f} | {row['precision@5']:.3f} | "
            f"{row['recall@5']:.3f} | {row['mrr@5']:.3f} | {row['map@5']:.3f} | "
            f"{row['ndcg@5']:.3f} | {row['avg_latency']:.2f}s | {row['avg_tokens']:.1f} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the 50-files PDF benchmark with Reducto-only ingestion.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--pdf-dir", type=Path, default=None, help="Directory containing PDFs directly. Useful for ingest-only datasets without golden QA files.")
    parser.add_argument("--storage", type=Path, default=DEFAULT_STORAGE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--reducto-mode", choices=["default", "agentic"], default=os.getenv("REDUCTO_PARSE", "agentic"))
    parser.add_argument("--reset", action="store_true", help="Delete benchmark storage before ingesting.")
    parser.add_argument("--ingest-only", action="store_true", help="Run only mandatory Reducto ingest/indexing, then stop.")
    parser.add_argument("--doc-id", default="", help="Comma-separated doc IDs to ingest/evaluate, e.g. DOC001,DOC048.")
    parser.add_argument("--pdf-name", default="", help="Comma-separated exact PDF filenames to ingest/evaluate.")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N questions after ingest. 0 means all.")
    parser.add_argument("--judge-pool", type=int, default=20, help="Retrieved candidates judged for relevance.")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.reset and args.storage.exists():
        shutil.rmtree(args.storage)

    settings = make_settings(args.storage, args.reducto_mode)
    kb = KnowledgeBase(settings)

    pdfs = filter_manifest(
        load_document_manifest(args.pdf_dir.resolve() if args.pdf_dir else data_dir),
        doc_ids=args.doc_id,
        pdf_names=args.pdf_name,
    )
    dataset: list[QAItem] = []
    golden_dir = data_dir / "golden_dataset"
    qa_dir = args.pdf_dir.resolve() if args.pdf_dir else data_dir
    if golden_dir.exists() and not args.pdf_dir:
        dataset = load_50_files_dataset(golden_dir)
    elif qa_dir.exists() and list(qa_dir.glob("*.md")):
        dataset = load_markdown_qa_dataset(qa_dir, pdfs)
    elif not args.ingest_only:
        raise SystemExit(
            f"No QA markdown files found for {data_dir}. "
            "Use --ingest-only for PDF-only datasets."
        )
    if dataset:
        doc_ids_in_dataset = {qa.doc_id for qa in dataset}
        missing_pdf = sorted(doc_ids_in_dataset - set(pdfs))
        if missing_pdf and not (args.doc_id.strip() or args.pdf_name.strip() or args.pdf_dir):
            raise SystemExit(f"Missing PDFs for doc IDs: {missing_pdf}")
        if args.doc_id.strip() or args.pdf_name.strip() or args.pdf_dir:
            selected_doc_ids = set(pdfs)
            dataset = [qa for qa in dataset if qa.doc_id in selected_doc_ids]

    ingest_results = ensure_reducto_index(kb, pdfs)
    (out_dir / "ingest_results.json").write_text(
        json.dumps(ingest_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args.ingest_only:
        stats = kb.stats()
        (out_dir / "ingest_summary.json").write_text(
            json.dumps(
                {
                    "storage": str(settings.storage_dir),
                    "documents": stats.get("documents", 0),
                    "chunks": stats.get("chunks", 0),
                    "new_ingested": len(ingest_results),
                    "parser": "reducto",
                    "reducto_mode": settings.reducto_parse,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"[done] ingest-only: documents={stats.get('documents', 0)} "
            f"chunks={stats.get('chunks', 0)} summary={out_dir / 'ingest_summary.json'}",
            flush=True,
        )
        return

    if args.limit:
        dataset = dataset[: args.limit]

    relevance_cache_path = out_dir / "relevance_judgments.json"
    ragas_cache_path = out_dir / "ragas_proxy_judgments.json"
    relevance_cache = json.loads(relevance_cache_path.read_text(encoding="utf-8")) if relevance_cache_path.exists() else {}
    ragas_cache = json.loads(ragas_cache_path.read_text(encoding="utf-8")) if ragas_cache_path.exists() else {}

    judge_llm = LLM()
    ragas_llm = LLM()
    per_question_path = out_dir / "per_question.jsonl"
    rows: list[dict[str, Any]] = load_completed_rows(per_question_path)
    completed_question_ids = {
        row["question"]["question_id"]
        for row in rows
        if isinstance(row.get("question"), dict) and row["question"].get("question_id")
    }
    if completed_question_ids:
        print(
            f"[resume] found {len(completed_question_ids)} completed question(s) in {per_question_path}",
            flush=True,
        )

    for idx, qa in enumerate(dataset, start=1):
        if qa.question_id in completed_question_ids:
            print(f"[eval] {idx}/{len(dataset)} {qa.question_id}: skip completed", flush=True)
            continue
        print(f"[eval] {idx}/{len(dataset)} {qa.question_id}: {qa.question[:80]}", flush=True)
        retrieved = kb.retrieve(qa.question, top_k=max(args.judge_pool, K))
        candidates = [asdict(item) for item in retrieved]
        ranked_ids = [int(c["chunk_id"]) for c in candidates]
        relevant = judge_relevant(judge_llm, qa, candidates, relevance_cache)
        relevance_cache_path.write_text(json.dumps(relevance_cache, ensure_ascii=False, indent=2), encoding="utf-8")
        retrieval_metrics = compute_retrieval_metrics(ranked_ids, relevant)

        agent = Agent(kb)
        final = run_agent(agent, qa.question)
        usage = final.get("usage", {}) or {}
        total_tokens = int(usage.get("prompt_tokens", 0) or 0) + int(usage.get("completion_tokens", 0) or 0)
        contexts = [c.get("text", "") for c in final.get("citations", [])] or [c["text"] for c in candidates[:K]]
        ragas_scores = score_ragas_proxy(ragas_llm, qa, final.get("answer", ""), contexts, ragas_cache)
        ragas_cache_path.write_text(json.dumps(ragas_cache, ensure_ascii=False, indent=2), encoding="utf-8")

        row = {
            "question": asdict(qa),
            "retrieval": retrieval_metrics,
            "relevant_chunk_ids": sorted(relevant),
            "retrieved_chunk_ids": ranked_ids[:K],
            "retrieved_chunks": compact_retrieved_chunks(candidates, limit=K),
            "judge_pool_chunks": compact_retrieved_chunks(candidates, limit=args.judge_pool),
            "answer": final.get("answer", ""),
            "citations": final.get("citations", []),
            "route": final.get("route"),
            "partial": final.get("partial"),
            "iterations": final.get("iterations"),
            "latency": round(float(final["elapsed"]), 3),
            "tokens": total_tokens,
            "usage": usage,
            "ragas": ragas_scores,
        }
        rows.append(row)
        completed_question_ids.add(qa.question_id)
        append_jsonl(per_question_path, row)

    retrieval_keys = ["hit@5", "precision@5", "recall@5", "mrr@5", "map@5", "ndcg@5"]
    ragas_keys = [
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
        "semantic_similarity",
        "context_precision",
        "context_recall",
        "context_entity_recall",
        "noise_sensitivity",
    ]
    total_latency = sum(float(r["latency"]) for r in rows)
    total_tokens = sum(int(r["tokens"]) for r in rows)
    result = {
        "n_questions": len(rows),
        "total_latency": total_latency,
        "avg_latency": total_latency / max(len(rows), 1),
        "total_tokens": total_tokens,
        "avg_tokens": total_tokens / max(len(rows), 1),
        "retrieval": aggregate(rows, retrieval_keys, prefix="retrieval"),
        "ragas": aggregate(rows, ragas_keys, prefix="ragas"),
        "by_difficulty": {},
        "per_question_path": str(per_question_path),
    }
    for diff in sorted({r["question"]["difficulty"] for r in rows}):
        subset = [r for r in rows if r["question"]["difficulty"] == diff]
        metrics = aggregate(subset, retrieval_keys, prefix="retrieval")
        metrics["n"] = len(subset)
        metrics["avg_latency"] = sum(float(r["latency"]) for r in subset) / max(len(subset), 1)
        metrics["avg_tokens"] = sum(int(r["tokens"]) for r in subset) / max(len(subset), 1)
        result["by_difficulty"][diff] = metrics

    (out_dir / "benchmark_50_files_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out_dir / "benchmark_50_files_report.md", result)
    print(f"[done] report: {out_dir / 'benchmark_50_files_report.md'}", flush=True)


if __name__ == "__main__":
    main()
