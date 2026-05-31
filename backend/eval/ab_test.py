"""A/B test: compare old agent (no loop) vs new agent (with loop).

Runs both configurations on the same QA dataset, uses LLM-as-judge to evaluate
answer quality, and produces a comparison report.

Usage (from backend/):
    python -m eval.ab_test --test-dir ../test
    python -m eval.ab_test --test-dir ../test --reset   # rebuild index
"""
import argparse
import json
import time
from pathlib import Path
from typing import Any

from app.agent.graph import Agent
from app.config import Settings, get_settings
from app.indexing.store import KnowledgeBase
from eval.dataset import load_dataset

JUDGE_SYSTEM = (
    "Bạn là giám khảo đánh giá chất lượng câu trả lời của hệ thống hỏi đáp tài liệu tiếng Việt.\n"
    "Cho mỗi CÂU HỎI, ĐÁP ÁN MONG ĐỢI, và hai CÂU TRẢ LỜI (A và B), hãy đánh giá:\n"
    "1. accuracy: câu trả lời có chính xác so với đáp án mong đợi không? (0-10)\n"
    "2. completeness: câu trả lời có đầy đủ thông tin không? (0-10)\n"
    "3. groundedness: câu trả lời có bám nguồn (trích dẫn) hay bịa đặt không? (0-10)\n"
    "4. clarity: câu trả lời có rõ ràng, dễ hiểu không? (0-10)\n"
    "5. overall: điểm tổng thể (0-10)\n"
    "6. preference: 'A', 'B', hoặc 'tie'\n\n"
    "Trả về JSON:\n"
    '{"accuracy_a": 0-10, "accuracy_b": 0-10, '
    '"completeness_a": 0-10, "completeness_b": 0-10, '
    '"groundedness_a": 0-10, "groundedness_b": 0-10, '
    '"clarity_a": 0-10, "clarity_b": 0-10, '
    '"overall_a": 0-10, "overall_b": 0-10, '
    '"preference": "A"|"B"|"tie", '
    '"reason": "<giải thích ngắn bằng tiếng Việt>"}'
)


def _run_agent(kb: KnowledgeBase, question: str, config: dict) -> dict:
    """Run agent with specific config, collect final answer + metrics."""
    agent = Agent(kb)
    agent.settings.enable_replan = config.get("enable_replan", True)
    agent.settings.enable_sufficiency = config.get("enable_sufficiency", True)
    agent.settings.enable_answer_verify = config.get("enable_answer_verify", True)
    agent.settings.max_replan_iters = config.get("max_replan_iters", 3)
    agent.settings.max_answer_regenerations = config.get("max_answer_regenerations", 1)

    start = time.time()
    final = None
    events = []
    try:
        for ev in agent.run(question):
            events.append(ev)
            if ev.get("type") == "final":
                final = ev.get("data", {})
    except Exception as e:
        return {
            "answer": f"Lỗi: {e}",
            "route": "error",
            "iterations": 0,
            "partial": False,
            "elapsed": time.time() - start,
            "regenerated": False,
            "events": [],
            "error": str(e),
        }

    elapsed = time.time() - start
    if final is None:
        return {
            "answer": "",
            "route": "error",
            "iterations": 0,
            "partial": False,
            "elapsed": elapsed,
            "regenerated": False,
            "events": [e for e in events if e.get("type") != "thinking"],
            "error": "No final event",
        }

    return {
        "answer": final.get("answer", ""),
        "route": final.get("route", ""),
        "iterations": final.get("iterations", 0),
        "partial": final.get("partial", False),
        "elapsed": elapsed,
        "regenerated": False,
        "events": [e for e in events if e.get("type") != "thinking"],
        "n_events": len([e for e in events if e.get("type") not in ("thinking", "token")]),
    }


def _judge(kb: KnowledgeBase, qa, answer_a: str, answer_b: str, cache: dict, cache_path: Path) -> dict:
    """Use LLM-as-judge to compare two answers."""
    key = f"{qa.qid}"
    if key in cache:
        return cache[key]

    from app.agent.llm import LLM
    llm = LLM()

    user_msg = (
        f"CÂU HỎI:\n{qa.question}\n\n"
        f"ĐÁP ÁN MONG ĐỢI:\n{qa.expected}\n\n"
        f"CÂU TRẢ LỜI A:\n{answer_a}\n\n"
        f"CÂU TRẢ LỜI B:\n{answer_b}"
    )
    msgs = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    try:
        result = llm.chat_json(msgs, fast=False)
    except Exception as e:
        result = {
            "overall_a": 5, "overall_b": 5, "preference": "tie",
            "reason": f"Lỗi judge: {e}",
            "accuracy_a": 5, "accuracy_b": 5,
            "completeness_a": 5, "completeness_b": 5,
            "groundedness_a": 5, "groundedness_b": 5,
            "clarity_a": 5, "clarity_b": 5,
        }

    cache[key] = result
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2))
    return result


def _extract_event_metrics(events: list[dict]) -> dict:
    """Extract metrics from agent trace events."""
    metrics = {
        "n_replans": 0,
        "n_sufficiency_checks": 0,
        "n_answer_verifications": 0,
        "n_converged": 0,
        "n_early_stops": 0,
        "n_max_iters": 0,
        "n_distill": 0,
        "n_verify": 0,
        "route": "",
    }
    for e in events:
        t = e.get("type", "")
        if t == "replan":
            metrics["n_replans"] += 1
        elif t == "sufficiency":
            metrics["n_sufficiency_checks"] += 1
        elif t == "verify_answer":
            metrics["n_answer_verifications"] += 1
        elif t == "converged":
            metrics["n_converged"] += 1
        elif t == "early_stop":
            metrics["n_early_stops"] += 1
        elif t == "max_iters":
            metrics["n_max_iters"] += 1
        elif t == "distill":
            metrics["n_distill"] += 1
        elif t == "verify":
            metrics["n_verify"] += 1
        elif t == "route":
            metrics["route"] = e.get("data", {}).get("route", "")
    return metrics


def main():
    ap = argparse.ArgumentParser(description="A/B test: old agent vs new agent")
    ap.add_argument("--test-dir", default="../test")
    ap.add_argument("--storage", default="storage_eval_ab")
    ap.add_argument("--out-dir", default="../data/eval_ab")
    ap.add_argument("--reset", action="store_true", help="Rebuild index from scratch")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Configuration A: OLD agent (no replanning, no sufficiency, no answer verify) ---
    config_old = {
        "enable_replan": False,
        "enable_sufficiency": False,
        "enable_answer_verify": False,
        "max_replan_iters": 0,
        "max_answer_regenerations": 0,
    }
    # --- Configuration B: NEW agent (all features enabled) ---
    config_new = {
        "enable_replan": True,
        "enable_sufficiency": True,
        "enable_answer_verify": True,
        "max_replan_iters": 3,
        "max_answer_regenerations": 1,
    }

    settings = Settings()
    settings.storage_dir = Path(args.storage).resolve()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    if not settings.has_openai:
        raise SystemExit("OPENAI_API_KEY chưa cấu hình trong backend/.env")

    kb = KnowledgeBase(settings)
    test_dir = Path(args.test_dir).resolve()

    if args.reset:
        import shutil
        sp = Path(args.storage).resolve()
        if sp.exists():
            shutil.rmtree(sp)
        jp = out_dir / "judgments_ab.json"
        if jp.exists():
            jp.unlink()
        settings.storage_dir.mkdir(parents=True, exist_ok=True)
        kb = KnowledgeBase(Settings().__class__(**{k: v for k, v in settings.__dict__.items() if k != '_lru_cache__'})) if hasattr(settings, '__class__') else KnowledgeBase(settings)

    # Re-ingest if needed
    if not kb.repo.list_documents():
        pdf = next(test_dir.glob("*.pdf"), None)
        if pdf is None:
            raise SystemExit(f"Không tìm thấy PDF trong {test_dir}")
        print(f"Ingesting {pdf.name} ...")
        kb.ingest_pdf(pdf.read_bytes(), pdf.name)

    dataset = load_dataset(test_dir)
    print(f"Loaded {len(dataset)} câu hỏi.")

    cache_path = out_dir / "judgments_ab.json"
    judgments = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    results = []
    for qa in dataset:
        print(f"\n--- {qa.qid}: {qa.question[:60]}... ---")

        # Run OLD config
        print(f"  Running OLD agent...")
        result_old = _run_agent(kb, qa.question, config_old)
        print(f"  OLD: route={result_old.get('route','?')}, "
              f"partial={result_old.get('partial', False)}, "
              f"elapsed={result_old.get('elapsed', 0):.1f}s, "
              f"events={result_old.get('n_events', 0)}")

        # Run NEW config
        print(f"  Running NEW agent...")
        result_new = _run_agent(kb, qa.question, config_new)
        print(f"  NEW: route={result_new.get('route','?')}, "
              f"partial={result_new.get('partial', False)}, "
              f"iterations={result_new.get('iterations', 0)}, "
              f"elapsed={result_new.get('elapsed', 0):.1f}s, "
              f"events={result_new.get('n_events', 0)}")

        # Judge comparison
        judge_result = _judge(kb, qa, result_old["answer"], result_new["answer"], judgments, cache_path)

        row = {
            "qid": qa.qid,
            "question": qa.question,
            "difficulty": qa.difficulty,
            "old": {
                "answer": result_old["answer"][:500],
                "route": result_old.get("route", ""),
                "partial": result_old.get("partial", False),
                "elapsed": round(result_old.get("elapsed", 0), 2),
                "n_events": result_old.get("n_events", 0),
                "events": _extract_event_metrics(result_old.get("events", [])),
            },
            "new": {
                "answer": result_new["answer"][:500],
                "route": result_new.get("route", ""),
                "partial": result_new.get("partial", False),
                "iterations": result_new.get("iterations", 0),
                "elapsed": round(result_new.get("elapsed", 0), 2),
                "n_events": result_new.get("n_events", 0),
                "regenerated": result_new.get("regenerated", False),
                "events": _extract_event_metrics(result_new.get("events", [])),
            },
            "judge": judge_result,
        }
        results.append(row)
        pref = judge_result.get("preference", "tie")
        print(f"  JUDGE: {pref} | "
              f"overall_old={judge_result.get('overall_a', '?')} "
              f"overall_new={judge_result.get('overall_b', '?')} | "
              f"{judge_result.get('reason', '')[:80]}")

    # --- Aggregate results ---
    def _avg(key: str) -> tuple[float, float]:
        vals_old = [r["judge"].get(f"{key}_a", 0) for r in results]
        vals_new = [r["judge"].get(f"{key}_b", 0) for r in results]
        n = max(len(results), 1)
        return sum(vals_old) / n, sum(vals_new) / n

    metrics_names = ["accuracy", "completeness", "groundedness", "clarity", "overall"]
    summary = {}
    for m in metrics_names:
        old_avg, new_avg = _avg(m)
        summary[m] = {"old": round(old_avg, 2), "new": round(new_avg, 2), "delta": round(new_avg - old_avg, 2)}

    prefs = {"A": 0, "B": 0, "tie": 0}
    for r in results:
        p = r["judge"].get("preference", "tie")
        if p in prefs:
            prefs[p] += 1
        else:
            prefs["tie"] += 1

    elapsed_old = sum(r["old"]["elapsed"] for r in results)
    elapsed_new = sum(r["new"]["elapsed"] for r in results)

    report = {
        "n_questions": len(results),
        "summary": summary,
        "preferences": prefs,
        "avg_elapsed_old": round(elapsed_old / max(len(results), 1), 2),
        "avg_elapsed_new": round(elapsed_new / max(len(results), 1), 2),
        "elapsed_ratio": round(elapsed_new / max(elapsed_old, 0.01), 2),
        "per_question": results,
    }

    (out_dir / "ab_test_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )

    # --- Write Markdown report ---
    md = [
        "# A/B Test: OLD Agent vs NEW Agent (with Loop)",
        "",
        f"- Số câu hỏi: **{len(results)}**",
        f"- OLD: no replan, no sufficiency check, no answer verify",
        f"- NEW: replan (max 3 iters), sufficiency check, answer verify (+1 regen)",
        "",
        "## Tổng hợp",
        "",
        "| Metric | OLD | NEW | Delta |",
        "|---|---|---|---|",
    ]
    for m in metrics_names:
        s = summary[m]
        delta = s["delta"]
        sign = "+" if delta > 0 else ""
        md.append(f"| {m} | {s['old']} | {s['new']} | {sign}{delta} |")

    md += [
        "",
        "## Preference",
        "",
        f"| Preference | Count |",
        f"|---|---|",
        f"| OLD (A) | {prefs['A']} |",
        f"| NEW (B) | {prefs['B']} |",
        f"| Tie | {prefs['tie']} |",
        "",
        "## Latency",
        "",
        f"| | OLD | NEW | Ratio |",
        f"|---|---|---|---|",
        f"| Avg per question | {report['avg_elapsed_old']}s | {report['avg_elapsed_new']}s | {report['elapsed_ratio']}x |",
        "",
    ]

    # Per-difficulty breakdown
    for diff in sorted({r["difficulty"] for r in results}):
        diff_rows = [r for r in results if r["difficulty"] == diff]
        md.append(f"## Theo độ khó: {diff}")
        md.append("")
        md.append("| Q | Route OLD | Route NEW | Overall OLD | Overall NEW | Pref | Reason |")
        md.append("|---|---|---|---|---|---|---|")
        for r in diff_rows:
            j = r["judge"]
            md.append(
                f"| {r['qid']} | {r['old']['route']} | {r['new']['route']} | "
                f"{j.get('overall_a', '?')} | {j.get('overall_b', '?')} | "
                f"{j.get('preference', '?')} | {j.get('reason', '')[:60]} |"
            )
        md.append("")

    md += [
        "## Phân tích mới (NEW Agent features)",
        "",
    ]
    new_feats = {
        "replan_used": sum(1 for r in results if r["new"]["events"].get("n_replans", 0) > 0),
        "sufficiency_used": sum(1 for r in results if r["new"]["events"].get("n_sufficiency_checks", 0) > 0),
        "answer_verify_used": sum(1 for r in results if r["new"]["events"].get("n_answer_verifications", 0) > 0),
        "converged": sum(1 for r in results if r["new"]["events"].get("n_converged", 0) > 0),
        "early_stop": sum(1 for r in results if r["new"]["events"].get("n_early_stops", 0) > 0),
        "max_iters": sum(1 for r in results if r["new"]["events"].get("n_max_iters", 0) > 0),
        "partial_answer": sum(1 for r in results if r["new"].get("partial", False)),
    }
    feat_labels = {
        "replan_used": "Sử dụng replanning",
        "sufficiency_used": "Sử dụng sufficiency check",
        "answer_verify_used": "Sử dụng answer verify",
        "converged": "Converged (tất cả grounded)",
        "early_stop": "Early stop (không cải thiện)",
        "max_iters": "Đạt max iterations",
        "partial_answer": "Câu trả lời partial",
    }
    md.append("| Feature | Count / Total |")
    md.append("|---|---|")
    for feat, count in new_feats.items():
        md.append(f"| {feat_labels[feat]} | {count} / {len(results)} |")

    (out_dir / "ab_test_report.md").write_text("\n".join(md), encoding="utf-8")

    # Print summary
    print(f"\n{'='*60}")
    print(f"A/B Test Results: {len(results)} questions")
    print(f"{'='*60}")
    for m in metrics_names:
        s = summary[m]
        sign = "+" if s["delta"] > 0 else ""
        print(f"  {m:15s}: OLD={s['old']:.2f}  NEW={s['new']:.2f}  Δ={sign}{s['delta']:.2f}")
    print(f"\n  Preferences: OLD={prefs['A']}  NEW={prefs['B']}  Tie={prefs['tie']}")
    print(f"  Avg latency:  OLD={report['avg_elapsed_old']}s  NEW={report['avg_elapsed_new']}s  (ratio={report['elapsed_ratio']}x)")
    print(f"\nReport saved to: {out_dir / 'ab_test_report.md'}")
    print(f"Full results: {out_dir / 'ab_test_results.json'}")


if __name__ == "__main__":
    main()