"""Run workflow scenario tests — exercise every path in the agent graph.

Tests all 8 scenarios:
  1. no_retrieval (chitchat)
  2. simple (single-hop)
  3. complex — converged first time
  4. complex — replan needed (subquestions fail)
  5. complex — sufficiency check fails
  6. complex — answer not found
  7. complex — verify answer detects hallucination
  8. complex — early stop / max iterations

Usage (from backend/):
    python -m eval.run_workflow_test
    python -m eval.run_workflow_test --only 1,2,3       # run specific scenarios
"""
import argparse
import json
import time
from pathlib import Path
from typing import Optional

from app.agent.graph import Agent
from app.config import Settings
from app.indexing.store import KnowledgeBase


SCENARIOS = [
    {
        "id": 1,
        "name": "Chitchat (no_retrieval)",
        "questions": [
            "Xin chào",
            "Bạn là ai?",
            "Cảm ơn bạn nhiều",
        ],
        "expect_route": "no_retrieval",
        "expect_events": ["route", "final"],
        "expect_not_events": ["retrieve", "plan", "distill", "verify", "synthesize"],
    },
    {
        "id": 2,
        "name": "Simple — Single-hop",
        "questions": [
            "Chỉ số an toàn S(c) phụ thuộc vào những yếu tố nào?",
            "Đầu ra của bài toán quy hoạch đường đi là gì?",
            "Hàm chi phí cục bộ f(x) đóng vai trò gì?",
        ],
        "expect_route": "simple",
        "expect_events": ["route", "retrieve", "synthesize", "verify_answer"],
        "expect_not_events": ["plan", "distill", "verify", "sufficiency", "replan"],
    },
    {
        "id": 3,
        "name": "Complex — Converged (all grounded)",
        "questions": [
            "So sánh độ rủi ro va chạm risk(P) và độ rủi ro phóng xạ R(P), chúng đo lường những khía cạnh khác nhau nào của đường đi?",
            "Bài toán quy hoạch đường đi an toàn đa đích có đầu vào và đầu ra như thế nào?",
        ],
        "expect_route": "complex",
        "expect_events": ["route", "plan", "retrieve", "distill", "verify"],
        "expect_may_events": ["sufficiency", "converged", "verify_answer"],
    },
    {
        "id": 4,
        "name": "Complex — Replan needed",
        "questions": [
            "So sánh chiến lược kinh doanh của Vinamilk và TH True Milk trong giai đoạn 2010-2020",
        ],
        "expect_route": "complex",
        "expect_may_events": ["replan", "max_iters", "early_stop", "sufficiency"],
        "expect_partial": True,
    },
    {
        "id": 5,
        "name": "Complex — Sufficiency check",
        "questions": [
            "Công thức giao thoa Dij được xác định như thế nào và tại sao cần hạng tử trung bình (f(x)+f(y))/2 thay vì chỉ f(x)?",
        ],
        "expect_route": "complex",
        "expect_may_events": ["sufficiency", "replan"],
    },
    {
        "id": 6,
        "name": "Complex — Answer not found",
        "questions": [
            "Ai là tác giả của bài báo khoa học này và nó được xuất bản năm nào?",
            "Phương pháp gradient descent được sử dụng như thế nào trong bài toán này?",
        ],
        "expect_may_route": ["simple", "complex"],
        "expect_may_events": ["sufficiency", "replan"],
        "expect_partial_or_notfound": True,
    },
    {
        "id": 7,
        "name": "Complex — Verify answer (hallucination detection)",
        "questions": [
            "Trong công thức S(c), C1 có giá trị mặc định là bao nhiêu và nó được tối ưu như thế nào?",
        ],
        "expect_route": "complex",
        "expect_may_events": ["verify_answer"],
    },
    {
        "id": 8,
        "name": "Complex — Early stop / Max iterations",
        "questions": [
            "Phân tích ảnh hưởng của việc tăng w2 đến đường đi, so sánh với ảnh hưởng của việc tăng w3, và giải thích tương tác giữa chúng khi cả hai đều tăng",
        ],
        "expect_route": "complex",
        "expect_may_events": ["replan", "max_iters", "early_stop"],
    },
]


def run_question(agent: Agent, question: str) -> dict:
    """Run agent on a question and collect all events."""
    events = []
    final = None
    start = time.time()

    try:
        for ev in agent.run(question):
            events.append(ev)
            if ev.get("type") == "final":
                final = ev.get("data", {})
    except Exception as e:
        return {
            "error": str(e),
            "events": [],
            "final": None,
            "elapsed": time.time() - start,
        }

    elapsed = time.time() - start
    key_events = [e for e in events if e["type"] not in ("thinking", "token")]
    event_types = [e["type"] for e in key_events]

    return {
        "error": None,
        "events": key_events,
        "event_types": event_types,
        "final": final,
        "elapsed": round(elapsed, 2),
        "route": final.get("route", "unknown") if final else "error",
        "iterations": final.get("iterations", 0) if final else 0,
        "partial": final.get("partial", False) if final else False,
        "answer_preview": (final.get("answer", "") or "")[:200] if final else "",
    }


def main():
    ap = argparse.ArgumentParser(description="Workflow scenario tests for agent graph")
    ap.add_argument("--only", default=None, help="Comma-separated scenario IDs to run (e.g. '1,2,3')")
    ap.add_argument("--no-verify", action="store_true", help="Disable answer verify (faster)")
    ap.add_argument("--no-sufficiency", action="store_true", help="Disable sufficiency check")
    ap.add_argument("--no-replan", action="store_true", help="Disable replanning")
    ap.add_argument("--out", default=str(Path(__file__).parent.parent / "data" / "workflow_test"),
                    help="Output directory")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Setup
    settings = Settings()
    if not settings.has_openai:
        raise SystemExit("OPENAI_API_KEY chưa cấu hình trong backend/.env")
    kb = KnowledgeBase(settings)

    if not kb.repo.list_documents():
        raise SystemExit("Chưa nạp tài liệu. Chạy `python -m eval.run_eval --test-dir ../test --reset` trước.")

    only_ids = None
    if args.only:
        only_ids = set(int(x.strip()) for x in args.only.split(","))

    results = []
    total = 0
    passed = 0

    for scenario in SCENARIOS:
        if only_ids and scenario["id"] not in only_ids:
            continue

        print(f"\n{'='*60}")
        print(f"Kịch bản {scenario['id']}: {scenario['name']}")
        print(f"{'='*60}")

        for q in scenario["questions"]:
            total += 1
            agent = Agent(kb)
            agent.settings.enable_replan = not args.no_replan
            agent.settings.enable_sufficiency = not args.no_sufficiency
            agent.settings.enable_answer_verify = not args.no_verify
            agent.settings.max_replan_iters = 3
            agent.settings.max_answer_regenerations = 1

            print(f"\n  Q: {q}")
            result = run_question(agent, q)

            if result["error"]:
                print(f"  ❌ ERROR: {result['error'][:80]}")
                results.append({"scenario": scenario["id"], "question": q, **result})
                continue

            route = result["route"]
            event_types = result["event_types"]
            print(f"  Route: {route}")
            print(f"  Events: {event_types}")
            print(f"  Iterations: {result['iterations']}, Partial: {result['partial']}")
            print(f"  Elapsed: {result['elapsed']}s")
            print(f"  Answer: {result['answer_preview'][:100]}...")

            # Check expectations
            checks = []

            # Route check
            if "expect_route" in scenario:
                ok = route == scenario["expect_route"]
                checks.append(("route", ok, f"expected={scenario['expect_route']}, got={route}"))
            if "expect_may_route" in scenario:
                ok = route in scenario["expect_may_route"]
                checks.append(("route", ok, f"expected one of {scenario['expect_may_route']}, got={route}"))

            # Events that SHOULD be present
            if "expect_events" in scenario:
                for ev in scenario["expect_events"]:
                    ok = ev in event_types
                    checks.append((f"event:{ev}", ok, f"{'FOUND' if ok else 'MISSING'}"))

            # Events that should NOT be present
            if "expect_not_events" in scenario:
                for ev in scenario["expect_not_events"]:
                    ok = ev not in event_types
                    checks.append((f"no_event:{ev}", ok, f"{'NOT FOUND (ok)' if ok else 'FOUND (bad)'}"))

            # Events that MAY be present
            if "expect_may_events" in scenario:
                found = [ev for ev in scenario["expect_may_events"] if ev in event_types]
                checks.append(("may_events", True, f"found: {found}"))

            # Partial / not found
            if scenario.get("expect_partial"):
                ok = result["partial"] or route == "not_found"
                checks.append(("partial", ok, f"partial={result['partial']}, route={route}"))
            if scenario.get("expect_partial_or_notfound"):
                ok = result["partial"] or route == "not_found" or "Không tìm thấy" in result.get("answer_preview", "")
                checks.append(("partial_or_notfound", ok, ""))

            # Verify answer present
            if "verify_answer" in event_types:
                for ev_data in [e["data"] for e in result["events"] if e["type"] == "verify_answer"]:
                    checks.append(("verify_answer_result", True, f"grounded={ev_data.get('grounded')}"))

            all_ok = all(c[1] for c in checks)
            passed += 1 if all_ok else 0

            for name, ok, detail in checks:
                print(f"    {'✓' if ok else '✗'} {name}: {detail}")

            results.append({"scenario": scenario["id"], "question": q, **result, "checks": checks})

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {passed}/{total} checks passed")
    print(f"{'='*60}")

    (out_dir / "workflow_test_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2)
    )
    print(f"Results saved to: {out_dir / 'workflow_test_results.json'}")

    return passed == total


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)