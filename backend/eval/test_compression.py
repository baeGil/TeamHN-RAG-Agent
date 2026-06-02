"""Contextual Compression — Full Benchmark using testQA.md & testQA_hard.md
Usage (from backend/):
    USE_RERANKER=false python -m eval.test_compression --both
    USE_RERANKER=false python -m eval.test_compression          # uses ENABLE_SIMPLE_COMPRESSION env var
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# Force disable reranker to avoid downloading the 2.27 GB model
os.environ.setdefault("USE_RERANKER", "false")

# ────────────────────────────────────────────────────────────────────────────
# Parse testQA markdown tables
# ────────────────────────────────────────────────────────────────────────────

def parse_md_table(md_path: Path) -> list[dict]:
    """Parse a markdown table with columns ID | Question | Expected."""
    rows = []
    for line in md_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|") or line.startswith("| ID") or line.startswith("|---"):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 3:
            continue
        try:
            qid = int(parts[0])
        except ValueError:
            continue
        rows.append({"id": qid, "question": parts[1], "expected": parts[2]})
    return rows


TEST_DIR = Path(__file__).resolve().parent.parent.parent / "test"
STANDARD_QA = parse_md_table(TEST_DIR / "testQA.md")
HARD_QA     = parse_md_table(TEST_DIR / "testQA_hard.md")
ALL_QA = [
    {**q, "difficulty": "standard"} for q in STANDARD_QA
] + [
    {**q, "difficulty": "hard"} for q in HARD_QA
]

# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _get_compression_event(events):
    for e in events:
        if e.get("type") == "compression":
            return e["data"]
    return None


def _get_event(events, etype):
    return next((e for e in events if e.get("type") == etype), None)


# ────────────────────────────────────────────────────────────────────────────
# Core runner
# ────────────────────────────────────────────────────────────────────────────

def run_mode(compression_enabled: bool, questions: list[dict]) -> list[dict]:
    import app.config as cfg_mod
    cfg_mod.get_settings.cache_clear()
    os.environ["ENABLE_SIMPLE_COMPRESSION"] = "true" if compression_enabled else "false"
    cfg_mod.get_settings.cache_clear()

    settings = cfg_mod.get_settings()
    from app.indexing.store import KnowledgeBase
    from app.agent.graph import Agent
    kb = KnowledgeBase(settings)
    agent = Agent(kb)

    mode_label = "WITH compression" if compression_enabled else "WITHOUT compression (baseline)"
    print(f"\n{'='*70}")
    print(f"  MODE: {mode_label}")
    print(f"{'='*70}")

    results = []
    for item in questions:
        q      = item["question"]
        exp    = item["expected"]
        diff   = item["difficulty"]
        qid    = item["id"]
        prefix = f"[{diff.upper()[:1]}{qid:02d}]"

        print(f"\n  {prefix} {q[:75]}")
        t0 = time.perf_counter()
        events  = []
        answer  = ""
        for ev in agent.run(q):
            events.append(ev)
            if ev["type"] == "final":
                answer = ev["data"].get("answer", "")
        elapsed = time.perf_counter() - t0

        compression_data = _get_compression_event(events)
        route_ev = _get_event(events, "route")
        route    = route_ev["data"].get("route", "?") if route_ev else "?"
        verify_ev = _get_event(events, "verify_answer")
        grounded  = verify_ev["data"].get("grounded") if verify_ev else None

        if compression_data:
            b4    = compression_data["chars_before"]
            after = compression_data["chars_after"]
            ratio = compression_data["compression_ratio"]
            print(f"  Route={route} | {b4}→{after} chars ({ratio*100:.1f}% ↓) | {elapsed:.1f}s | grounded={grounded}")
        else:
            print(f"  Route={route} | no compression | {elapsed:.1f}s | grounded={grounded}")
        print(f"  Answer: {answer[:100].strip()}...")

        results.append({
            "id": qid,
            "difficulty": diff,
            "question": q,
            "expected": exp,
            "route": route,
            "elapsed_s": round(elapsed, 2),
            "compression_enabled": compression_enabled,
            "compression": compression_data,
            "grounded": grounded,
            "answer": answer,
            "answer_chars": len(answer),
        })

    return results


# ────────────────────────────────────────────────────────────────────────────
# Report helpers
# ────────────────────────────────────────────────────────────────────────────

def print_comparison(baseline: list[dict], compressed: list[dict]):
    print(f"\n\n{'='*90}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*90}")

    total_b4 = total_after = 0
    regressions = []
    improvements = []

    hdr = f"{'ID':<6} {'Difficulty':<10} {'Baseline Grnd':<15} {'Compressed Grnd':<17} {'Chars Before':>13} {'Chars After':>12} {'Saved%':>8}"
    print(hdr)
    print("-" * 90)

    for b, c in zip(baseline, compressed):
        tag = f"{b['difficulty'][0].upper()}{b['id']:02d}"
        b_chars = b["compression"]["chars_before"] if b["compression"] else None
        c_chars = c["compression"]["chars_after"]  if c["compression"] else None

        saved_str = ""
        if b_chars is None:
            # Baseline has no compression event — use chars_before from compressed run
            b_chars = c["compression"]["chars_before"] if c["compression"] else None
        if b_chars and c_chars:
            saved = (1 - c_chars / b_chars) * 100
            saved_str = f"{saved:.1f}%"
            total_b4    += b_chars
            total_after += c_chars

        b_grnd_sym = "✅" if b["grounded"] else ("❌" if b["grounded"] is False else "—")
        c_grnd_sym = "✅" if c["grounded"] else ("❌" if c["grounded"] is False else "—")

        regression  = (b["grounded"] is True and c["grounded"] is False)
        improvement = (b["grounded"] is False and c["grounded"] is True)
        marker = " ← REGRESSION" if regression else (" ← IMPROVEMENT" if improvement else "")
        if regression:
            regressions.append(c)
        if improvement:
            improvements.append(c)

        print(f"  {tag:<6} {b['difficulty']:<10} {b_grnd_sym:<15} {c_grnd_sym:<17} "
              f"{str(b_chars or 'N/A'):>13} {str(c_chars or 'N/A'):>12} {saved_str:>8}{marker}")

    print()
    if total_b4:
        overall = (1 - total_after / total_b4) * 100
        print(f"  Overall context reduction: {total_b4:,} → {total_after:,} chars ({overall:.1f}% saved)")
    print(f"  Regressions  (True→False): {len(regressions)}")
    print(f"  Improvements (False→True): {len(improvements)}")

    if regressions:
        print("\n  ─── REGRESSIONS ───")
        for r in regressions:
            print(f"    [{r['difficulty'][0].upper()}{r['id']:02d}] {r['question'][:80]}")
    if improvements:
        print("\n  ─── IMPROVEMENTS ───")
        for r in improvements:
            print(f"    [{r['difficulty'][0].upper()}{r['id']:02d}] {r['question'][:80]}")


def diff_stats(baseline: list[dict], compressed: list[dict]) -> dict:
    b_grnd = [b["grounded"] for b in baseline]
    c_grnd = [c["grounded"] for c in compressed]

    def pass_rate(lst):
        true_count = sum(1 for x in lst if x is True)
        total = len([x for x in lst if x is not None])
        return true_count, total

    b_pass, b_total = pass_rate(b_grnd)
    c_pass, c_total = pass_rate(c_grnd)
    return {
        "baseline_grounding": f"{b_pass}/{b_total}",
        "compressed_grounding": f"{c_pass}/{c_total}",
        "regressions": sum(1 for b, c in zip(b_grnd, c_grnd) if b is True and c is False),
        "improvements": sum(1 for b, c in zip(b_grnd, c_grnd) if b is False and c is True),
    }


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Contextual Compression benchmark")
    parser.add_argument("--both", action="store_true", help="Run baseline + compressed and compare")
    parser.add_argument(
        "--set",
        choices=["standard", "hard", "all"],
        default="all",
        help="Which question set to use (default: all)",
    )
    args = parser.parse_args()

    STANDARD_QA_TAGGED = [{**q, "difficulty": "standard"} for q in STANDARD_QA]
    HARD_QA_TAGGED     = [{**q, "difficulty": "hard"}     for q in HARD_QA]
    questions = {
        "standard": STANDARD_QA_TAGGED,
        "hard":     HARD_QA_TAGGED,
        "all":      ALL_QA,
    }[args.set]

    out_dir = Path("data/compression_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.both:
        print(f"\nRunning {len(questions)} questions × 2 modes = {len(questions)*2} total calls")
        baseline   = run_mode(False, questions)
        compressed = run_mode(True,  questions)
        print_comparison(baseline, compressed)
        stats = diff_stats(baseline, compressed)
        all_results = {
            "baseline":   baseline,
            "compressed": compressed,
            "stats":      stats,
        }
        out_path = out_dir / "results_both.json"
        out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  Results saved to: {out_path}")
    else:
        enabled  = os.environ.get("ENABLE_SIMPLE_COMPRESSION", "true").lower() == "true"
        results  = run_mode(enabled, questions)
        out_path = out_dir / ("results_compressed.json" if enabled else "results_baseline.json")
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
