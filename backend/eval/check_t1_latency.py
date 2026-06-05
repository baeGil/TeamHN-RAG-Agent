"""Quick lookup: T1 latency_s for any item in partial_results.jsonl.

Usage (from backend/):
    python -m eval.check_t1_latency                        # summary + all rows
    python -m eval.check_t1_latency --item 24              # by line number (1-based)
    python -m eval.check_t1_latency --item 024             # same
    python -m eval.check_t1_latency --item context_retrieval_benchmark_table_md_024
    python -m eval.check_t1_latency --all                  # print every row
"""
import argparse
import json
import statistics
from pathlib import Path

CONFIGS = ["baseline", "T1_query_transform", "T4_ctxt_headers", "T8_rse", "T10_hierarchical"]
LABELS  = ["Baseline", "T1", "T4", "T8", "T10"]
DEFAULT_FILE = "../data/ab_golden5_imported8_resumable/partial_results.jsonl"


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                try:
                    obj = json.loads(s)
                    if obj.get("status") == "complete":
                        rows.append(obj)
                except json.JSONDecodeError:
                    pass
    return rows


def _row_latencies(row: dict) -> dict:
    return {cfg: row["configs"].get(cfg, {}).get("latency_s", None) for cfg in CONFIGS}


def _print_row(idx: int, row: dict) -> None:
    lats = _row_latencies(row)
    t1   = lats.get("T1_query_transform")
    bl   = lats.get("baseline")
    flag = ""
    if t1 is not None and bl is not None:
        flag = "  [OLD]" if t1 > bl * 1.4 else "  [OK]"
    print(f"[{idx:>4}] {row['item_id']}")
    for cfg, lbl in zip(CONFIGS, LABELS):
        v = lats.get(cfg)
        marker = " <-- T1" if cfg == "T1_query_transform" else ""
        print(f"       {lbl:<8}: {v:>7.2f}s{marker}" if v is not None else f"       {lbl:<8}: N/A")
    print(f"       T1 status : {flag.strip()}")


def _print_summary(rows: list[dict]) -> None:
    t1_all = [r["configs"]["T1_query_transform"]["latency_s"] for r in rows
              if "T1_query_transform" in r["configs"]]
    bl_all = [r["configs"]["baseline"]["latency_s"] for r in rows
              if "baseline" in r["configs"]]

    old = [v for v in t1_all if v > 30]
    new = [v for v in t1_all if v <= 30]

    print(f"Total rows : {len(rows)}")
    print(f"T1 mean    : {statistics.mean(t1_all):.2f}s  "
          f"median={statistics.median(t1_all):.2f}s  "
          f"min={min(t1_all):.2f}s  max={max(t1_all):.2f}s")
    print(f"BL mean    : {statistics.mean(bl_all):.2f}s  "
          f"median={statistics.median(bl_all):.2f}s")
    print(f"Patched (<=30s): {len(new)}  |  Old (>30s): {len(old)}")
    if new:
        print(f"  Patched mean={statistics.mean(new):.2f}s  median={statistics.median(new):.2f}s")
    if old:
        print(f"  Old     mean={statistics.mean(old):.2f}s  median={statistics.median(old):.2f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file",  default=DEFAULT_FILE)
    ap.add_argument("--item",  default=None,
                    help="Line number (e.g. 24) or item_id suffix (e.g. 024) or full item_id")
    ap.add_argument("--all",   action="store_true", help="Print every row")
    args = ap.parse_args()

    src  = Path(args.file).resolve()
    rows = load_rows(src)

    if not rows:
        raise SystemExit("No completed rows found.")

    if args.item is None and not args.all:
        _print_summary(rows)
        return

    if args.all:
        _print_summary(rows)
        print()
        for i, row in enumerate(rows, start=1):
            _print_row(i, row)
        return

    # -- lookup by --item -------------------------------------------------------
    q = args.item.strip()

    # try as line number
    if q.isdigit():
        n = int(q)
        if 1 <= n <= len(rows):
            _print_row(n, rows[n - 1])
            return
        # maybe it's a zero-padded suffix like "024"
        suffix = q.zfill(3)
        matches = [i for i, r in enumerate(rows) if r["item_id"].endswith(f"_{suffix}")]
        if matches:
            for i in matches:
                _print_row(i + 1, rows[i])
            return
        raise SystemExit(f"No row at line {n} (total={len(rows)})")

    # try as full item_id or suffix
    matches = [i for i, r in enumerate(rows)
               if r["item_id"] == q or r["item_id"].endswith(f"_{q}")]
    if not matches:
        raise SystemExit(f"item_id not found: {q}")
    for i in matches:
        _print_row(i + 1, rows[i])


if __name__ == "__main__":
    main()
