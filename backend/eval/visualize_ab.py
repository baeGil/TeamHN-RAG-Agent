"""Visualize A/B retrieval evaluation results from a partial_results.jsonl file.

Usage (from backend/):
    python -m eval.visualize_ab
    python -m eval.visualize_ab --file ../data/ab_resumable_smoke/partial_results.jsonl
    python -m eval.visualize_ab --file ../data/ab_golden5_imported8_resumable/partial_results.jsonl

Output (12 PNG files in the same directory as --file):
    01_retrieval_quality_k5.png     – Recall/MRR/Hit/NDCG @5
    02_retrieval_quality_k10.png    – Recall/MRR/Hit/NDCG @10
    03_k5_vs_k10.png                – @5 vs @10 side-by-side per method
    04_precision.png                – Precision@5 and @10
    05_context_quality.png          – Context Precision / Recall / Redundancy
    06_latency_box.png              – Latency box+strip plot per method
    07_latency_breakdown.png        – Stacked bar: BM25 / Dense / Fusion / Rerank ms
    08_token_cost.png               – LLM + Embedding token cost per method
    09_heatmap_recall5.png          – Per-question × method heatmap (Recall@5)
    10_heatmap_hit5.png             – Per-question × method heatmap (Hit@5)
    11_heatmap_ndcg5.png            – Per-question × method heatmap (NDCG@5)
    12_by_difficulty.png            – Recall/MRR/Hit/NDCG @5 grouped by difficulty
    13_radar.png                    – Radar chart (5 metrics)
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── constants ─────────────────────────────────────────────────────────────────
METHOD_ORDER = [
    "baseline",
    "T1_query_transform",
    "T4_ctxt_headers",
    "T8_rse",
    "T10_hierarchical",
]
METHOD_LABELS = {
    "baseline":           "Baseline",
    "T1_query_transform": "T1 Query\nTransform",
    "T4_ctxt_headers":    "T4 Ctxt\nHeaders",
    "T8_rse":             "T8 RSE",
    "T10_hierarchical":   "T10 Hierarchical",
}
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
LATENCY_COLORS = {"bm25_ms": "#5B9BD5", "dense_ms": "#ED7D31",
                  "fusion_ms": "#A9D18E", "rerank_ms": "#FF0000"}


# ── data loading ──────────────────────────────────────────────────────────────
def load_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                obj = json.loads(s)
                if obj.get("status") == "complete":
                    rows.append(obj)
    return rows


def _methods_present(rows: list[dict]) -> list[str]:
    return [m for m in METHOD_ORDER
            if all(m in r.get("configs", {}) for r in rows)]


def agg(rows: list[dict], method: str, metric: str) -> list[float]:
    return [r["configs"][method][metric]
            for r in rows if method in r.get("configs", {})
            and metric in r["configs"][method]]


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _color(i: int) -> str:
    return COLORS[i % len(COLORS)]


# ── shared bar helper ─────────────────────────────────────────────────────────
def _grouped_bar(ax, methods, avg_dict, metric_keys, xlabels, title, ylim=1.15):
    x = np.arange(len(metric_keys))
    n = len(methods)
    w = 0.75 / n
    for i, method in enumerate(methods):
        vals = [avg_dict[method].get(k, 0.0) for k in metric_keys]
        bars = ax.bar(x + i * w - (n - 1) * w / 2, vals,
                      width=w * 0.9, color=_color(i),
                      label=METHOD_LABELS.get(method, method))
        for bar, v in zip(bars, vals):
            if v > 0.005:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylim(0, ylim)
    ax.set_title(title, fontweight="bold", pad=8)
    ax.set_ylabel("Score")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


# ── fig 01 & 02 : quality @5 and @10 ─────────────────────────────────────────
def fig_quality(rows, methods, out_dir, k: int, fig_num: int):
    metrics = [f"recall@{k}", f"mrr@{k}", f"hit@{k}", f"ndcg@{k}"]
    labels  = [f"Recall@{k}", f"MRR@{k}", f"Hit@{k}", f"NDCG@{k}"]
    avg = {m: {met: mean(agg(rows, m, met)) for met in metrics} for m in methods}

    fig, ax = plt.subplots(figsize=(10, 5))
    _grouped_bar(ax, methods, avg, metrics, labels,
                 f"Retrieval Quality @{k}  (n={len(rows)} questions)")
    fig.tight_layout()
    path = out_dir / f"{fig_num:02d}_retrieval_quality_k{k}.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path.name}")


# ── fig 03 : @5 vs @10 comparison ────────────────────────────────────────────
def fig_k5_vs_k10(rows, methods, out_dir):
    base_metrics = ["recall", "mrr", "hit", "ndcg"]
    fig, axes = plt.subplots(1, len(base_metrics), figsize=(14, 5), sharey=False)
    for ax, bm in zip(axes, base_metrics):
        x = np.arange(len(methods))
        w = 0.35
        v5 = [mean(agg(rows, m, f"{bm}@5"))  for m in methods]
        v10= [mean(agg(rows, m, f"{bm}@10")) for m in methods]
        b5  = ax.bar(x - w / 2, v5,  width=w, color="#4C72B0", label="@5",  alpha=0.85)
        b10 = ax.bar(x + w / 2, v10, width=w, color="#DD8452", label="@10", alpha=0.85)
        for bar, v in list(zip(b5, v5)) + list(zip(b10, v10)):
            if v > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=6)
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS.get(m, m).replace("\n", "\n") for m in methods],
                           fontsize=7)
        ax.set_ylim(0, 1.18)
        ax.set_title(bm.upper(), fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"@5 vs @10 Comparison  (n={len(rows)} questions)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "03_k5_vs_k10.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path.name}")


# ── fig 04 : precision ────────────────────────────────────────────────────────
def fig_precision(rows, methods, out_dir):
    metrics = ["precision@5", "precision@10"]
    labels  = ["Precision@5", "Precision@10"]
    avg = {m: {met: mean(agg(rows, m, met)) for met in metrics} for m in methods}

    fig, ax = plt.subplots(figsize=(8, 5))
    _grouped_bar(ax, methods, avg, metrics, labels,
                 f"Precision  (n={len(rows)} questions)")
    fig.tight_layout()
    path = out_dir / "04_precision.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path.name}")


# ── fig 05 : context quality ──────────────────────────────────────────────────
def fig_context_quality(rows, methods, out_dir):
    metrics = ["context_precision", "context_recall", "redundancy_rate"]
    labels  = ["Context\nPrecision", "Context\nRecall", "Redundancy\nRate"]
    avg = {m: {met: mean(agg(rows, m, met)) for met in metrics} for m in methods}

    fig, ax = plt.subplots(figsize=(9, 5))
    _grouped_bar(ax, methods, avg, metrics, labels,
                 f"Context Quality  (n={len(rows)} questions)", ylim=1.25)
    fig.tight_layout()
    path = out_dir / "05_context_quality.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path.name}")


# ── fig 06 : latency box+strip ────────────────────────────────────────────────
def fig_latency_box(rows, methods, out_dir):
    data   = [agg(rows, m, "latency_s") for m in methods]
    labels = [METHOD_LABELS.get(m, m) for m in methods]

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops={"color": "black", "lw": 2})
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    rng = np.random.default_rng(42)
    for i, (vals, color) in enumerate(zip(data, COLORS), start=1):
        jitter = rng.uniform(-0.15, 0.15, len(vals))
        ax.scatter([i + j for j in jitter], vals,
                   color=color, s=30, zorder=5, alpha=0.8)
    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Latency (s)")
    ax.set_title(f"Retrieval Latency per Method  (n={len(rows)} questions)",
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = out_dir / "06_latency_box.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path.name}")


# ── fig 07 : latency breakdown stacked bar ────────────────────────────────────
def fig_latency_breakdown(rows, methods, out_dir):
    steps = ["bm25_ms", "dense_ms", "fusion_ms", "rerank_ms"]
    step_labels = ["BM25", "Dense+Embed", "Fusion/RRF", "Rerank"]

    avg_breakdown = {}
    for m in methods:
        parts = {s: [] for s in steps}
        for row in rows:
            lb = row["configs"].get(m, {}).get("latency_breakdown_ms", {})
            for s in steps:
                parts[s].append(lb.get(s, 0.0))
        avg_breakdown[m] = {s: mean(v) for s, v in parts.items()}

    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(10, 5))
    bottoms = np.zeros(len(methods))
    step_colors = list(LATENCY_COLORS.values())
    for step, label, color in zip(steps, step_labels, step_colors):
        vals = np.array([avg_breakdown[m][step] for m in methods])
        bars = ax.bar(x, vals, bottom=bottoms, label=label, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            if v > 5:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f"{v:.0f}", ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")
        bottoms += vals

    # total label on top
    for i, m in enumerate(methods):
        total = sum(avg_breakdown[m].values())
        ax.text(i, bottoms[i] + 5, f"{total:.0f} ms",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], fontsize=9)
    ax.set_ylabel("Avg latency (ms)")
    ax.set_title(f"Retrieval Latency Breakdown  (n={len(rows)} questions)",
                 fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = out_dir / "07_latency_breakdown.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path.name}")


# ── fig 08 : token cost ───────────────────────────────────────────────────────
def fig_token_cost(rows, methods, out_dir):
    """LLM prompt/completion tokens and embedding tokens per method (avg per question)."""
    token_keys = ["llm_prompt_tokens", "llm_completion_tokens", "embedding_tokens"]
    tok_labels  = ["LLM Prompt", "LLM Completion", "Embedding"]
    tok_colors  = ["#4C72B0", "#DD8452", "#55A868"]

    avg_tokens = {}
    for m in methods:
        parts = {k: [] for k in token_keys}
        for row in rows:
            tc = row.get("retrieval_token_cost", {}).get(m, {})
            for k in token_keys:
                parts[k].append(tc.get(k, 0))
        avg_tokens[m] = {k: mean(v) for k, v in parts.items()}

    # skip if all zero (reranker-only run without LLM)
    if all(avg_tokens[m]["llm_prompt_tokens"] == 0 and
           avg_tokens[m]["embedding_tokens"] == 0
           for m in methods):
        print("  (08_token_cost: all zeros, skipped)")
        return

    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(10, 5))
    bottoms = np.zeros(len(methods))
    for key, label, color in zip(token_keys, tok_labels, tok_colors):
        vals = np.array([avg_tokens[m][key] for m in methods])
        bars = ax.bar(x, vals, bottom=bottoms, label=label, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            if v > 2:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f"{v:.0f}", ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")
        bottoms += vals

    for i, m in enumerate(methods):
        total = sum(avg_tokens[m].values())
        if total > 0:
            ax.text(i, bottoms[i] + 1, f"{total:.0f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], fontsize=9)
    ax.set_ylabel("Avg tokens / question")
    ax.set_title(f"Retrieval Token Cost per Method  (n={len(rows)} questions)",
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = out_dir / "08_token_cost.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path.name}")


# ── fig 09-11 : per-question heatmaps ────────────────────────────────────────
def fig_heatmap(rows, methods, out_dir, metric: str, fig_num: int):
    n_q, n_m = len(rows), len(methods)
    matrix = np.zeros((n_q, n_m))
    q_labels = []
    for i, row in enumerate(rows):
        q_labels.append(row["item_id"].split("_")[-1])
        for j, m in enumerate(methods):
            matrix[i, j] = row["configs"].get(m, {}).get(metric, 0.0)

    col_labels = [METHOD_LABELS.get(m, m).replace("\n", " ") for m in methods]
    fig_h = max(4, n_q * 0.38)
    fig, ax = plt.subplots(figsize=(max(8, n_m * 1.9), fig_h))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(n_m)); ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(n_q)); ax.set_yticklabels(q_labels, fontsize=8)
    ax.set_title(f"{metric.upper()} per Question × Method  (green=1, red=0)",
                 fontweight="bold")
    for i in range(n_q):
        for j in range(n_m):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if v > 0.35 else "white")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    path = out_dir / f"{fig_num:02d}_heatmap_{metric.replace('@','')}.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path.name}")


# ── fig 12 : per-difficulty full breakdown ────────────────────────────────────
def fig_by_difficulty(rows, methods, out_dir):
    """One subplot per difficulty level showing all key metrics × methods.
    Also saves a cross-difficulty comparison (fig 12b) when ≥2 levels exist.
    """
    difficulties = sorted({r.get("difficulty", "unknown") for r in rows})
    DIFF_COLORS = {"easy": "#55A868", "medium": "#4C72B0", "hard": "#C44E52", "unknown": "#8172B2"}

    metrics      = ["recall@5", "mrr@5", "hit@5", "ndcg@5", "context_precision", "redundancy_rate"]
    metric_short = ["Recall@5", "MRR@5", "Hit@5", "NDCG@5", "CtxPrec", "Redundancy"]

    # ── 12_by_difficulty_per_level.png : one row per level ───────────────────
    n_diff = len(difficulties)
    fig, axes = plt.subplots(n_diff, 1,
                             figsize=(13, 4.5 * n_diff),
                             squeeze=False)

    for row_idx, diff in enumerate(difficulties):
        ax = axes[row_idx][0]
        sub = [r for r in rows if r.get("difficulty") == diff]
        x = np.arange(len(metrics))
        n = len(methods)
        w = 0.72 / n
        for i, method in enumerate(methods):
            vals = [mean(agg(sub, method, m)) for m in metrics]
            bars = ax.bar(x + i * w - (n - 1) * w / 2, vals,
                          width=w * 0.88, color=_color(i),
                          label=METHOD_LABELS.get(method, method))
            for bar, v in zip(bars, vals):
                if v > 0.01:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.012,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=6)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_short, fontsize=9)
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("Score")
        diff_color = DIFF_COLORS.get(diff, "#333333")
        ax.set_title(f"Difficulty: {diff.upper()}  ({len(sub)} questions)",
                     fontweight="bold", color=diff_color, fontsize=11)
        ax.legend(fontsize=7, ncol=3, loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Full Metrics by Difficulty Level  (total n={len(rows)})",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = out_dir / "12_by_difficulty_per_level.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path.name}")

    # ── 12b_difficulty_comparison.png : how methods change across levels ──────
    if n_diff < 2:
        print("  (12b_difficulty_comparison: only 1 level so far, will appear when more data is processed)")
        return

    cmp_metrics      = ["recall@5", "mrr@5", "hit@5", "ndcg@5"]
    cmp_metric_short = ["Recall@5", "MRR@5", "Hit@5", "NDCG@5"]

    fig, axes = plt.subplots(1, len(cmp_metrics), figsize=(15, 5), sharey=True)
    for ax, met, lab in zip(axes, cmp_metrics, cmp_metric_short):
        x = np.arange(len(difficulties))
        n = len(methods)
        w = 0.7 / n
        for i, method in enumerate(methods):
            vals = [mean(agg([r for r in rows if r.get("difficulty") == d], method, met))
                    for d in difficulties]
            bars = ax.bar(x + i * w - (n - 1) * w / 2, vals,
                          width=w * 0.9, color=_color(i),
                          label=METHOD_LABELS.get(method, method))
            for bar, v in zip(bars, vals):
                if v > 0.01:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.012,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=6)
        diff_labels = [f"{d}\n(n={sum(1 for r in rows if r.get('difficulty')==d)})"
                       for d in difficulties]
        ax.set_xticks(x); ax.set_xticklabels(diff_labels, fontsize=9)
        ax.set_ylim(0, 1.18)
        ax.set_title(lab, fontweight="bold")
        if ax == axes[0]:
            ax.set_ylabel("Score")
        ax.legend(fontsize=6.5, ncol=2)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Cross-Difficulty Comparison  (total n={len(rows)})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "12b_difficulty_comparison.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path.name}")

    # ── 12c_difficulty_heatmap.png : avg score per method × difficulty ────────
    # rows = methods, cols = difficulties, cells = avg recall@5
    score_metrics = ["recall@5", "hit@5", "ndcg@5", "mrr@5", "context_precision"]
    score_labels  = ["Recall@5", "Hit@5", "NDCG@5", "MRR@5", "CtxPrec"]
    n_m, n_d = len(methods), len(difficulties)

    fig, axes = plt.subplots(1, len(score_metrics), figsize=(4 * len(score_metrics), n_m * 0.6 + 2))
    for ax, met, lab in zip(axes, score_metrics, score_labels):
        matrix = np.zeros((n_m, n_d))
        for i, method in enumerate(methods):
            for j, diff in enumerate(difficulties):
                sub = [r for r in rows if r.get("difficulty") == diff]
                matrix[i, j] = mean(agg(sub, method, met))
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(n_d))
        ax.set_xticklabels([d.capitalize() for d in difficulties], fontsize=9)
        ax.set_yticks(range(n_m))
        ax.set_yticklabels([METHOD_LABELS.get(m, m).replace("\n", " ") for m in methods], fontsize=8)
        ax.set_title(lab, fontweight="bold", fontsize=9)
        for i in range(n_m):
            for j in range(n_d):
                v = matrix[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if v > 0.35 else "white")
        plt.colorbar(im, ax=ax, fraction=0.05, pad=0.02)

    fig.suptitle(f"Score Heatmap: Method × Difficulty  (total n={len(rows)})",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "12c_difficulty_heatmap.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path.name}")


# ── fig 13 : radar ────────────────────────────────────────────────────────────
def fig_radar(rows, methods, out_dir):
    radar_metrics = ["recall@5", "mrr@5", "hit@5", "ndcg@5", "context_precision"]
    radar_labels  = ["Recall@5", "MRR@5", "Hit@5", "NDCG@5", "Ctx Precision"]
    avg = {m: [mean(agg(rows, m, k)) for k in radar_metrics] for m in methods}

    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    for i, method in enumerate(methods):
        vals = avg[method] + avg[method][:1]
        ax.plot(angles, vals, "o-", linewidth=1.8, color=_color(i),
                label=METHOD_LABELS.get(method, method).replace("\n", " "))
        ax.fill(angles, vals, alpha=0.1, color=_color(i))
    ax.set_thetagrids(np.degrees(angles[:-1]), radar_labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("Method Comparison — Radar Chart", fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "13_radar.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path.name}")


# ── summary table ─────────────────────────────────────────────────────────────
def print_summary_table(rows, methods):
    metrics = ["recall@5", "recall@10", "mrr@5", "hit@5", "ndcg@5",
               "context_precision", "redundancy_rate", "latency_s"]
    col_w = 17
    header = f"{'Method':<{col_w}}" + "".join(f"{m:>13}" for m in metrics)
    sep = "=" * len(header)
    print(f"\n{sep}\n{header}\n{'-' * len(header)}")
    for m in methods:
        label = METHOD_LABELS.get(m, m).replace("\n", " ")
        row = f"{label:<{col_w}}" + "".join(
            f"{mean(agg(rows, m, met)):>13.3f}" for met in metrics
        )
        print(row)
    print(sep)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--file",
        default="../data/ab_golden5_imported8_resumable/partial_results.jsonl",
        help="Path to partial_results.jsonl",
    )
    ap.add_argument(
        "--out-dir", default=None,
        help="Output directory for PNG files (default: same dir as --file)",
    )
    args = ap.parse_args()

    src = Path(args.file).resolve()
    if not src.exists():
        raise SystemExit(f"File not found: {src}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(src)
    if not rows:
        raise SystemExit("No completed rows found in file.")

    methods = _methods_present(rows)
    print(f"\nLoaded {len(rows)} questions | Methods: {methods}")
    print(f"Output dir: {out_dir}\n")

    print_summary_table(rows, methods)

    print("\nGenerating plots...")
    fig_quality(rows, methods, out_dir, k=5,  fig_num=1)   # 01
    fig_quality(rows, methods, out_dir, k=10, fig_num=2)   # 02
    fig_k5_vs_k10(rows, methods, out_dir)                  # 03
    fig_precision(rows, methods, out_dir)                  # 04
    fig_context_quality(rows, methods, out_dir)            # 05
    fig_latency_box(rows, methods, out_dir)                # 06
    fig_latency_breakdown(rows, methods, out_dir)          # 07
    fig_token_cost(rows, methods, out_dir)                 # 08
    fig_heatmap(rows, methods, out_dir, "recall@5",  9)    # 09
    fig_heatmap(rows, methods, out_dir, "hit@5",    10)    # 10
    fig_heatmap(rows, methods, out_dir, "ndcg@5",   11)    # 11
    fig_by_difficulty(rows, methods, out_dir)              # 12
    fig_radar(rows, methods, out_dir)                      # 13

    print(f"\nDone — 13 plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
