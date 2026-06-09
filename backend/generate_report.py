import pandas as pd
import numpy as np

def parse_csv(filepath):
    df = pd.read_csv(filepath)
    # Convert numerical columns
    numeric_cols = [
        'hit@5', 'precision@5', 'recall@5', 'mrr@5', 'map@5', 'ndcg@5',
        'latency_s', 'prompt_tokens', 'completion_tokens', 'total_tokens',
        'ragas_faithfulness', 'ragas_answer_relevancy', 'ragas_answer_correctness',
        'ragas_semantic_similarity', 'ragas_context_precision', 'ragas_context_recall',
        'ragas_context_entity_recall', 'ragas_noise_sensitivity'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df

def generate_report(df, output_path):
    n_questions = len(df)
    total_latency = df['latency_s'].sum()
    avg_latency = df['latency_s'].mean()
    total_tokens = df['total_tokens'].sum()
    avg_tokens = df['total_tokens'].mean()
    
    # Retrieval averages
    hit_5 = df['hit@5'].mean()
    prec_5 = df['precision@5'].mean()
    rec_5 = df['recall@5'].mean()
    mrr_5 = df['mrr@5'].mean()
    map_5 = df['map@5'].mean()
    ndcg_5 = df['ndcg@5'].mean()
    
    # Ragas averages
    r_faith = df['ragas_faithfulness'].mean()
    r_relev = df['ragas_answer_relevancy'].mean()
    r_corr = df['ragas_answer_correctness'].mean()
    r_sim = df['ragas_semantic_similarity'].mean()
    r_cprec = df['ragas_context_precision'].mean()
    r_crec = df['ragas_context_recall'].mean()
    r_cent = df['ragas_context_entity_recall'].mean()
    r_nsens = df['ragas_noise_sensitivity'].mean()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# VinFast RAG Benchmark\n\n")
        f.write(f"- Questions: **{n_questions}**\n")
        f.write(f"- Total latency: **{total_latency:.2f}s**\n")
        f.write(f"- Avg latency: **{avg_latency:.2f}s**\n")
        f.write(f"- Total tokens: **{int(total_tokens)}**\n")
        f.write(f"- Avg tokens: **{avg_tokens:.1f}**\n")
        f.write("- Relevance mode: **llm**\n\n")
        
        f.write("## Retrieval\n\n")
        f.write("| Metric | Score |\n")
        f.write("|---|---:|\n")
        f.write(f"| hit@5 | {hit_5:.3f} |\n")
        f.write(f"| precision@5 | {prec_5:.3f} |\n")
        f.write(f"| recall@5 | {rec_5:.3f} |\n")
        f.write(f"| mrr@5 | {mrr_5:.3f} |\n")
        f.write(f"| map@5 | {map_5:.3f} |\n")
        f.write(f"| ndcg@5 | {ndcg_5:.3f} |\n\n")
        
        f.write("## RAGAS\n\n")
        f.write(f"- ragas_faithfulness: **{r_faith:.3f}**\n")
        f.write(f"- ragas_answer_relevancy: **{r_relev:.3f}**\n")
        f.write(f"- ragas_answer_correctness: **{r_corr:.3f}**\n")
        f.write(f"- ragas_semantic_similarity: **{r_sim:.3f}**\n")
        f.write(f"- ragas_context_precision: **{r_cprec:.3f}**\n")
        f.write(f"- ragas_context_recall: **{r_crec:.3f}**\n")
        f.write(f"- ragas_context_entity_recall: **{r_cent:.3f}**\n")
        f.write(f"- ragas_noise_sensitivity: **{r_nsens:.3f}**\n\n")
        
        f.write("## By Difficulty\n\n")
        f.write("| Difficulty | n | Hit@5 | Precision@5 | Recall@5 | MRR@5 | MAP@5 | NDCG@5 | Avg latency | Avg tokens |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        
        diff_group = df.groupby('do_kho')
        for name, group in diff_group:
            n = len(group)
            g_hit = group['hit@5'].mean()
            g_prec = group['precision@5'].mean()
            g_rec = group['recall@5'].mean()
            g_mrr = group['mrr@5'].mean()
            g_map = group['map@5'].mean()
            g_ndcg = group['ndcg@5'].mean()
            g_lat = group['latency_s'].mean()
            g_tok = group['total_tokens'].mean()
            f.write(f"| {name.lower()} | {n} | {g_hit:.3f} | {g_prec:.3f} | {g_rec:.3f} | {g_mrr:.3f} | {g_map:.3f} | {g_ndcg:.3f} | {g_lat:.2f}s | {g_tok:.1f} |\n")

if __name__ == "__main__":
    df = parse_csv("../artifact/vinfast_eval_metrics.csv")
    generate_report(df, "../data/data_lam/vinfast_benchmark_report.md")
    print("Generated report at ../data/data_lam/vinfast_benchmark_report.md")
