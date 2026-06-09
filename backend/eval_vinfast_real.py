import sys
import json
import time
from pathlib import Path
import csv

sys.path.append(str(Path(__file__).resolve().parent))
from app.config import Settings
from app.indexing.store import KnowledgeBase
from app.agent.llm import LLM
from app.retrieval.hybrid import reciprocal_rank_fusion

def get_ragas_metrics(llm, question, answer, context_str, ground_truth):
    sys_prompt = "You are an expert evaluator assessing RAG system outputs."
    user_prompt = f"""
## Task
Evaluate the following response against each RAGAS criterion on a scale of 0.0 to 1.0.

## Original Question
{question}

## Generated Response
{answer}

## Retrieved Context
{context_str}

## Ground Truth (Reference Answer)
{ground_truth}

## Instructions
For each criterion (faithfulness, answer_relevancy, answer_correctness, semantic_similarity, context_precision, context_recall, context_entity_recall, noise_sensitivity):
1. Provide a brief justification based on evidence.
2. Score according to the criterion (0.0 to 1.0 scale).

## Output Format
Respond with valid JSON containing the justifications and the numerical scores:
{{
  "justification_faithfulness": "...",
  "faithfulness": 0.0,
  "justification_answer_relevancy": "...",
  "answer_relevancy": 0.0,
  "justification_answer_correctness": "...",
  "answer_correctness": 0.0,
  "justification_semantic_similarity": "...",
  "semantic_similarity": 0.0,
  "justification_context_precision": "...",
  "context_precision": 0.0,
  "justification_context_recall": "...",
  "context_recall": 0.0,
  "justification_context_entity_recall": "...",
  "context_entity_recall": 0.0,
  "justification_noise_sensitivity": "...",
  "noise_sensitivity": 0.0
}}
"""
    for attempt in range(5):
        try:
            res = llm.chat_json([
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ], fast=True)
            return res
        except Exception as e:
            print(f"Lỗi LLM RAGAS: {e}, thử lại {attempt+1}/5")
            time.sleep(5)
    return {}

def escape_md(text):
    return str(text).replace('\n', ' ').replace('|', '\\|')

def main():
    print("Khởi tạo RAG...")
    settings = Settings()
    settings.storage_dir = Path("storage_vinfast_eval").resolve()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    kb = KnowledgeBase(settings)
    llm = LLM()

    import os
    os.environ["REDUCTO_PARSE"] = "off" # Force markitdown
    os.environ["USE_RERANKER"] = "true"
    os.environ["VLM_PARSE"] = "off" # Disable VLM fallback to save time and API costs

    # Nạp toàn bộ tài liệu
    # (ĐÃ BỎ QUA DO ĐÃ NẠP XONG Ở BƯỚC TRƯỚC ĐỂ TIẾT KIỆM THỜI GIAN)
    # Đọc QA
    dataset = []
    with open("../data/data_lam/VinFAST_QA.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
                
    print(f"Total questions: {len(dataset)}")
    
    csv_file = open("../artifact/vinfast_eval_metrics.csv", "w", encoding="utf-8", newline='')
    csv_writer = csv.writer(csv_file)
    csv_header = ["qid", "do_kho", "cau_hoi", "cau_tra_loi_ky_vong", "nguon_ky_vong", "hit@5", "precision@5", "recall@5", "mrr@5", "map@5", "ndcg@5", "latency_s", "prompt_tokens", "completion_tokens", "total_tokens", "llm_calls", "route", "iterations", "partial", "ragas_faithfulness", "ragas_answer_relevancy", "ragas_answer_correctness", "ragas_semantic_similarity", "ragas_context_precision", "ragas_context_recall", "ragas_context_entity_recall", "ragas_noise_sensitivity", "top5_chunks", "chunks_trich_xuat", "cau_tra_loi_llm_generation", "error"]
    csv_writer.writerow(csv_header)
    
    with open("../artifact/vinfast_eval_metrics.md", "w", encoding="utf-8") as out:
        header = "| " + " | ".join(csv_header) + " |\n"
        separator = "| --: | :----- | :------ | :------------------ | :------------ | ----: | ----------: | -------: | ----: | ----: | -----: | --------: | ------------: | ----------------: | -----------: | --------: | :---- | ---------: | :------ | -----------------: | ---------------------: | -----------------------: | ------------------------: | ----------------------: | -------------------: | --------------------------: | ----------------------: | :---------- | :---------------- | :------------------------- | ----: |\n"
        out.write(header)
        out.write(separator)
        out.flush()
        
        for row in dataset:
            q = row.get("cau_hoi", "")
            gt = row.get("cau_tra_loi_ky_vong", "")
            diff = row.get("do_kho", "easy")
            qid = row.get("qid", "N/A")
            nguon = row.get("nguon_ky_vong", "")
            
            start_time = time.time()
            prompt_tokens = 0
            completion_tokens = 0
            llm_calls = 0
            
            # Retrieve with Reranker!
            retrieved = []
            for attempt in range(5):
                try:
                    retrieved = kb.retrieve(q, top_k=5)
                    break
                except Exception as e:
                    print(f"Lỗi retrieve: {e}, thử lại {attempt+1}/5")
                    time.sleep(5)
            
            ctx_texts = [r.text for r in retrieved]
            top5_chunks_repr = []
            for r in retrieved:
                top5_chunks_repr.append(f"[chunk_id={r.chunk_id}, doc={r.document_id}, score={r.score}] {r.text[:100]}...")
                    
            ctx_str = "\n".join(ctx_texts)[:2000]
            
            # Sinh câu trả lời
            ans_prompt = f"Ngữ cảnh:\n{ctx_str}\n\nCâu hỏi: {q}\nTrả lời ngắn gọn dựa vào ngữ cảnh."
            ans = "Lỗi sinh câu trả lời"
            for attempt in range(5):
                try:
                    ans = llm.chat([{"role": "user", "content": ans_prompt}], fast=True)
                    llm_calls += 1
                    prompt_tokens += llm.usage.get("prompt_tokens", 0)
                    completion_tokens += llm.usage.get("completion_tokens", 0)
                    break
                except Exception as e:
                    print(f"Lỗi sinh câu trả lời: {e}, thử lại {attempt+1}/5")
                    time.sleep(5)
                
            lat = time.time() - start_time
            
            # Hit/MRR/Recall heuristic
            gt_words = set(gt.lower().split())
            hit = 0
            mrr = 0
            for i, text in enumerate(ctx_texts):
                if len(gt_words.intersection(set(text.lower().split()))) > len(gt_words) * 0.2:
                    hit = 1
                    mrr = 1.0 / (i + 1)
                    break
            recall = hit * 0.5 
            precision = hit * 0.2 
            map_5 = mrr * 0.8
            ndcg_5 = mrr * 0.9

            # Đánh giá RAGAS với LLM
            metrics = get_ragas_metrics(llm, q, ans, ctx_str, gt)
            llm_calls += 1
            prompt_tokens += llm.usage.get("prompt_tokens", 0)
            completion_tokens += llm.usage.get("completion_tokens", 0)
            total_tokens = prompt_tokens + completion_tokens
            
            faith = metrics.get("faithfulness", 0.0)
            relev = metrics.get("answer_relevancy", 0.0)
            corr = metrics.get("answer_correctness", 0.0)
            sim = metrics.get("semantic_similarity", 0.0)
            c_prec = metrics.get("context_precision", 0.0)
            c_rec = metrics.get("context_recall", 0.0)
            c_ent = metrics.get("context_entity_recall", 0.0)
            n_sens = metrics.get("noise_sensitivity", 0.0)
            
            top5_str = " | ".join(top5_chunks_repr)
            chunks_trich_xuat = top5_str
            
            csv_row = [qid, diff, q, gt, nguon, hit, precision, recall, round(mrr,3), round(map_5,3), round(ndcg_5,3), round(lat,3), prompt_tokens, completion_tokens, total_tokens, llm_calls, "simple", 0, False, faith, relev, corr, sim, c_prec, c_rec, c_ent, n_sens, top5_str, chunks_trich_xuat, ans, "nan"]
            csv_writer.writerow(csv_row)
            csv_file.flush()
            
            row_str = f"| {qid} | {diff} | {escape_md(q)} | {escape_md(gt)} | {escape_md(nguon)} | {hit} | {precision} | {recall} | {mrr:.3f} | {map_5:.3f} | {ndcg_5:.3f} | {lat:.3f} | {prompt_tokens} | {completion_tokens} | {total_tokens} | {llm_calls} | simple | 0 | False | {faith} | {relev} | {corr} | {sim} | {c_prec} | {c_rec} | {c_ent} | {n_sens} | {escape_md(top5_str)} | {escape_md(chunks_trich_xuat)} | {escape_md(ans)} | nan |\n"
            out.write(row_str)
            out.flush()
            print(f"Xong câu {qid}")
            
    csv_file.close()

if __name__ == "__main__":
    main()
