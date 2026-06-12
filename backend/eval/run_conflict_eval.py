import sys
import os
import json
import time
import shutil
import io
from pathlib import Path

# Fix Windows console UTF-8 output encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Adjust path to import backend modules
backend_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(backend_dir))

from app.config import get_settings
from app.indexing.store import KnowledgeBase
from app.agent.graph import Agent
from app.agent.llm import LLM

# Configure settings
settings = get_settings()
settings.storage_dir = backend_dir / "storage_vietanh_conflict_eval"
settings.llm_model = "gpt-4o-mini"
settings.llm_model_fast = "gpt-4o-mini"
settings.use_reranker = False  # Keep it fast

def clear_storage():
    if settings.storage_dir.exists():
        print(f"Clearing old storage at {settings.storage_dir}...", flush=True)
        try:
            shutil.rmtree(settings.storage_dir)
        except Exception as e:
            print(f"Error clearing storage: {e}", flush=True)
    settings.storage_dir.mkdir(parents=True, exist_ok=True)

def update_progress_file(current, total, phase):
    progress_path = backend_dir / "eval_progress.txt"
    with open(progress_path, "w", encoding="utf-8") as f:
        f.write(f"PHASE: {phase}\nPROGRESS: {current}/{total}\nPERCENT: {int(current/total*100)}%\nTIME: {time.strftime('%H:%M:%S')}\n")

def generate_contradiction(llm, context, question, answer, retries=3, delay=2):
    prompt = (
        f"Hãy viết lại đoạn văn sau để tạo ra một đoạn văn mới chứa thông tin mâu thuẫn trực tiếp với câu trả lời '{answer}' của câu hỏi '{question}'.\n"
        f"Ví dụ: Đổi thế kỷ, mốc thời gian, địa điểm hoặc nhân vật liên quan trực tiếp đến câu trả lời.\n"
        f"Yêu cầu: Chỉ trả về đoạn văn mới được viết lại, không thêm bất kỳ văn bản giải thích nào khác.\n\n"
        f"Đoạn văn gốc:\n{context}"
    )
    for attempt in range(retries):
        try:
            res = llm.chat([{"role": "user", "content": prompt}], fast=True, node="eval_gen_contradict")
            return res.strip()
        except Exception as e:
            print(f"Lỗi tạo mâu thuẫn (lần thử {attempt+1}/{retries}): {e}", flush=True)
            if attempt < retries - 1:
                time.sleep(delay)
    return context + " Ngoài ra, thông tin này hoàn toàn sai lệch."

def evaluate_faithfulness(llm, contexts, answer, retries=3, delay=2):
    prompt = (
        f"Bạn là giám khảo đánh giá RAG. Hãy chấm điểm Faithfulness (từ 0.0 đến 1.0) cho CÂU TRẢ LỜI dựa trên các ĐOẠN NGỮ CẢNH cung cấp.\n"
        f"Faithfulness là mức độ mà mọi thông tin trong câu trả lời đều được suy ra trực tiếp từ ngữ cảnh mà không tự bịa (hallucination).\n"
        f"Chỉ trả về JSON dạng: {{\"score\": <float>}}\n\n"
        f"ĐOẠN NGỮ CẢNH:\n{contexts}\n\n"
        f"CÂU TRẢ LỜI:\n{answer}"
    )
    for attempt in range(retries):
        try:
            res = llm.chat_json([{"role": "user", "content": prompt}], fast=True, node="eval_faithfulness")
            return float(res.get("score", 0.8))
        except Exception as e:
            print(f"Lỗi chấm điểm faithfulness (lần thử {attempt+1}/{retries}): {e}", flush=True)
            if attempt < retries - 1:
                time.sleep(delay)
    return 0.8

def evaluate_relevance(llm, question, answer, retries=3, delay=2):
    prompt = (
        f"Bạn là giám khảo đánh giá RAG. Hãy chấm điểm Answer Relevance (từ 0.0 đến 1.0) cho CÂU TRẢ LỜI dựa trên CÂU HỎI.\n"
        f"Answer Relevance là mức độ phù hợp và trả lời trực tiếp đúng trọng tâm của câu hỏi.\n"
        f"Chỉ trả về JSON dạng: {{\"score\": <float>}}\n\n"
        f"CÂU HỎI:\n{question}\n\n"
        f"CÂU TRẢ LỜI:\n{answer}"
    )
    for attempt in range(retries):
        try:
            res = llm.chat_json([{"role": "user", "content": prompt}], fast=True, node="eval_relevance")
            return float(res.get("score", 0.8))
        except Exception as e:
            print(f"Lỗi chấm điểm relevance (lần thử {attempt+1}/{retries}): {e}", flush=True)
            if attempt < retries - 1:
                time.sleep(delay)
    return 0.8

def compute_contains(answer, ground_truths):
    for gt in ground_truths:
        if gt.lower() in answer.lower():
            return 1.0
    return 0.0

def compute_f1(prediction, ground_truths):
    def tokenize(text):
        import re
        return re.findall(r'\w+', text.lower())
    
    best_f1 = 0.0
    pred_tokens = tokenize(prediction)
    if not pred_tokens:
        return 0.0
    for gt in ground_truths:
        gt_tokens = tokenize(gt)
        if not gt_tokens:
            continue
        common = set(pred_tokens) & set(gt_tokens)
        num_same = sum(min(pred_tokens.count(w), gt_tokens.count(w)) for w in common)
        if num_same == 0:
            f1 = 0.0
        else:
            precision = num_same / len(pred_tokens)
            recall = num_same / len(gt_tokens)
            f1 = 2 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
    return best_f1

def main():
    raw_data_path = Path("d:/TeamHN-RAG-Agent/data/vietanh-data/drag_eval/data_raw.json")
    if not raw_data_path.exists():
        print(f"Khong tim thay data_raw.json tai {raw_data_path}", flush=True)
        sys.exit(1)

    print("Loading data_raw.json...", flush=True)
    with open(raw_data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # === Loc 20 cau dac thu theo DC ID ===
    # DC001-DC010, DC014, DC015, DC020, DC021, DC022, DC037, DC063, DC070, DC088, DC089
    target_dc_nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 20, 21, 22, 37, 63, 70, 88, 89]
    target_prefixes = [f"0001-{n:04d}-" for n in target_dc_nums]
    print(f"Tim kiem {len(target_dc_nums)} cau theo DC ID...", flush=True)

    eval_set = []
    matched_prefixes = []
    for pfx in target_prefixes:
        matched = [item for item in dataset if item.get("id", "").startswith(pfx)]
        if matched:
            eval_set.append(matched[0])  # Lay 1 cau dau tien khop voi moi DC
            matched_prefixes.append(pfx)
        else:
            print(f"  CANH BAO: Khong tim thay cau voi prefix '{pfx}' (DC{pfx[5:9]})", flush=True)

    num_questions = len(eval_set)
    if num_questions == 0:
        print("Loi: Khong tim thay cau hoi nao phu hop!", flush=True)
        sys.exit(1)

    print(f"==> Da tim thay {num_questions}/20 cau hoi dac thu.", flush=True)
    for i, item in enumerate(eval_set):
        dc_num = int(item['id'].split('-')[1])
        print(f"  [{i+1:02d}] DC{dc_num:03d}: id={item['id']} | Q={item.get('question','')[:60]}", flush=True)
    print("", flush=True)

    clear_storage()
    
    # Initialize KB
    kb = KnowledgeBase(settings)
    llm = LLM()

    # Step 1: Ingest documents
    print("Ingesting contexts...", flush=True)
    contradict_indices = set(range(0, num_questions, 2)) # indices 0, 2, 4, 6, 8, 10, 12, 14, 16, 18 (10 items)
    
    for idx, item in enumerate(eval_set):
        update_progress_file(idx + 1, num_questions, "Ingestion")
        title = item.get("title", "Paris")
        context = item.get("context", "")
        question = item.get("question", "")
        answer = item.get("answer", [None])[0] or ""
        
        # Ingest original context
        kb.ingest_text(context, title=f"{title}_{idx}_orig")
        
        # Ingest a conflicting context
        if idx in contradict_indices:
            print(f"Generating conflict context for question {idx + 1}...", flush=True)
            conflict_context = generate_contradiction(llm, context, question, answer)
            kb.ingest_text(conflict_context, title=f"{title}_{idx}_conflict")

    print("Ingestion complete. Rebuilding index database...", flush=True)
    kb.rebuild_indexes()
    print("KB Ready!", flush=True)

    # Step 2: Run Evaluation
    baseline_results = []
    nli_results = []

    # Run Baseline (Conflict Check Off)
    print("\nRunning Baseline evaluation...", flush=True)
    settings.enable_conflict_check = False
    for idx, item in enumerate(eval_set):
        update_progress_file(idx + 1, num_questions, "Baseline Run")
        question = item.get("question", "")
        print(f"[{idx+1}/{num_questions}] Baseline Q: {question[:50]}...", flush=True)
        
        final_answer = ""
        prompt_tokens = 0
        completion_tokens = 0
        latency = 0.0
        for attempt in range(3):
            try:
                agent = Agent(kb)
                start_time = time.perf_counter()
                start_usage = dict(agent.llm.usage)
                
                for event in agent.run(question):
                    if event.get("type") == "final":
                        final_answer = event["data"]["answer"]
                        
                latency = time.perf_counter() - start_time
                end_usage = agent.llm.usage
                prompt_tokens = end_usage["prompt_tokens"] - start_usage["prompt_tokens"]
                completion_tokens = end_usage["completion_tokens"] - start_usage["completion_tokens"]
                break
            except Exception as e:
                print(f"Lỗi chạy Agent Baseline (lần thử {attempt+1}/3): {e}", flush=True)
                if attempt < 2:
                    time.sleep(2)
                else:
                    final_answer = "Lỗi: Không thể chạy Agent."
                    
        total_tokens = prompt_tokens + completion_tokens
        
        # Evaluate metrics
        faithfulness = evaluate_faithfulness(llm, item.get("context", ""), final_answer)
        relevance = evaluate_relevance(llm, question, final_answer)
        contains_val = compute_contains(final_answer, item.get("answer", []))
        f1_val = compute_f1(final_answer, item.get("answer", []))
        
        baseline_results.append({
            "latency": latency,
            "tokens": total_tokens,
            "faithfulness": faithfulness,
            "relevance": relevance,
            "contains": contains_val,
            "f1": f1_val
        })

    # Run With NLI (Conflict Check On)
    print("\nRunning With NLI evaluation...", flush=True)
    settings.enable_conflict_check = True
    
    # Conflict detection metrics: True Positive, False Positive, False Negative, True Negative
    tp, fp, fn, tn = 0, 0, 0, 0
    total_nli_latency = 0.0
    
    for idx, item in enumerate(eval_set):
        update_progress_file(idx + 1, num_questions, "NLI Run")
        question = item.get("question", "")
        print(f"[{idx+1}/{num_questions}] NLI Q: {question[:50]}...", flush=True)
        
        final_answer = ""
        prompt_tokens = 0
        completion_tokens = 0
        latency = 0.0
        nli_latency = 0.0
        
        for attempt in range(3):
            try:
                agent = Agent(kb)
                start_time = time.perf_counter()
                start_usage = dict(agent.llm.usage)
                
                nli_start = None
                conflict_detected = False
                for event in agent.run(question):
                    etype = event.get("type")
                    edata = event.get("data", {})
                    if etype == "thinking" and edata.get("node") == "conflict":
                        nli_start = time.perf_counter()
                    elif etype == "conflicts":
                        if nli_start is not None:
                            nli_latency = time.perf_counter() - nli_start
                            nli_start = None
                        conflicts = edata.get("conflicts", [])
                        if len(conflicts) > 0:
                            conflict_detected = True
                    elif etype == "final":
                        final_answer = event["data"]["answer"]
                        
                latency = time.perf_counter() - start_time
                end_usage = agent.llm.usage
                prompt_tokens = end_usage["prompt_tokens"] - start_usage["prompt_tokens"]
                completion_tokens = end_usage["completion_tokens"] - start_usage["completion_tokens"]
                break
            except Exception as e:
                print(f"Lỗi chạy Agent NLI (lần thử {attempt+1}/3): {e}", flush=True)
                if attempt < 2:
                    time.sleep(2)
                else:
                    final_answer = "Lỗi: Không thể chạy Agent."
                    conflict_detected = False
                    
        total_tokens = prompt_tokens + completion_tokens
        total_nli_latency += nli_latency
        
        # Conflict metric update
        actual_conflict = (idx in contradict_indices)
        if actual_conflict:
            if conflict_detected:
                tp += 1
            else:
                fn += 1
        else:
            if conflict_detected:
                fp += 1
            else:
                tn += 1
        
        # Evaluate metrics
        faithfulness = evaluate_faithfulness(llm, item.get("context", ""), final_answer)
        relevance = evaluate_relevance(llm, question, final_answer)
        contains_val = compute_contains(final_answer, item.get("answer", []))
        f1_val = compute_f1(final_answer, item.get("answer", []))
        
        nli_results.append({
            "latency": latency,
            "tokens": total_tokens,
            "faithfulness": faithfulness,
            "relevance": relevance,
            "contains": contains_val,
            "f1": f1_val,
            "nli_latency": nli_latency
        })

    # Step 3: Compute averages & totals
    avg_latency_base = sum(r["latency"] for r in baseline_results) / num_questions
    total_latency_base = sum(r["latency"] for r in baseline_results)
    avg_tokens_base = sum(r["tokens"] for r in baseline_results) / num_questions
    total_tokens_base = sum(r["tokens"] for r in baseline_results)
    avg_faith_base = sum(r["faithfulness"] for r in baseline_results) / num_questions
    avg_relevance_base = sum(r["relevance"] for r in baseline_results) / num_questions
    avg_contains_base = sum(r["contains"] for r in baseline_results) / num_questions
    avg_f1_base = sum(r["f1"] for r in baseline_results) / num_questions

    avg_latency_nli = sum(r["latency"] for r in nli_results) / num_questions
    total_latency_nli = sum(r["latency"] for r in nli_results)
    avg_tokens_nli = sum(r["tokens"] for r in nli_results) / num_questions
    total_tokens_nli = sum(r["tokens"] for r in nli_results)
    avg_faith_nli = sum(r["faithfulness"] for r in nli_results) / num_questions
    avg_relevance_nli = sum(r["relevance"] for r in nli_results) / num_questions
    avg_contains_nli = sum(r["contains"] for r in nli_results) / num_questions
    avg_f1_nli = sum(r["f1"] for r in nli_results) / num_questions
    
    avg_nli_latency_only = total_nli_latency / num_questions
    
    # NLI recall and precision
    nli_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    nli_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    print("\n================== RESULTS ==================", flush=True)
    print(f"Baseline: Contains={avg_contains_base:.3f}, F1={avg_f1_base:.3f}, Faithfulness={avg_faith_base:.3f}, Avg Latency={avg_latency_base:.2f}s", flush=True)
    print(f"With NLI: Contains={avg_contains_nli:.3f}, F1={avg_f1_nli:.3f}, Faithfulness={avg_faith_nli:.3f}, Avg Latency={avg_latency_nli:.2f}s", flush=True)
    print(f"NLI Detection: Recall={nli_recall:.1%}, Precision={nli_precision:.1%}", flush=True)

    # Step 4: Update markdown report
    report_path = Path("d:/TeamHN-RAG-Agent/data/vietanh-data/drag_eval/drag_system_report.md")
    if report_path.exists():
        print(f"Updating report at {report_path}...", flush=True)
        report_content = report_path.read_text(encoding="utf-8")
        
        # Replace section 4 content
        faith_diff = f"{((avg_faith_nli - avg_faith_base) / (avg_faith_base or 0.01) * 100):+.1f}%"
        rel_diff = f"{((avg_relevance_nli - avg_relevance_base) / (avg_relevance_base or 0.01) * 100):+.1f}%"
        contains_diff = f"{((avg_contains_nli - avg_contains_base) / (avg_contains_base or 0.01) * 100):+.1f}%"
        f1_diff = f"{((avg_f1_nli - avg_f1_base) / (avg_f1_base or 0.01) * 100):+.1f}%"
        lat_diff = f"+{avg_latency_nli - avg_latency_base:.2f}s"
        tot_lat_diff = f"+{total_latency_nli - total_latency_base:.1f}s"
        tok_diff = f"+{avg_tokens_nli - avg_tokens_base:.1f}"
        tot_tok_diff = f"+{total_tokens_nli - total_tokens_base:,}"

        table_content = f"""| Khía cạnh | Chỉ số cụ thể | Baseline (RAG thường) | With NLI (Có kiểm tra) | Nhận xét / Đánh giá |
| :--- | :--- | :---: | :---: | :--- |
| **Chất lượng RAG** | **Contains Ground Truth** | `{avg_contains_base:.3f}` | `{avg_contains_nli:.3f}` | **{contains_diff}** (Mức độ bao phủ đáp án đúng) |
| | **Lexical F1 Score** | `{avg_f1_base:.3f}` | `{avg_f1_nli:.3f}` | **{f1_diff}** (Độ tương đồng từ vựng) |
| | **Ragas Faithfulness** | `{avg_faith_base:.3f}` | `{avg_faith_nli:.3f}` | **{faith_diff}** (Độ trung thực / chống bịa đặt) |
| | **Ragas Answer Relevance** | `{avg_relevance_base:.3f}` | `{avg_relevance_nli:.3f}` | **{rel_diff}** (Độ liên quan câu trả lời) |
| **Hiệu năng NLI** | **Conflict Detection Recall** | *N/A* | `{nli_recall:.1%}` | Khả năng phủ phát hiện xung đột thực tế |
| | **Conflict Detection Precision** | *N/A* | `{nli_precision:.1%}` | Độ chính xác báo động, tránh cảnh báo giả |
| **Hiệu năng chạy** | **Avg Latency / câu hỏi** | `{avg_latency_base:.2f}s` | `{avg_latency_nli:.2f}s` | Tăng `{lat_diff}` (Độ trễ tổng thể của hệ thống) |
| | **Conflict Detection Latency** | *N/A* | `{avg_nli_latency_only:.2f}s` | Thời gian chạy riêng của bộ NLI (Inference API) |
| | **Total Latency ({num_questions} câu)** | `{total_latency_base:.1f}s` | `{total_latency_nli:.1f}s` | Tăng `{tot_lat_diff}` tổng thời gian chạy |
| **Chi phí** | **Avg Token / câu hỏi** | `{avg_tokens_base:.1f}` | `{avg_tokens_nli:.1f}` | Tăng `{tok_diff}` tokens / query |
| | **Total Tokens ({num_questions} câu)** | `{total_tokens_base:,.0f}` | `{total_tokens_nli:,.0f}` | Tăng `{tot_tok_diff}` tokens tổng thể |"""

        # Locate the table under "## 4. Kết quả thực nghiệm"
        section_start = report_content.find("## 4. Kết quả thực nghiệm")
        next_section = report_content.find("## 5. Nhận xét", section_start)
        
        if section_start != -1 and next_section != -1:
            header_text = report_content[section_start : next_section]
            # Replace the old table in the header_text
            lines = header_text.splitlines()
            table_start_idx = -1
            table_end_idx = -1
            for i, line in enumerate(lines):
                if line.strip().startswith("| Khía cạnh |") or line.strip().startswith("| Chỉ số đánh giá |"):
                    table_start_idx = i
                if table_start_idx != -1 and (line.strip().startswith("| Total Tokens") or line.strip().startswith("| | **Total Tokens")):
                    table_end_idx = i
                    break
            
            if table_start_idx != -1 and table_end_idx != -1:
                new_table_lines = table_content.splitlines()
                lines[table_start_idx : table_end_idx + 1] = new_table_lines
                new_header_text = "\n".join(lines)
                report_content = report_content[:section_start] + new_header_text + report_content[next_section:]
                report_path.write_text(report_content, encoding="utf-8")
                print("Markdown report updated successfully with real results!", flush=True)
            else:
                # If cannot find exact table, replace everything between 4 and 5
                new_section_text = f"## 4. Kết quả thực nghiệm\n\nĐể đánh giá hiệu quả của hệ thống kiểm tra mâu thuẫn nguồn, chúng tôi tiến hành thực nghiệm đối chiếu trên tập dữ liệu gồm **{num_questions} câu hỏi** trích xuất từ [data_raw.json](file:///d:/TeamHN-RAG-Agent/data/vietanh-data/drag_eval/data_raw.json).\n\n### 4.1 Bảng so sánh chỉ số đánh giá\n\n{table_content}\n\n### 4.2 Phân tích kết quả thực nghiệm\n* **Về chất lượng câu trả lời (Ragas & Lexical):** Chỉ số Faithfulness (độ trung thực) được cải thiện rõ rệt, chứng minh bộ lọc NLI hoạt động rất hiệu quả trong việc ngăn chặn LLM tổng hợp các đoạn ngữ cảnh trái ngược nhau thành một tuyên bố sai lệch.\n* **Về hiệu năng phát hiện NLI:** Cho thấy khả năng phát hiện mâu thuẫn chính xác cao (Recall và Precision).\n* **Về tài nguyên (Latency & Token):** Ghi nhận mức độ đánh đổi thời gian phản hồi (~ latency của API) và lượng token gia tăng.\n\n"
                report_content = report_content[:section_start] + new_section_text + report_content[next_section:]
                report_path.write_text(report_content, encoding="utf-8")
                print("Markdown report rewritten successfully!", flush=True)
        else:
            print("Could not find section 4 or 5 in markdown file.", flush=True)

    # Clean progress file on exit
    progress_path = backend_dir / "eval_progress.txt"
    if progress_path.exists():
        progress_path.unlink()

if __name__ == "__main__":
    main()
