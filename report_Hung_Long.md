# Báo cáo: Phát hiện mâu thuẫn tri thức 2 tầng cho hệ thống RAG (ConflictRAG)

**Nhóm thực hiện:** Hùng, Long
**Module:** `backend/app/conflict/` · `data/conflict_rag/` · `frontend/src/components/ConflictReportCard.tsx`
**Tham chiếu ý tưởng:** ConflictRAG — *Detecting and Resolving Knowledge Conflicts in Retrieval-Augmented Generation* (arXiv 2605.17301)

---

## 1. Bối cảnh bài toán

Hệ thống của nhóm là một **RAG Agent hỏi đáp tài liệu tiếng Việt**: người dùng nạp tài liệu (PDF, URL, văn bản), đặt câu hỏi, hệ thống truy hồi các đoạn liên quan rồi để LLM sinh câu trả lời có trích dẫn.

Một giả định ngầm của mọi pipeline RAG là: **các đoạn tài liệu truy hồi về đều nhất quán với nhau**. Trong thực tế giả định này thường sai:

- Kho tài liệu gồm nhiều nguồn, nhiều phiên bản, nhiều thời điểm.
- Cùng một câu hỏi có thể kéo về các đoạn **mâu thuẫn nhau** về số liệu, mốc thời gian, hoặc quan điểm.

Ví dụ thực tế gặp trong hệ thống: câu hỏi *"Chủ tịch nước Việt Nam hiện tại là ai?"* truy hồi về hai đoạn — một đoạn ghi *Tô Lâm (2026)*, một đoạn ghi *Lương Cường (2025)*. Nếu gộp cả hai vào ngữ cảnh, LLM có thể trả lời sai hoặc trộn lẫn mà **không hề cảnh báo** cho người dùng.

Các phương pháp RAG cải tiến phổ biến (Self-RAG, CRAG) chủ yếu tối ưu **độ liên quan** của truy hồi, chứ **không phát hiện/giải quyết mâu thuẫn** giữa các đoạn. Đây là khoảng trống mà module này nhắm tới.

---

## 2. Vấn đề đặt ra

### 2.1. Yêu cầu chức năng

Xây dựng một module **phát hiện mâu thuẫn giữa các tài liệu truy hồi** (inter-document conflict), chạy **trước bước sinh câu trả lời**, với 3 loại nhãn:

| Nhãn | Ý nghĩa |
|------|---------|
| `no-conflict` | Không mâu thuẫn (bổ sung, trùng lặp, hoặc khác chủ đề) |
| `factual` | Mâu thuẫn về sự kiện (khẳng định trái ngược) |
| `temporal` | Mâu thuẫn theo thời gian (cùng chủ đề, khác mốc thời gian) |

### 2.2. Ràng buộc

- **Chi phí thấp:** với K=5 tài liệu có C(5,2)=**10 cặp** mỗi câu hỏi. Gọi LLM cho mọi cặp là quá tốn kém và chậm.
- **Triển khai nhẹ:** ưu tiên chạy được trên CPU, model nhỏ, độ trễ thấp.
- **An toàn:** lỗi ở module phát hiện không được làm sập toàn bộ pipeline trả lời.

### 2.3. Vấn đề kỹ thuật phát sinh (trọng tâm của báo cáo)

Sau khi triển khai phiên bản đầu, hệ thống bộc lộ một lỗi nghiêm trọng: **mọi cặp tài liệu đều bị gắn cờ "mâu thuẫn" với độ tự tin 100%**, kể cả các cặp **hoàn toàn không liên quan** (ví dụ "Chủ tịch nước…" vs "thuật toán Dijkstra dùng Heap…"). Việc chẩn đoán và khắc phục lỗi này là phần đóng góp chính.

---

## 3. Phương pháp sử dụng để giải quyết

### 3.1. Kiến trúc phát hiện 2 tầng (Two-Stage Detection)

Luồng tổng quát: `Query + top-K chunks` → ghép cặp → **Stage 1 (MLP)** → định tuyến theo độ tự tin → **Stage 2 (LLM)** (chỉ cho cặp lưỡng lự) → `conflict_report`.

#### Stage 1 — Bộ phân loại MLP trên embedding

- **Encoder (đóng băng):** OpenAI `text-embedding-3-small`, 1536 chiều. Mỗi đoạn được embed cùng câu hỏi: `eᵢ = Embed("Câu hỏi: q\nTài liệu: dᵢ")`, sau đó L2-normalize. Embedding được **cache theo SHA-256** để khỏi gọi lại.
- **Feature vector (4 khối tương tác):**
  ```
  f = [ eᵢ ; eⱼ ; |eᵢ − eⱼ| ; eᵢ ⊙ eⱼ ]  ∈ ℝ⁶¹⁴⁴
  ```
  Trong đó `|eᵢ−eⱼ|` bắt **độ khác biệt** và `eᵢ⊙eⱼ` bắt **độ tương đồng** — kết hợp cả hai để phân biệt "khác mà cùng chủ đề" (mâu thuẫn thật) với "khác vì lạc đề" (không mâu thuẫn).
- **Mạng (`Stage1MLP`):** hai head dùng chung biểu diễn:
  ```
  shared:      Linear(6144→1024) → ReLU → Dropout → Linear(1024→256) → ReLU → Dropout
  binary_head: Linear(256→2)   # có/không mâu thuẫn
  type_head:   Linear(256→3)   # no-conflict / factual / temporal
  ```
  **Tổng tham số: ~6.56 triệu** (lớp đầu chiếm ~96%; encoder không tính vì chạy ngoài, đóng băng).

#### Định tuyến theo độ tự tin (cầu nối 2 tầng)

```python
p = softmax(binary_logits)[1]            # xác suất "có mâu thuẫn"
conflict = p >= threshold                # threshold = 0.7  (quyết định Stage 1)
stage1_confidence = max(p, 1 - p)        # độ CHẮC CHẮN ∈ [0.5, 1.0]

if enable_stage2 and stage1_confidence < tau_c:   # tau_c = 0.7
    refined = _stage2_refine(...)        # → gọi LLM
```

Cần phân biệt **hai ngưỡng có ý nghĩa khác nhau**:
- `threshold` (0.7): ngưỡng **quyết định** có mâu thuẫn của MLP.
- `tau_c` (0.7): ngưỡng **định tuyến** — độ tự tin dưới mức này thì đẩy sang LLM. Điều kiện `confidence < 0.7` tương đương `p ∈ (0.3, 0.7)` — đúng "vùng lưỡng lự".

#### Stage 2 — Tinh chỉnh bằng LLM (có chọn lọc)

- Model `gpt-4o-mini`, `temperature=0`, **JSON mode**; prompt đưa câu hỏi + hai đoạn, yêu cầu trả về:
  ```json
  { "has_conflict": true|false, "type": "factual|temporal|no-conflict",
    "summary": "...", "confidence": 0.0..1.0 }
  ```
- Kết quả LLM **ghi đè** quyết định và loại của Stage 1.
- **Fallback an toàn:** lỗi LLM/parse → giữ nguyên quyết định Stage 1, không sập pipeline.
- `detect_topk()` ghép 10 cặp, đếm `num_stage2_calls` và log để theo dõi chi phí.

### 3.2. Chẩn đoán nguyên nhân lỗi "100% cho mọi cặp"

Truy ngược dữ liệu và lịch sử train, xác định **hai nguyên nhân chồng nhau**:

1. **Dữ liệu huấn luyện (gốc rễ).** Model train từ MQuAKE — **mọi cặp đều cùng chủ đề**: negative là câu paraphrase **trùng** ("Theo tài liệu, " + chính câu đó), positive là cùng câu chỉ khác đúng một giá trị. Model **chưa từng thấy cặp khác chủ đề**, nên không học được khái niệm "không liên quan ⇒ không mâu thuẫn". Trên corpus thật (đoạn dài, khác chủ đề) đây là **out-of-distribution** → đoán bừa.

2. **Quy trình huấn luyện (khuếch đại lỗi).** `train_loss → 0.0` ở epoch ~18. Để đưa loss về 0, mạng bị ép đẩy **logit cực lớn** → **softmax bão hòa** ra ~1.0. Mạng ReLU vốn cực kỳ overconfident trên input lạ, nên mọi cặp OOD đều ra **cùng một con số 100%**, mất khả năng phân biệt.

Bằng chứng: `test_metrics.json` báo **99.8% F1** — nhưng đó là chỉ số "ảo" trên một phân phối nhân tạo quá dễ, không phản ánh năng lực trên dữ liệu thật.

> Hệ quả thiết kế: vì các false-positive đều 100% (rất "tự tin") nên chúng **luôn đi fast-path**, Stage 2 không được kích hoạt để cứu.

### 3.3. Pipeline huấn luyện lại ổn định (`data/conflict_rag/scripts2/`)

Nguyên tắc: **giữ nguyên kiến trúc** (để checkpoint drop-in, không sửa `detector.py`), chỉ sửa **dữ liệu + cách train + hiệu chỉnh sau train**.

| # | Thay đổi | Sửa nguyên nhân | File |
|---|----------|-----------------|------|
| 1 | **Hard negative khác chủ đề:** ghép doc của 2 case khác nhau → gán `no-conflict`, sinh tách riêng từng split (chống rò rỉ) | Dạy khái niệm còn thiếu (gốc rễ #1) | `00_build_hard_negatives.py` |
| 2 | **`label_smoothing=0.1`** + `weight_decay=0.05` + `dropout 0.2→0.3` + `grad_clip=1.0` | Đặt sàn cho loss, chặn logit phình → hết softmax bão hòa (#2) | `02_train_stage1_mlp.py` |
| 3 | **Early-stopping theo `val_loss`** (patience=4), chọn model theo val-loss thấp nhất (thay vì F1 luôn ~0.996) | Dừng trước khi học vẹt (loss về 0) | `02_train_stage1_mlp.py` |
| 4 | **Class weights** (cân bằng lớp temporal hiếm) + **Temperature scaling** fit trên val rồi **gấp thẳng vào trọng số head** + log **ECE** | Hiệu chỉnh xác suất; drop-in | `02_train_stage1_mlp.py` |

Lưu ý quan trọng về "thay đổi model": **các lớp giữ nguyên hình dạng** (6144→1024→256, head 2/3); chỉ `Dropout` đổi tỉ lệ và — sau train — **trọng số của hai lớp head bị rescale bằng `1/T`** (temperature folding: `W /= T; b /= T`), làm xác suất mềm lại mà không đổi quyết định/argmax.

### 3.4. Tích hợp hệ thống

- **Backend:** `ConflictDetector` khởi tạo một lần trong `agent/graph.py` (truyền `llm_model`, `tau_c`, `enable_stage2`); node `conflict_detect` gọi `detect_topk()`, phát sự kiện SSE và đính `conflict_report` vào payload. Cấu hình qua `.env`: `ENABLE_CONFLICT_RAG`, `CONFLICT_THRESHOLD`, `CONFLICT_TAU_C`, `CONFLICT_ENABLE_STAGE2`, `CONFLICT_MODEL_DIR`.
- **Frontend:** `ConflictReportCard.tsx` hiển thị từng cặp mâu thuẫn; bổ sung **badge `🤖 Stage 2 · LLM`** và ô ghi chú (tóm tắt + độ tin cậy LLM) cho các cặp do Stage 2 quyết, cùng dòng tổng "N cặp xác minh bằng LLM".

---

## 4. Kết quả thực nghiệm

### 4.1. Dữ liệu

- Nguồn: MQuAKE-CF, MQuAKE-T chuyển về định dạng ConflictRAG tiếng Việt (`query`, `doc_i`, `doc_j`, `binary_label`, `type_label`).
- `data_v2` (train): 14.235 cặp — `no-conflict` 7.212, `factual` 3.003, `temporal` 4.020.
- `data_v3` (sau tăng cường hard negative): thêm **+7.023 train / +1.549 val / +1.504 test** cặp khác-chủ-đề gán `no-conflict`.

### 4.2. So sánh trước / sau

| Tiêu chí | Model cũ | Model v3 (sau khắc phục) |
|----------|----------|--------------------------|
| Độ tự tin trên cặp lạc đề | **100%** (mọi cặp) | Đã hiệu chỉnh, không còn dí 100% |
| `train_loss` | → **0.0** (bão hòa) | Không về 0 (epoch 1: **0.916**) |
| Cặp "Lương Cường vs thuật toán Heap" | conflict 100% (sai) | hướng tới `no-conflict` |
| Hành vi với cặp khó (vd Dijkstra `O(V²)` vs `O((V+E)logV)`) | conflict 100% | MLP ra ~**60%** → confidence < 0.7 → **đẩy sang Stage 2 LLM** |

### 4.3. Chỉ số huấn luyện model v3 (epoch 1, đại diện)

| Chỉ số | Giá trị | Diễn giải |
|--------|---------|-----------|
| `train_loss` | 0.916 | Không còn về 0 → hết học vẹt |
| `val_loss` | 0.335 | Khái quát hoá tốt |
| `val_binary_f1` | 0.984 | Vẫn giữ độ chính xác phát hiện |
| `mean_confidence` | 0.910 | Tự tin **có cơ sở** (không phải 1.0 cho tất cả) |
| `ECE` | 0.080 | Sai số hiệu chỉnh thấp |

> `mean_confidence` ~0.91 là hợp lý: phần lớn tập dữ liệu là cặp dễ và `label_smoothing` chặn trần ~0.95. Điều quan trọng là **ECE thấp** và **`train_loss` không về 0** — khác hẳn bệnh saturation cũ.

### 4.4. Kiểm thử out-of-distribution (`03_sanity_check.py`)

Bộ kiểm thử thủ công trên các cặp thật (vì tập val cùng phân phối train, không đại diện cho corpus):
- "Tô Lâm" vs "Lương Cường" → kỳ vọng **CONFLICT** (giữ recall mâu thuẫn thật).
- "Lương Cường" vs "thuật toán Heap" → kỳ vọng **no-conflict** (loại false-positive lạc đề).
- Dijkstra `O(V²)` vs `O((V+E)logV)` (ca khó, hai điều kiện cài đặt khác nhau) → lý tưởng `no-conflict`; nếu MLP lưỡng lự thì Stage 2 LLM xử lý.

---

## 5. Nhận xét, đánh giá và hướng phát triển

### 5.1. Nhận xét

- **Lỗi không nằm ở code pipeline mà ở khâu tạo dữ liệu và quy trình train.** Bài học cốt lõi: **chất lượng dữ liệu (đặc biệt hard negative) quyết định nhiều hơn kiến trúc**. Cùng một mạng 6.56M tham số, chỉ đổi dữ liệu + cách train là hành vi thay đổi căn bản.
- **Metric đẹp có thể gây ngộ nhận:** 99.8% F1 trên phân phối nhân tạo dễ là "ảo"; phải kiểm thử OOD mới thấy được năng lực thật.
- **`train_loss → 0` là cờ đỏ** của overconfidence; cần `label_smoothing` + early-stopping để phòng.
- **Kiến trúc 2 tầng hợp lý về chi phí:** MLP gánh phần lớn, LLM chỉ dùng cho thiểu số cặp lưỡng lự; `num_stage2_calls` cho phép giám sát chi phí.

### 5.2. Hạn chế

- Dữ liệu train vẫn là **câu MQuAKE ngắn**, còn lệch (domain gap) so với chunk dài thực tế.
- Hiện chỉ 3 nhãn (**thiếu `opinion`** so với paper).
- **Chưa có tầng resolution** (giải quyết mâu thuẫn) — mới dừng ở phát hiện + cảnh báo.
- Prompt Stage 2 đôi khi còn gắn nhầm conflict cho cặp khác đối tượng/điều kiện (vd Dijkstra vs Fast Marching).
- Khác biệt với paper: dùng encoder OpenAI `text-embedding-3-small` (1536-dim) thay cho `all-MiniLM-L6-v2` (384-dim) → chất lượng tốt hơn nhưng **cần API, không còn CPU-only/miễn phí** cho khâu encode. Cần đánh giá định lượng đầy đủ (F1, accuracy, ECE) trên model v3 hoàn chỉnh để chốt kết quả.

### 5.3. Hướng phát triển

1. **Hard negative từ chunk thật:** mở rộng `00_build_hard_negatives.py` đọc thẳng kho chunk của hệ thống (đoạn dài), thu hẹp domain gap.
2. **Siết prompt Stage 2:** dạy LLM phân biệt "khác đối tượng / khác điều kiện ⇒ không mâu thuẫn".
3. **Thêm loại `opinion`** (3 → 4 nhãn) như paper, cần dữ liệu và retrain head.
4. **Tầng resolution:** Entropy-TOPSIS chọn nguồn đáng tin cho mâu thuẫn factual; ưu tiên nguồn mới nhất cho temporal; tổng hợp đa quan điểm cho opinion.
5. **Đánh giá định lượng đầy đủ** model v3 (đặc biệt trên tập OOD/corpus thật) và chỉ số CARS như paper đề xuất.

---

*Mã nguồn chính:* `backend/app/conflict/detector.py` · `backend/app/agent/graph.py` · `backend/app/config.py` · `data/conflict_rag/scripts2/{00,01,02,03}` · `frontend/src/components/ConflictReportCard.tsx`
