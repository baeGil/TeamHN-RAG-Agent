# Báo cáo Đánh giá và Tối ưu hóa Retrieval: HyDE vs HyPe RAG

Tài liệu này trình bày phân tích so sánh, kết quả đo đạc thực nghiệm và hướng dẫn cấu hình cho hai phương pháp tối ưu hóa truy hồi nâng cao đã được tích hợp vào sản phẩm: **HyDE (Hypothetical Document Embeddings)** và **HyPe (Hypothetical Prompt Embeddings)**.

---

## 1. Tổng quan Kiến trúc

Hệ thống RAG ban đầu gặp khoảng cách ngữ nghĩa (semantic mismatch) giữa câu hỏi của người dùng (ngắn gọn, mang tính chất hỏi) và tài liệu gốc (dài, mang tính chất mô tả/học thuật). Để giải quyết vấn đề này, hai kỹ thuật sau được triển khai:

```
                  ┌──────────────────────┐
                  │ CÂU HỎI NGƯỜI DÙNG   │
                  └──────────┬───────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
       [Nhánh HyDE - Online]         [Nhánh HyPe - Offline]
  Sinh Câu trả lời giả định (LLM)     Sinh Câu hỏi giả định (LLM)
              │                             │
              ▼                             ▼
  Embed(Câu trả lời giả định)       Embed(Câu hỏi giả định)
              │                             │
              ▼                             ▼
   Khớp với Vector văn bản gốc    Khớp với Vector câu hỏi đã lưu
```

### 1.1. HyDE (Hypothetical Document Embeddings)
- **Hoạt động ở runtime (Online)**: Khi người dùng đặt câu hỏi, hệ thống sử dụng LLM để sinh một câu trả lời giả định. Vector biểu diễn của câu trả lời giả định này sau đó được so khớp với chỉ mục vector chứa văn bản gốc.
- **Ưu điểm**: Khớp nối ngữ nghĩa dạng **Văn bản - Văn bản** giúp cải thiện độ chính xác truy hồi.
- **Hạn chế**: Tăng latency truy hồi vì phải thực hiện 1 cuộc gọi LLM trước khi tìm kiếm vector.

### 1.2. HyPe (Hypothetical Prompt Embeddings / doc2query)
- **Hoạt động ở indexing-time (Offline)**: Trong quá trình nạp tài liệu, hệ thống dùng LLM tạo ra **3 câu hỏi giả định** cho mỗi chunk. Các câu hỏi này được nhúng (embed) và đưa vào chỉ mục vector. Khi tìm kiếm, vector câu hỏi của người dùng được so khớp trực tiếp với chỉ mục chứa các câu hỏi giả định.
- **Giải quyết giới hạn chỉ mục**: Thư viện `turbovec` (TurboQuant) yêu cầu ID trong chỉ mục là duy nhất, không cho phép gán nhiều vector vào cùng một chunk ID gốc. Hệ thống đã giải quyết bằng **ID Mapping**:
  - Vector gốc: ID = `chunk_id * 10`
  - Vector câu hỏi giả định: ID = `chunk_id * 10 + index` (ví dụ: `10 + 1`, `10 + 2`, `10 + 3`)
  - Khi tìm kiếm, kết quả được chia lấy nguyên cho 10 (`matched_id // 10`) để quy về chunk gốc và tiến hành lọc trùng.
- **Ưu điểm**: Khớp nối dạng **Câu hỏi - Câu hỏi** hiệu quả cao, **không tăng latency runtime** vì quá trình sinh câu hỏi đã thực hiện từ trước.
- **Hạn chế**: Tăng dung lượng chỉ mục vector gấp 4 lần và tốn token trong quá trình nạp.

---

## 2. Kết quả Đánh giá Thực nghiệm (k=5)

Tất cả các thử nghiệm được tiến hành trên tập dữ liệu gồm **20 câu hỏi đánh giá** (10 dễ, 10 khó) của tài liệu `mô_hình_hóa_ĐATN`. Định nhãn liên quan sử dụng phương pháp LLM-as-judge (TREC pooling).

| Cấu hình | Recall@5 | MRR@5 | Hit@5 | Runtime Latency (Search) | Phí Token Runtime |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline (Standard)** | 0.825 | 0.950 | 1.000 | **Cực thấp** (<10ms) | Không tốn |
| **2. HyDE Only** | **0.832** | **0.975** | **1.000** | Trung bình (~500ms - 1s) | Tốn (~150 tokens/query) |
| **3. HyPe Only** | 0.825 | **0.975** | **1.000** | **Cực thấp** (<10ms) | Không tốn |
| **4. HyDE + HyPe Combined** | 0.793 | 0.967 | 1.000 | Trung bình (~500ms - 1s) | Tốn (~150 tokens/query) |

### Phân tích số liệu:
* **HyDE mang lại Recall@5 cao nhất (83.2%)** và tăng MRR@5 lên **97.5%**. Câu trả lời giả định hỗ trợ truy hồi các phân đoạn học thuật có độ phức tạp cao tốt hơn.
* **HyPe đạt MRR@5 tối đa tương đương HyDE (97.5%)** mà **không phát sinh bất kỳ độ trễ nào ở runtime**. Điều này cực kỳ lý tưởng cho các ứng dụng yêu cầu thời gian phản hồi tức thì.
* **Kết hợp cả hai (HyDE + HyPe) làm giảm hiệu năng (Recall giảm xuống 79.3%)**. Nguyên nhân do sự không tương thích phong cách diễn đạt giữa câu trả lời giả định (mang tính giải thích) và các câu hỏi giả định (mang tính chất truy vấn) trong không gian vector.

---

## 3. Khuyến nghị Lựa chọn cho Product

Dựa trên kết quả đo đạc thực tế, chúng tôi đề xuất cấu hình hệ thống theo hai kịch bản:

### Kịch bản A: Ưu tiên chất lượng câu trả lời cao nhất (Cấu hình mặc định hiện tại)
* **Thiết lập**: Bật HyDE, Tắt HyPe (`USE_HYDE=true`, `USE_HYPE=false` trong `.env`).
* **Phù hợp**: Các hệ thống hỏi đáp chuyên sâu, tài liệu kỹ thuật phức tạp (như luận văn, tài liệu y tế) nơi độ chính xác thông tin được đặt lên hàng đầu và người dùng chấp nhận độ trễ nhỏ để nhận được trích dẫn grounded chính xác nhất.

### Kịch bản B: Ưu tiên tốc độ và tối ưu chi phí vận hành (Cấu hình Low-Latency)
* **Thiết lập**: Tắt HyDE, Bật HyPe (`USE_HYDE=false`, `USE_HYPE=true` trong `.env`).
* **Phù hợp**: Các hệ thống chat thương mại, trợ lý ảo thời gian thực cần phản hồi nhanh dưới 100ms và cần tối ưu chi phí API OpenAI ở runtime.

---

## 4. Hướng dẫn Thay đổi Cấu hình

Mở file [backend/.env](file:///home/lam/code/TeamHN-RAG-Agent/backend/.env) và cập nhật các tham số sau:

```env
# Kịch bản A: HyDE (Chất lượng cao nhất - Mặc định)
USE_HYDE=true
USE_HYPE=false

# Kịch bản B: HyPe (Tốc độ tối đa, không trễ runtime)
# USE_HYDE=false
# USE_HYPE=true
# HYPE_NUM_QUESTIONS=3
```

> **Lưu ý:** Mỗi khi chuyển trạng thái bật/tắt `USE_HYPE` từ `false` sang `true`, bạn cần xoá các file chỉ mục cũ để hệ thống tự động sinh và nạp câu hỏi giả định vào cơ sở dữ liệu:
> ```bash
> rm -f backend/storage/vector.tvim backend/storage/bm25.pkl
> ```
