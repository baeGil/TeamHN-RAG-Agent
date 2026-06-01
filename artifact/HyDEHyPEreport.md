# Báo cáo Tối ưu hóa Retrieval: HyDE & HyPe

Hệ thống RAG Agent đã được tích hợp hai phương pháp tối ưu hóa truy hồi nâng cao:
1. **HyDE (Hypothetical Document Embeddings)**: Sinh câu trả lời giả định trước khi biểu diễn vector truy hồi.
2. **HyPe (Hypothetical Prompt Embeddings)**: Sinh câu hỏi giả định offline cho mỗi chunk trong giai đoạn ingestion/indexing và tìm kiếm dạng câu hỏi-câu hỏi.

Dưới đây là kết quả đánh giá chi tiết của các cấu hình trên tập dữ liệu gồm **20 câu hỏi đánh giá (10 dễ, 10 khó)** của tài liệu `mô_hình_hóa_ĐATN`.

---

## 1. Kết quả Đánh giá So sánh (k=5)

| Cấu hình RAG | Phương pháp | Recall@5 | MRR@5 | Hit@5 | Ghi chú |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Cấu hình Baseline**<br>*(Standard Dense + BM25)* | BM25<br>Dense<br>**Hybrid (RRF)** | 0.738<br>0.798<br>**0.825** | 0.850<br>0.887<br>**0.950** | 0.950<br>1.000<br>**1.000** | Cấu hình mặc định ban đầu. |
| **Cấu hình HyDE**<br>*(USE_HYDE=true)* | BM25<br>Dense<br>**Hybrid (RRF)** | 0.738<br>0.796<br>**0.832** | 0.850<br>0.900<br>**0.975** | 0.950<br>1.000<br>**1.000** | **Tốt nhất về cả Recall và MRR**.<br>Nội dung giả định giúp khớp ngữ nghĩa tốt hơn. |
| **Cấu hình HyPe**<br>*(USE_HYPE=true)* | BM25<br>Dense<br>**Hybrid (RRF)** | 0.738<br>0.733<br>**0.825** | 0.850<br>0.885<br>**0.975** | 0.950<br>1.000<br>**1.000** | **Cải thiện MRR** của Hybrid ngang HyDE.<br>Tốc độ truy hồi nhanh hơn HyDE (không tốn LLM ở query-time). |
| **Cấu hình HyDE + HyPe**<br>*(Cả hai hoạt động)* | BM25<br>Dense<br>**Hybrid (RRF)** | 0.738<br>0.750<br>**0.793** | 0.850<br>0.829<br>**0.967** | 0.950<br>0.950<br>**1.000** | Hiệu năng bị giảm nhẹ do stylistic mismatch giữa câu trả lời giả định (HyDE) và câu hỏi giả định (HyPe). |

---

## 2. Phân tích chi tiết từng kỹ thuật

### 2.1. HyDE (Hypothetical Document Embeddings)
- **Cơ chế**: Khi người dùng hỏi, hệ thống gửi câu hỏi qua LLM (`gpt-4o-mini`) để tạo câu trả lời giả thuyết dài 2-3 câu. Vector của câu trả lời giả thuyết này sau đó được sử dụng cho Dense retrieval.
- **Ưu điểm**:
  - Tăng **Recall@5** của Hybrid RRF từ **82.5% lên 83.2%**.
  - Tăng **MRR@5** của Hybrid RRF từ **95.0% lên 97.5%**.
  - Giúp giải quyết các câu hỏi phức tạp (hard) đòi hỏi suy luận chéo, vì LLM tạo câu trả lời giả định có khả năng chứa các từ khoá chuyên ngành xuất hiện trong văn bản gốc.
- **Hạn chế**: Tăng latency truy hồi do phải chờ LLM sinh câu trả lời trước khi thực hiện tìm kiếm vector (~0.5 - 1.2s tuỳ tốc độ mạng).

### 2.2. HyPe (Hypothetical Prompt Embeddings / doc2query)
- **Cơ chế**: Trong giai đoạn indexing/rebuild, LLM sinh ra **3 câu hỏi giả định** mà người dùng có thể dùng để hỏi thông tin của chunk này. Các câu hỏi được lưu vào SQLite (`hype_questions` trong bảng `chunks`). Mỗi câu hỏi này được embed độc lập và đưa vào `VectorIndex`.
- **Giải pháp kỹ thuật cho `turbovec`**: `turbovec` yêu cầu ID trong chỉ mục là duy nhất (`ValueError: id already present`). Để hỗ trợ gán nhiều câu hỏi cho cùng một chunk, chúng tôi triển khai kỹ thuật **ID Mapping**:
  - Chunk ID gốc: `cid` (ví dụ: `1`).
  - Embedding của văn bản gốc: được đánh ID `cid * 10` (ví dụ: `10`).
  - Embedding của các câu hỏi giả định: đánh ID `cid * 10 + 1`, `cid * 10 + 2`, `cid * 10 + 3` (ví dụ: `11`, `12`, `13`).
  - Khi `VectorIndex.search` tìm được các ID này, nó thực hiện ánh xạ ngược `cid = matched_id // 10` và lọc trùng để lấy top-k chunk gốc duy nhất.
- **Ưu điểm**:
  - Không có runtime overhead / không tăng latency truy hồi (tất cả các câu hỏi được sinh và embed offline).
  - Tăng **MRR@5** của Hybrid RRF lên **97.5%** nhờ việc so khớp trực tiếp giữa câu hỏi của người dùng và các câu hỏi giả định trong cơ sở dữ liệu.
- **Hạn chế**: Làm tăng kích thước chỉ mục vector gấp 4 lần và tốn token/thời gian trong giai đoạn nạp tài liệu (ingestion).

---

## 3. Đề xuất cấu hình sản phẩm

1. **Nếu ưu tiên chất lượng câu trả lời cao nhất**: Khuyên dùng cấu hình **HyDE (USE_HYDE=true)**. Điểm MRR@5 và Recall@5 là cao nhất trên bộ kiểm thử.
2. **Nếu ưu tiên độ trễ (latency) thấp và tiết kiệm chi phí runtime**: Khuyên dùng cấu hình **HyPe (USE_HYPE=true)**. Phương pháp này mang lại điểm MRR@5 tương đương HyDE (97.5%) nhưng chạy ở thời gian thực mà không phát sinh thêm bất kỳ chi phí hay trễ LLM nào ở query-time.
3. **Không khuyến nghị bật cả hai cùng lúc**: Sự kết hợp giữa câu trả lời giả định của HyDE và câu hỏi giả định của HyPe làm lệch không gian ngữ nghĩa khớp nối và làm giảm chất lượng truy hồi.
