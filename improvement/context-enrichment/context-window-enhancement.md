Kỹ thuật bạn đang tìm hiểu được gọi là **Context Enrichment Window for Document Retrieval** (Cửa sổ làm giàu ngữ cảnh cho truy xuất tài liệu). Đây là một phương pháp nâng cao trong hệ thống RAG (Retrieval-Augmented Generation) nhằm giải quyết vấn đề các đoạn văn bản (chunks) bị rời rạc và thiếu thông tin khi truy xuất bằng vector search truyền thống.

Dưới đây là giải thích chi tiết về cách thức hoạt động của kỹ thuật này dựa trên nguồn tài liệu:

### 1. Động lực và Mục tiêu
Trong các hệ thống tìm kiếm vector thông thường, tài liệu thường được chia nhỏ thành các đoạn độc lập. Khi truy xuất, hệ thống chỉ lấy ra các đoạn có độ tương đồng ngữ nghĩa cao nhất, nhưng những đoạn này thường **thiếu ngữ cảnh cần thiết** (ví dụ: thông tin ở câu trước hoặc câu sau) để mô hình ngôn ngữ (LLM) có thể hiểu đầy đủ vấn đề. Kỹ thuật này nhằm mục đích tạo ra một cái nhìn toàn diện hơn bằng cách bao gồm cả các đoạn văn bản lân cận.

### 2. Quy trình thực hiện chi tiết
Kỹ thuật này bao gồm 4 giai đoạn chính:

*   **Tiền xử lý tài liệu (Document Preprocessing):**
    *   Tài liệu (ví dụ: file PDF) được đọc và chuyển đổi thành một chuỗi văn bản dài.
    *   Văn bản được chia thành các đoạn (chunks) có độ chồng lấp (overlap) nhất định.
    *   **Điểm mấu chốt:** Mỗi đoạn văn bản khi được tạo ra sẽ được gắn thêm **metadata là chỉ mục thời gian (chronological index)**, cho biết vị trí chính xác của đoạn đó trong tài liệu gốc.

*   **Tạo kho lưu trữ Vector (Vector Store Creation):**
    *   Sử dụng các mô hình nhúng (như OpenAI embeddings) để chuyển đổi các đoạn văn bản thành vector.
    *   Các vector này cùng với metadata (chỉ mục) được lưu trữ vào một cơ sở dữ liệu vector như FAISS.

*   **Truy xuất làm giàu ngữ cảnh (Context-Enriched Retrieval):**
    *   Khi có một truy vấn, hệ thống trước tiên sẽ tìm các đoạn văn bản có độ tương đồng ngữ nghĩa cao nhất.
    *   Thay vì chỉ trả về kết quả đó, hàm xử lý (`retrieve_with_context_overlap`) sẽ dựa vào chỉ mục của đoạn văn bản để **lấy thêm các đoạn "hàng xóm"** (neighbors) ở ngay trước và sau nó.
    *   Số lượng đoạn hàng xóm này (`num_neighbors`) có thể được điều chỉnh linh hoạt tùy theo nhu cầu.

*   **Hợp nhất và Xử lý chồng lấp:**
    *   Các đoạn văn bản (đoạn gốc và các đoạn hàng xóm) được nối lại với nhau.
    *   Hệ thống sẽ tính toán và xử lý phần nội dung chồng lấp giữa các đoạn để tạo ra một "cửa sổ ngữ cảnh" mở rộng, liền mạch và không bị lặp từ.

### 3. Lợi ích của kỹ thuật
*   **Tăng tính mạch lạc:** Cung cấp thông tin đầy đủ và có hệ thống hơn cho LLM, tránh tình trạng câu trả lời bị cắt vụn.
*   **Độ chính xác cao hơn:** Việc hiểu rõ ngữ cảnh xung quanh giúp mô hình đưa ra các câu trả lời chính xác hơn cho các tác vụ hạ nguồn như trả lời câu hỏi.
*   **Linh hoạt:** Người dùng có thể dễ dàng điều chỉnh kích thước cửa sổ ngữ cảnh (số lượng đoạn lân cận) để cân bằng giữa độ chi tiết và giới hạn token của mô hình.

Tóm lại, kỹ thuật này giúp duy trì ưu điểm của tìm kiếm vector nhưng khắc phục được xu hướng trả về các mảnh văn bản cô lập, từ đó cải thiện chất lượng tổng thể của hệ thống RAG.

https://colab.research.google.com/github/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/context_enrichment_window_around_chunk.ipynb