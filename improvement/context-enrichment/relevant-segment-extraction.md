Kỹ thuật được đề cập trong tài liệu của bạn là **Relevant Segment Extraction (RSE)** - Trích xuất Phân đoạn Liên quan. Đây là một phương pháp hậu xử lý trong hệ thống RAG (Retrieval-Augmented Generation) nhằm mục đích tái cấu trúc lại các đoạn văn bản dài và liên tục từ những "chunk" (mảnh văn bản) riêng lẻ đã được truy xuất.

Dưới đây là giải thích chi tiết về cách hoạt động và lý do tại sao kỹ thuật này hiệu quả:

### 1. Động lực và Ý tưởng cốt lõi
Trong các hệ thống RAG truyền thống, việc chọn kích thước chunk thường là một sự đánh đổi: chunk nhỏ thì truy xuất chính xác nhưng thiếu ngữ cảnh, chunk lớn thì giàu ngữ cảnh nhưng khó tìm kiếm chính xác. RSE giải quyết vấn đề này dựa trên một quan sát đơn giản: **các thông tin liên quan thường có xu hướng nằm tập trung (clustered) cùng nhau** trong tài liệu gốc.

### 2. Các bước triển khai kỹ thuật
Quy trình của RSE diễn ra sau bước tìm kiếm vector (và sau cả bước xếp hạng lại - reranking) nhưng trước khi đưa dữ liệu vào LLM.

*   **Phân mảnh tài liệu (Chunking):** Tài liệu cần được chia nhỏ nhưng **không được chồng lấp (no overlap)**. Điều này cho phép hệ thống dễ dàng ghép nối các chunk lại thành một đoạn văn bản liên tục bằng cách nối các chunk có chỉ số (index) liền kề.
*   **Hệ thống lưu trữ Key-Value:** Ngoài cơ sở dữ liệu vector, RSE cần một kho lưu trữ key-value (sử dụng `doc_id` và `chunk_index` làm khóa) để có thể nhanh chóng lấy ra nội dung của các chunk không nằm trong kết quả tìm kiếm ban đầu nhưng lại nằm giữa các chunk liên quan.
*   **Tính toán giá trị Chunk (Chunk Scoring):**
    *   Đầu tiên, hệ thống kết hợp điểm tương đồng (similarity score) và thứ hạng (rank) để tạo ra một giá trị tin cậy hơn là chỉ dùng một trong hai.
    *   Sau đó, một giá trị ngưỡng gọi là **irrelevant_chunk_penalty** (thường là 0.2) sẽ được trừ đi từ điểm số của mỗi chunk. Những chunk không liên quan sẽ có giá trị âm, trong khi chunk liên quan có giá trị dương.
*   **Tối ưu hóa phân đoạn (Optimization):** Việc tìm kiếm phân đoạn tốt nhất giờ đây trở thành biến thể của bài toán **"Maximum Sum Subarray"** (tìm dãy con có tổng lớn nhất). Mục tiêu là tìm ra chuỗi các chunk liên tiếp mà khi cộng lại có tổng giá trị cao nhất. Quá trình này rất nhanh, thường chỉ mất khoảng 5-10ms.

### 3. Ưu điểm vượt trội
*   **Khôi phục ngữ cảnh bị mất:** RSE có khả năng tự động "điền vào chỗ trống". Nếu chunk A và chunk C rất liên quan nhưng chunk B ở giữa lại không được hệ thống tìm kiếm đánh giá cao, RSE vẫn sẽ bao gồm cả chunk B để đảm bảo tính liên tục và đầy đủ của ngữ cảnh cho LLM.
*   **Linh hoạt theo truy vấn:** Hệ thống có thể trả về các chunk đơn lẻ cho các câu hỏi ngắn về sự thật, hoặc các chương/đoạn dài cho các câu hỏi mang tính tổng hợp cao.
*   **Hiệu suất thực tế:** Qua đánh giá trên benchmark KITE, RSE giúp tăng hiệu suất trung bình lên **42.6%** so với phương pháp Top-k truyền thống. Đặc biệt trong lĩnh vực tài chính (FinanceBench), khi kết hợp với Contextual Chunk Headers, nó mang lại sự cải thiện rõ rệt về độ chính xác.

Nói một cách ngắn gọn, RSE biến kết quả truy xuất từ "một danh sách các mảnh vụn" thành "một vài đoạn văn bản có ý nghĩa", giúp LLM hiểu sâu hơn và trả lời chính xác hơn.

https://colab.research.google.com/github/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/relevant_segment_extraction.ipynb