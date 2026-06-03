**Dartboard RAG** (viết tắt của Retrieval-Augmented Generation với sự cân bằng giữa Tính liên quan và Tính đa dạng) là một kỹ thuật cải tiến quy trình truy xuất dữ liệu nhằm giải quyết vấn đề **dư thừa thông tin** trong các hệ thống RAG truyền thống.

Dưới đây là giải thích chi tiết về kỹ thuật này dựa trên các nguồn tài liệu:

### 1. Động lực và Mục tiêu
Trong các cơ sở dữ liệu lớn và dày đặc, việc truy xuất "top-k" tài liệu dựa trên độ tương đồng thuần túy thường dẫn đến kết quả là nhiều tài liệu có nội dung lặp lại hoặc cực kỳ giống nhau. Điều này làm lãng phí không gian ngữ cảnh (context window) của mô hình ngôn ngữ lớn (LLM) và tạo ra hiệu ứng "phòng hồi thanh" (echo chamber).

**Dartboard RAG** giải quyết vấn đề này bằng cách tối ưu hóa đồng thời hai yếu tố:
*   **Tính liên quan (Relevance):** Tài liệu phải trả lời đúng câu hỏi của người dùng.
*   **Tính đa dạng (Diversity):** Các tài liệu được chọn phải mang lại những thông tin khác biệt, không trùng lặp với những tài liệu đã được chọn trước đó.

### 2. Quy trình hoạt động của thuật toán
Kỹ thuật này sử dụng một thuật toán tìm kiếm tham lam (Greedy Search) để tinh lọc danh sách tài liệu. Quy trình gồm các bước sau:

1.  **Truy xuất ứng viên ban đầu:** Sử dụng các phương pháp thông thường (như Cosine similarity hoặc BM25) để lấy ra một tập hợp $N$ tài liệu ứng viên tiềm năng nhất.
2.  **Lựa chọn tài liệu đầu tiên:** Tài liệu có độ liên quan cao nhất với truy vấn sẽ được chọn đầu tiên.
3.  **Lặp lại việc chọn lọc có điều kiện:** Để chọn các tài liệu tiếp theo, thuật toán tính toán một điểm số kết hợp:
    *   **Điểm liên quan:** Độ khớp của tài liệu với truy vấn gốc.
    *   **Hình phạt tương đồng:** Những tài liệu quá giống với các tài liệu đã được chọn trước đó sẽ bị giảm điểm (bị phạt).
4.  **Hoàn tất:** Quá trình này lặp lại cho đến khi đạt đủ số lượng $k$ tài liệu cần thiết.

### 3. Các thành phần và Tham số chính
Để kiểm soát sự cân bằng giữa các yếu tố, Dartboard RAG sử dụng các tham số sau:
*   **RELEVANCE_WEIGHT:** Trọng số xác định mức độ ưu tiên cho tính liên quan đến truy vấn.
*   **DIVERSITY_WEIGHT:** Trọng số xác định mức độ quan trọng của sự khác biệt so với các lựa chọn hiện có.
*   **SIGMA:** Một tham số làm mịn (smoothing parameter) dùng trong quá trình chuyển đổi khoảng cách thành xác suất.

### 4. Khả năng tích hợp linh hoạt
Kỹ thuật này không chỉ áp dụng cho tìm kiếm vector đơn thuần mà còn hỗ trợ:
*   **Hybrid / Fusion Retrieval:** Kết hợp cả tìm kiếm dày đặc (dense) và thưa thớt (sparse/BM25) bằng cách hợp nhất các độ tương đồng thành một khoảng cách duy nhất (ví dụ: 1 - trung bình cộng của cosine similarity).
*   **Cross-Encoders:** Sử dụng trực tiếp điểm số tương đồng từ cross-encoder để điều chỉnh việc chọn lọc.

### 5. Lợi ích cuối cùng
Bằng cách tích hợp cả tính liên quan và tính đa dạng, **Dartboard RAG** đảm bảo rằng các tài liệu được đưa vào ngữ cảnh của LLM sẽ cung cấp thông tin phong phú và toàn diện hơn, từ đó dẫn đến các câu trả lời có chất lượng cao hơn trong các hệ thống AI. Ví dụ, khi tìm kiếm tin tức, nó sẽ ưu tiên các bài báo vừa đúng chủ đề vừa cung cấp thêm thông tin mới thay vì lặp lại cùng một nội dung từ nhiều nguồn khác nhau.

https://colab.research.google.com/github/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/dartboard.ipynb#scrollTo=zlQVqajV071L