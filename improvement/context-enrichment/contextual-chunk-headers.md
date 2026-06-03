Kỹ thuật được đề cập chi tiết trong tài liệu bạn cung cấp là **Contextual Chunk Headers (CCH)** (tạm dịch: Tiêu đề đoạn văn bản theo ngữ cảnh). Đây là một phương pháp tối ưu hóa cho hệ thống RAG (Retrieval-Augmented Generation) nhằm cải thiện độ chính xác của việc truy xuất và thế hệ nội dung của mô hình ngôn ngữ lớn (LLM).

Dưới đây là giải thích chi tiết về kỹ thuật này:

### 1. Khái niệm và Mục tiêu
Thông thường, khi xây dựng hệ thống RAG, các tài liệu dài được chia thành các đoạn nhỏ (chunks). Tuy nhiên, các đoạn này thường gặp vấn đề là **thiếu ngữ cảnh**. Ví dụ, một đoạn văn có thể dùng đại từ "nó" hoặc "họ" mà không nhắc lại chủ thể, hoặc nội dung của nó chỉ có ý nghĩa khi đặt trong một chương cụ thể.

**CCH** giải quyết vấn đề này bằng cách tạo ra các tiêu đề chứa ngữ cảnh cấp cao (như tên tài liệu hoặc tên chương) và **chèn trực tiếp vào phía trước** mỗi đoạn văn bản trước khi thực hiện nhúng (embedding).

### 2. Các thành phần chính của một Tiêu đề Ngữ cảnh
Một tiêu đề (header) có thể bao gồm các thông tin sau để làm phong phú dữ liệu:
*   **Tiêu đề tài liệu (Document Title):** Đây là thông tin quan trọng nhất.
*   **Tóm tắt ngắn gọn tài liệu:** Giúp mô hình hiểu nội dung tổng quát của toàn bộ tệp tin.
*   **Cấu trúc chương/mục:** Bao gồm tiêu đề của phần và phần phụ để xử lý các truy vấn về các chủ đề lớn hơn.

### 3. Quy trình thực hiện kỹ thuật
Kỹ thuật CCH được triển khai qua các bước sau:
*   **Tạo ngữ cảnh (Context Generation):** Sử dụng LLM để tạo ra một tiêu đề mô tả hoặc tóm tắt ngắn cho tài liệu (nếu tài liệu chưa có tiêu đề rõ ràng).
*   **Ghép nối (Concatenation):** Văn bản dùng để nhúng sẽ là sự kết hợp giữa: `Tiêu đề ngữ cảnh + Nội dung đoạn văn`.
*   **Nhúng và Lưu trữ:** Thực hiện nhúng chuỗi văn bản đã ghép nối này vào cơ sở dữ liệu vector.
*   **Truy xuất và Xếp hạng:** Khi người dùng đặt câu hỏi, hệ thống sẽ tìm kiếm dựa trên các đoạn văn đã có tiêu đề. Nếu sử dụng thêm bộ lọc xếp hạng (reranker), bạn cũng phải dùng chuỗi văn bản đã ghép nối này để đảm bảo tính nhất quán.
*   **Hiển thị kết quả:** Khi gửi kết quả truy xuất cho LLM để tạo câu trả lời, việc bao gồm cả tiêu đề ngữ cảnh sẽ giúp LLM ít bị hiểu sai ý nghĩa của đoạn văn hơn.

### 4. Hiệu quả thực tế
Tài liệu cung cấp một ví dụ minh họa cụ thể:
*   Một đoạn văn nói về tác động của biến đổi khí hậu đối với một tổ chức nhưng không nhắc tên "Nike". Khi người dùng hỏi về "Nike", điểm tương quan chỉ đạt **0.1**.
*   Sau khi thêm tiêu đề tài liệu (có tên Nike) vào đoạn văn đó, điểm tương quan tăng vọt lên **0.92**.

Dựa trên đánh giá qua bộ tiêu chuẩn KITE (Knowledge-Intensive Task Evaluation), kỹ thuật CCH giúp tăng điểm hiệu suất trung bình từ **4.72 lên 6.04** (tăng khoảng 27.9%). Đặc biệt trong các lĩnh vực chuyên sâu như tài chính (FinanceBench), sự kết hợp giữa CCH và các kỹ thuật khác có thể nâng điểm số từ 19% lên đến 83%.

**Tóm lại**, CCH là một kỹ thuật đơn giản nhưng cực kỳ hiệu quả để đảm bảo các đoạn văn bản nhỏ luôn mang theo "linh hồn" và ngữ cảnh của toàn bộ tài liệu, từ đó giúp hệ thống AI thông minh và chính xác hơn.

https://www.linkedin.com/pulse/dsrag-công-cụ-truy-vấn-dữ-liệu-phi-cấu-trúc-cho-ứng-dụng-ngoc-phan-dqilc