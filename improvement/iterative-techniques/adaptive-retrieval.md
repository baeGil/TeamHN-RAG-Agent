Kỹ thuật được đề cập trong tài liệu là **Hệ thống Tạo tăng cường Truy xuất Thích ứng (Adaptive Retrieval-Augmented Generation - RAG)**. Đây là một phương pháp RAG tiên tiến, thay vì sử dụng một cách tiếp cận duy nhất cho mọi truy vấn, hệ thống sẽ tự động điều chỉnh chiến lược truy xuất dựa trên loại câu hỏi của người dùng.

Dưới đây là chi tiết về cách hoạt động của kỹ thuật này:

### 1. Phân loại truy vấn (Query Classification)
Hệ thống bắt đầu bằng việc phân loại truy vấn của người dùng vào một trong bốn nhóm chính:
*   **Factual (Sự thật):** Các câu hỏi tìm kiếm thông tin cụ thể, có thể xác minh được.
*   **Analytical (Phân tích):** Các truy vấn yêu cầu giải thích hoặc phân tích toàn diện.
*   **Opinion (Ý kiến):** Các câu hỏi về các vấn đề chủ quan hoặc tìm kiếm các quan điểm khác nhau.
*   **Contextual (Ngữ cảnh):** Các truy vấn phụ thuộc vào ngữ cảnh riêng biệt của người dùng.

### 2. Các chiến lược truy xuất thích ứng
Mỗi loại truy vấn sẽ kích hoạt một quy trình truy xuất riêng biệt:

*   **Chiến lược Factual:** Sử dụng mô hình ngôn ngữ lớn (LLM) để làm phong phú truy vấn ban đầu nhằm tăng độ chính xác, sau đó truy xuất và dùng LLM để xếp hạng các tài liệu theo mức độ liên quan.
*   **Chiến lược Analytical:** LLM sẽ tạo ra nhiều truy vấn phụ (sub-queries) để bao quát các khía cạnh khác nhau của vấn đề. Sau đó, hệ thống truy xuất tài liệu cho từng câu hỏi phụ và đảm bảo tính đa dạng trong việc lựa chọn tài liệu cuối cùng.
*   **Chiến lược Opinion:** LLM xác định các quan điểm khác nhau về chủ đề, truy xuất các tài liệu đại diện cho từng quan điểm đó để cung cấp một cái nhìn đa chiều.
*   **Chiến lược Contextual:** Tích hợp ngữ cảnh cụ thể của người dùng vào truy vấn thông qua LLM. Việc truy xuất và xếp hạng sẽ xem xét cả mức độ liên quan lẫn thông tin cá nhân của người dùng.

### 3. Xếp hạng nâng cao và Tạo phản hồi
Sau khi truy xuất, hệ thống thực hiện một bước **xếp hạng cuối cùng bằng LLM** để đảm bảo chọn ra những tài liệu phù hợp nhất. Cuối cùng, tập hợp các tài liệu này được chuyển đến mô hình OpenAI GPT để tạo ra câu trả lời hoàn chỉnh dựa trên ngữ cảnh đã thu thập được.

### Lợi ích của phương pháp này:
*   **Độ chính xác cao hơn:** Nhờ tùy chỉnh chiến lược cho từng loại câu hỏi.
*   **Tính linh hoạt:** Xử lý được nhiều nhu cầu khác nhau của người dùng.
*   **Đa dạng góc nhìn:** Đặc biệt hữu ích cho các câu hỏi về ý kiến.
*   **Cá nhân hóa:** Hiểu rõ ngữ cảnh của người dùng để đưa ra phản hồi phù hợp.
*   **Phân tích chuyên sâu:** Đảm bảo các chủ đề phức tạp được khám phá kỹ lưỡng qua chiến lược phân tích.

https://colab.research.google.com/github/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/adaptive_retrieval.ipynb